use std::cell::RefCell;
use std::path::PathBuf;

use image::DynamicImage;
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array4, ArrayD, Slice, s};
use ndarray::{ArrayView, ShapeError};
use ndarray_stats::QuantileExt;
use ort::{
    session::Session,
    value::{Tensor, TensorRef},
};
use std::error::Error;
use std::fmt::Display;
use tokenizers::Tokenizer;

use crate::config::GenerationConfig;
use crate::hf::pull_model;

fn rgb_to_array(img: &RgbImage) -> Result<Array4<f32>, ShapeError> {
    let (width, height) = img.dimensions();
    let raw_data = img.as_raw();

    // The shape of the array is (rows, columns, channels)
    // which corresponds to (height, width, 3)
    let shape = (1, height as usize, width as usize, 3);

    // Create a non-owning view of the flat image data
    let view: ArrayView<u8, _> =
        ArrayView::from_shape(shape, raw_data)?.permuted_axes([0, 3, 1, 2]);

    // Create a new, owned array by mapping every element from u8 to f32
    Ok(view.mapv(|x| x as f32).as_standard_layout().into_owned())
}

struct ModelDir {
    encoder_path: PathBuf,
    decoder_path: PathBuf,
    tokenizer_config_path: PathBuf,
    gen_config_path: PathBuf,
}

impl ModelDir {
    fn new(paths: &[PathBuf]) -> anyhow::Result<Self> {
        let mut encoder_path = None;
        let mut decoder_path = None;
        let mut tokenizer_config_path = None;
        let mut gen_config_path = None;

        for p in paths.iter() {
            if p.ends_with("encoder_model.onnx") {
                encoder_path = Some(p.clone());
            }
            if p.ends_with("decoder_model.onnx") {
                decoder_path = Some(p.clone());
            }
            if p.ends_with("tokenizer.json") {
                tokenizer_config_path = Some(p.clone());
            }
            if p.ends_with("generation_config.json") {
                gen_config_path = Some(p.clone());
            }
        }

        if encoder_path.is_none()
            || decoder_path.is_none()
            || tokenizer_config_path.is_none()
            || gen_config_path.is_none()
        {
            return Err(anyhow::anyhow!(
                "Missing encoder, decoder model / tokenizer or generation config"
            ));
        }

        Ok(Self {
            encoder_path: encoder_path.unwrap(),
            decoder_path: decoder_path.unwrap(),
            tokenizer_config_path: tokenizer_config_path.unwrap(),
            gen_config_path: gen_config_path.unwrap(),
        })
    }
}

pub struct OCRModel {
    encoder: Encoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
}

impl OCRModel {
    pub fn from_name_or_path(name_or_path: &str) -> anyhow::Result<Self> {
        ort::init().with_name("OCRModel").commit()?;

        let files = pull_model(name_or_path)?;
        let model_dir = ModelDir::new(&files)?;

        let tokenizer = Tokenizer::from_file(model_dir.tokenizer_config_path)
            .map_err(|_| anyhow::Error::msg("Failed to load tokenizer"))?;

        let gen_config = GenerationConfig::from_file(&model_dir.gen_config_path)?;

        let encoder = Encoder::from_path(&model_dir.encoder_path)?;
        let decoder = Decoder::from_path(&model_dir.decoder_path, gen_config)?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn run(&self, img: &DynamicImage) -> anyhow::Result<String> {
        let out = self.encoder.encode(img)?;
        let dec_out = self.decoder.decode(out, 300)?;

        let idx: Vec<u32> = dec_out.iter().map(|i| *i as u32).collect();
        let decoded = self.tokenizer.decode(&idx, true);

        if decoded.is_err() {
            return Err(anyhow::anyhow!("Failed to decode"));
        }
        let decoded = decoded.unwrap().replace(" ", "");
        Ok(decoded)
    }
}

pub struct Encoder {
    session: RefCell<Session>,
}

impl Encoder {
    pub fn new(session: Session) -> Self {
        Self {
            session: RefCell::new(session),
        }
    }

    pub fn from_path(path: &PathBuf) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        Ok(Self::new(session))
    }

    pub fn encode(&self, input: &DynamicImage) -> anyhow::Result<ArrayD<f32>> {
        let resized = input.resize_exact(224, 224, FilterType::Nearest);
        let resized = resized.to_rgb8();

        let scale = 0.003_921_569;

        let arr = (rgb_to_array(&resized)? * scale - 0.5) / 0.5;

        let mut session = self.session.borrow_mut();

        let outputs = session.run(ort::inputs![TensorRef::from_array_view(&arr)?])?;
        Ok(outputs[0].try_extract_array::<f32>()?.to_owned())
    }
}

#[derive(Debug)]
struct DecodeError(String);

impl Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for DecodeError {}

fn last_token_idx(input: ArrayD<f32>) -> Result<i64, DecodeError> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(DecodeError(String::from(
            "Expected array to have 3 dimensions",
        )));
    }
    let slice = Slice::new(0, None, 1);
    let idx = input.slice(s![0, -1, slice]).argmax().unwrap();
    Ok(idx as i64)
}

pub struct Decoder {
    session: RefCell<Session>,
    gen_config: GenerationConfig,
}

impl Decoder {
    pub fn new(session: Session, gen_config: GenerationConfig) -> Self {
        Self {
            session: RefCell::new(session),
            gen_config,
        }
    }

    pub fn from_path(path: &PathBuf, gen_config: GenerationConfig) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        Ok(Self::new(session, gen_config))
    }

    fn stop_decoding(&self, tokens: &[i64]) -> bool {
        let max_repeats = self.gen_config.no_repeat_ngram_size as usize;
        if tokens.len() < max_repeats {
            return false;
        }
        let mut count = 0;
        for token in tokens.iter().skip(tokens.len() - max_repeats) {
            if *token == self.gen_config.eos_token_id as i64 {
                count += 1;
            }
        }
        count >= max_repeats
    }

    pub fn decode(&self, input: ArrayD<f32>, max_tokens: usize) -> anyhow::Result<Vec<i64>> {
        let mut input_ids = vec![self.gen_config.decoder_start_token_id as i64];
        let mut session = self.session.borrow_mut();
        for _ in 0..max_tokens {
            let input_ref = TensorRef::from_array_view(&input)?;
            let input_ids_tensor = Tensor::from_array(([1, input_ids.len()], input_ids.clone()))?;
            let outputs = session.run(ort::inputs! {
                "encoder_hidden_states" => input_ref,
                "input_ids" => input_ids_tensor,
            })?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();
            let idx = last_token_idx(arr)?;
            input_ids.push(idx);
            if self.stop_decoding(&input_ids) {
                break;
            }
        }
        Ok(input_ids.clone())
    }
}

#[cfg(all(test, feature = "hf-tests"))]
mod tests {
    use super::*;

    #[test]
    fn test_ocr() {
        let img = image::ImageReader::open("tests/fixtures/test.png")
            .unwrap()
            .decode()
            .unwrap();

        let model = OCRModel::from_name_or_path("l0wgear/manga-ocr-2025-onnx").unwrap();
        let output = model.run(&img);
        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(&output, "悪魔との戦い");
    }
}
