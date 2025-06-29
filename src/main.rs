mod hf;
mod model;
use std::path::PathBuf;

use hf_hub::api::sync::{Api, ApiError};
use image::imageops::FilterType;
use ort::{
    environment::Environment,
    session::{Session, builder::GraphOptimizationLevel},
};
use tokenizers::{Tokenizer, TokenizerBuilder};

use crate::model::{Decoder, Encoder};

struct ModelDir {
    encoder_path: PathBuf,
    decoder_path: PathBuf,
    tokenizer_config_path: PathBuf,
}

impl ModelDir {
    fn new(paths: &Vec<PathBuf>) -> anyhow::Result<Self> {
        let mut encoder_path = None;
        let mut decoder_path = None;
        let mut tokenizer_config_path = None;

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
        }

        if encoder_path.is_none() || decoder_path.is_none() || tokenizer_config_path.is_none() {
            return Err(anyhow::anyhow!(
                "Missing encoder, decoder, or tokenizer model"
            ));
        }

        Ok(Self {
            encoder_path: encoder_path.unwrap(),
            decoder_path: decoder_path.unwrap(),
            tokenizer_config_path: tokenizer_config_path.unwrap(),
        })
    }
}

fn main() -> anyhow::Result<()> {
    ort::init().with_name("test").commit()?;

    let paths = hf::pull_model("l0wgear/manga-ocr-2025-onnx")?;
    let model_files = ModelDir::new(&paths)?;

    let tokenizer = Tokenizer::from_file(model_files.tokenizer_config_path);
    if tokenizer.is_err() {
        return Err(anyhow::anyhow!("Failed to load tokenizer"));
    }

    let mut tokenizer = tokenizer.unwrap();

    let end_tokens = tokenizer.encode("[SEP]", false).unwrap();
    let end_token_ids = end_tokens.get_ids();
    println!("{:?}", end_token_ids);

    let s = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_files.encoder_path)?;
    let mut encoder = Encoder::new(s);
    let s = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(model_files.decoder_path)?;
    let decoder = Decoder::new(s, end_token_ids);

    let img = image::ImageReader::open("./test1.png")?.decode()?;

    let out = encoder.encode(img)?;
    // println!("{:?}", out);
    let dec_out = decoder.decode(out, 300)?;
    // println!("{:?}", dec_out);

    let idx: Vec<u32> = dec_out.iter().map(|i| *i as u32).collect();
    let decoded = tokenizer.decode(&idx, true);

    if decoded.is_err() {
        return Err(anyhow::anyhow!("Failed to decode"));
    }
    let decoded = decoded.unwrap().replace(" ", "");

    println!("{:?}", decoded);

    Ok(())
}
