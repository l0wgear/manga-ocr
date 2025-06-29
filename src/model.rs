use std::cell::RefCell;
use std::{path::PathBuf, sync::Arc};

use image::{EncodableLayout, RgbImage, imageops::FilterType};
use ndarray::{Array4, ArrayD, ArrayViewD, ArrayViewMut, Slice, s};
use ndarray::{ArrayView, ShapeError};
use ndarray_stats::QuantileExt;
use ort::{
    environment::Environment,
    session::Session,
    value::{Tensor, TensorRef},
};
use std::error::Error;
use std::fmt::Display;

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
    // The `?` operator propagates any error from `from_shape`.
    Ok(view.mapv(|x| x as f32).as_standard_layout().into_owned())
}

pub struct Encoder {
    session: Session,
}

impl Encoder {
    pub fn new(session: Session) -> Self {
        Self { session }
    }

    pub fn encode(&mut self, input: image::DynamicImage) -> anyhow::Result<ArrayD<f32>> {
        // let input = ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
        let resized = input.resize_exact(224, 224, FilterType::Nearest);
        let resized = resized.to_rgb8();

        let scale = 0.00392156862745098;

        let arr = (rgb_to_array(&resized)? * scale - 0.5) / 0.5;

        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(&arr)?])?;
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
    max_end_token_repeats: usize,
    end_token_ids: Vec<i64>,
}

impl Decoder {
    pub fn new(session: Session, end_token_ids: &[u32]) -> Self {
        Self {
            session: RefCell::new(session),
            max_end_token_repeats: 3,
            end_token_ids: end_token_ids.iter().map(|i| *i as i64).collect(),
        }
    }

    fn stop_decoding(&self, tokens: &Vec<i64>) -> bool {
        if tokens.len() < self.max_end_token_repeats {
            return false;
        }
        let mut count = 0;
        for i in tokens.len() - self.max_end_token_repeats..tokens.len() {
            if self.end_token_ids.contains(&(tokens[i])) {
                count += 1;
            }
        }
        count >= self.max_end_token_repeats
    }

    pub fn decode(&self, input: ArrayD<f32>, max_tokens: usize) -> anyhow::Result<Vec<i64>> {
        // let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
        // let mut output_vec = vec![];
        let mut input_ids = vec![2i64];
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
