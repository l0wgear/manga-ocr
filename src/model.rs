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

        resized.save("./test1_out.png");

        let scale = 0.00392156862745098;

        let arr = (rgb_to_array(&resized)? * scale - 0.5) / 0.5;

        let outputs = self
            .session
            .run(ort::inputs![TensorRef::from_array_view(&arr)?])?;
        Ok(outputs[0].try_extract_array::<f32>()?.to_owned())
    }
}

pub struct Decoder {
    session: Session,
    max_end_token_repeats: usize,
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

impl Decoder {
    pub fn new(session: Session) -> Self {
        Self {
            session,
            max_end_token_repeats: 3,
        }
    }

    pub fn decode(&mut self, input: ArrayD<f32>) -> anyhow::Result<Vec<i64>> {
        // let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
        // let mut output_vec = vec![];
        let mut input_ids = vec![2i64];
        for _ in 0..100 {
            let input_ref = TensorRef::from_array_view(&input)?;
            let input_ids_tensor = Tensor::from_array(([1, input_ids.len()], input_ids.clone()))?;
            let outputs = self.session.run(ort::inputs! {
                "encoder_hidden_states" => input_ref,
                "input_ids" => input_ids_tensor,
            })?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();

            // let shape = arr.shape();
            // println!("Shape: {:?}", shape);
            // println!("Item: {:?}", arr[[0, 0, 0]]);
            let idx = last_token_idx(arr)?;
            input_ids.push(idx);
            // let idx = arr[shape[0] - 1].argmax()
            // output_vec.push(arr);
        }
        Ok(input_ids.clone())
    }
}
