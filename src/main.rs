mod hf;
mod model;
use hf_hub::api::sync::{Api, ApiError};
use image::imageops::FilterType;
use ort::{environment::Environment, session::Session};

use crate::model::{Decoder, Encoder};

fn main() -> anyhow::Result<()> {
    ort::init().with_name("test").commit()?;

    let paths = hf::pull_model("l0wgear/manga-ocr-2025-onnx")?;
    let mut encoder_path = None;
    let mut decoder_path = None;

    for p in paths.iter() {
        if p.ends_with("encoder_model.onnx") {
            encoder_path = Some(p.clone());
        }
        if p.ends_with("decoder_model.onnx") {
            decoder_path = Some(p.clone());
        }
    }

    if encoder_path.is_none() || decoder_path.is_none() {
        return Err(anyhow::anyhow!("Missing encoder or decoder model"));
    }

    let s = Session::builder()?.commit_from_file(encoder_path.unwrap())?;
    let mut encoder = Encoder::new(s);
    let s = Session::builder()?.commit_from_file(decoder_path.unwrap())?;
    let mut decoder = Decoder::new(s);

    let img = image::ImageReader::open("./test1.png")?.decode()?;

    let out = encoder.encode(img)?;
    println!("{:?}", out);
    let dec_out = decoder.decode(out);
    println!("{:?}", dec_out);
    Ok(())
}
