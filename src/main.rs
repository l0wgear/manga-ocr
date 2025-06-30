mod config;
mod hf;
mod model;

use crate::model::OCRModel;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "l0wgear/manga-ocr-2025-onnx")]
    model: String,

    #[arg(short, long)]
    image: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model = OCRModel::from_name_or_path(&args.model)?;

    let img = image::ImageReader::open(&args.image)?.decode()?;
    let text = model.run(&img);

    match text {
        Ok(text) => println!("{}", text),
        Err(err) => println!("Error: {:?}", err),
    }

    Ok(())
}
