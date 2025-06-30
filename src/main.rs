mod config;
mod hf;
mod model;

use crate::model::OCRModel;
use arboard::{Clipboard, ImageData};
use clap::{Parser, ValueEnum};
use image::{DynamicImage, ImageBuffer, Rgba};
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::{hash::DefaultHasher, path::PathBuf, thread::sleep, time::Duration};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "l0wgear/manga-ocr-2025-onnx")]
    model: String,

    #[arg(short, long)]
    image: Option<PathBuf>,

    #[arg(long, default_value_t = Mode::Clipboard)]
    mode: Mode,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    File,
    Clipboard,
}
impl Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::File => write!(f, "file"),
            Mode::Clipboard => write!(f, "clipboard"),
        }
    }
}

fn to_dyn_image(arboard_image: ImageData) -> Option<DynamicImage> {
    let image_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(
        arboard_image.width as u32,
        arboard_image.height as u32,
        arboard_image.bytes.into_owned(),
    )?;

    // Convert the ImageBuffer to a DynamicImage
    Some(DynamicImage::ImageRgba8(image_buffer))
}

fn hash(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model = OCRModel::from_name_or_path(&args.model)?;

    match args.mode {
        Mode::File => {
            if let Some(path) = args.image {
                let img = image::ImageReader::open(path)?.decode()?;
                let text = model.run(&img);

                match text {
                    Ok(text) => println!("{}", text),
                    Err(err) => println!("Error: {:?}", err),
                }
            } else {
                println!("No image provided");
            }
        }
        Mode::Clipboard => {
            let mut clipboard = Clipboard::new()?;
            let mut old_hash = match clipboard.get_image() {
                Ok(img) => Some(hash(&img.bytes)),
                Err(_) => None,
            };
            loop {
                sleep(Duration::from_secs_f32(1.0));

                let img = clipboard.get_image();
                if img.is_err() {
                    // println!("Error getting image from clipboard");
                    continue;
                }
                let img = img.unwrap();

                let new_hash = hash(&img.bytes);
                if old_hash.is_some() && new_hash == old_hash.unwrap() {
                    continue;
                }
                old_hash = Some(new_hash);
                let dyn_img = to_dyn_image(img);
                if let Some(dyn_img) = dyn_img {
                    let text = model.run(&dyn_img);
                    match text {
                        Ok(text) => {
                            println!("{}", text);
                            let _ = clipboard.set_text(text);
                        }
                        Err(err) => println!("Error: {:?}", err),
                    }
                }
            }
        }
    }
    Ok(())
}
