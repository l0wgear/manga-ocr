mod clipboard;
mod config;
mod hf;
mod model;

use crate::clipboard::ClipboardHandler;
use crate::model::OCRModel;
use clap::{Parser, ValueEnum};
use std::fmt::Display;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "l0wgear/manga-ocr-2025-onnx")]
    model: String,

    #[arg(short, long)]
    image: Option<PathBuf>,

    #[arg(long, default_value_t = Mode::Clipboard)]
    mode: Mode,

    #[arg(long, default_value = "1.0")]
    refresh_timeout: f64,
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Initializing model...");
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
            let mut clipboard = ClipboardHandler::new(args.refresh_timeout)?;
            println!("Ready to do OCR");

            loop {
                let image = clipboard.get_image();
                if let Some(img) = image {
                    let text = model.run(&img);
                    match text {
                        Ok(text) => {
                            println!("{}", text);
                            let _ = clipboard.set_text(&text);
                        }
                        Err(err) => println!("Error: {:?}", err),
                    }
                }
            }
        }
    }
    Ok(())
}
