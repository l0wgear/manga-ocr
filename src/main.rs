mod hf;
mod model;

use crate::model::OCRModel;

fn main() -> anyhow::Result<()> {
    let model = OCRModel::from_name_or_path("l0wgear/manga-ocr-2025-onnx", 3, "[SEP]")?;

    let img = image::ImageReader::open("./test1.png")?.decode()?;
    let text = model.run(&img);

    match text {
        Ok(text) => println!("{}", text),
        Err(err) => println!("Error: {:?}", err),
    }

    Ok(())
}
