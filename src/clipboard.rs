use std::{
    hash::{DefaultHasher, Hash, Hasher},
    thread::sleep,
    time::Duration,
};

use arboard::{Clipboard, ImageData};
use image::{DynamicImage, ImageBuffer, Rgba};

fn hash(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
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

pub struct ClipboardHandler {
    clipboard: Clipboard,
    old_hash: Option<u64>,
    refresh_timeout: f64,
    do_wait: bool,
}

impl ClipboardHandler {
    pub fn new(refresh_timeout: f64) -> anyhow::Result<Self> {
        let mut clipboard = Clipboard::new()?;
        let old_hash = match clipboard.get_image() {
            Ok(img) => Some(hash(&img.bytes)),
            Err(_) => None,
        };
        Ok(Self {
            clipboard,
            refresh_timeout,
            old_hash,
            do_wait: false,
        })
    }
    pub fn get_image(&mut self) -> Option<DynamicImage> {
        if self.do_wait {
            sleep(Duration::from_secs_f64(self.refresh_timeout));
        }
        self.do_wait = true;
        let image = match self.clipboard.get_image() {
            Ok(img) => img,
            Err(arboard::Error::ContentNotAvailable) => {
                return None;
            }
            Err(err) => {
                println!("Error getting image from clipboard: {}", err);
                return None;
            }
        };
        let new_hash = hash(&image.bytes);
        if self.old_hash == Some(new_hash) {
            return None;
        }

        self.old_hash = Some(new_hash);
        to_dyn_image(image)
    }

    pub fn set_text(&mut self, text: &str) -> anyhow::Result<()> {
        self.clipboard.set_text(text)?;
        Ok(())
    }
}
