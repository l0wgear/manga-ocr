# Manga OCR (Rust Version)

This project is a Rust port of the original [Manga OCR](https://github.com/kha-white/manga-ocr) by [kha-white](https://github.com/kha-white).

It provides optical character recognition for Japanese text, with a primary focus on Japanese manga. It uses a custom end-to-end model built with Transformers' [Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) framework, converted to the ONNX format.

## Motivation

The main motivation behind this project is to provide a slim, native manga OCR application that does not require setting up a Python environment and downloading large dependencies like PyTorch (which can be upwards of a gigabyte).

Additionally, this version aims to fix clipboard handling issues present in the original project, which required spawning invisible windows to work around Wayland limitations.

## Default Model

The default model used is [`l0wgear/manga-ocr-2025-onnx`](https://huggingface.co/l0wgear/manga-ocr-2025-onnx), which is an ONNX version of the model fine-tuned by [jzhang533](https://huggingface.co/jzhang533). You can find more details in the model's [readme](https://huggingface.co/l0wgear/manga-ocr-2025-onnx/blob/main/README.md). It's designed to be a high-quality text recognition tool, robust against various scenarios specific to manga:
- Both vertical and horizontal text
- Text with furigana
- Text overlaid on images
- A wide variety of fonts and font styles
- Low-quality images

## Installation

1.  Ensure you have the Rust toolchain installed. You can get it from [rustup.rs](https://rustup.rs/).
2.  Clone this repository.
3.  Build the project in release mode:
    ```bash
    cargo build --release
    ```
4.  The executable will be available at `target/release/manga-ocr-rs`.

## Usage

The application is controlled via command-line arguments.

```
Usage: manga-ocr-rs [OPTIONS]

Options:
  -m, --model <MODEL>
          The Hugging Face repository ID or local path for the ONNX model.
          [default: l0wgear/manga-ocr-2025-onnx]

  -i, --image <IMAGE>
          Path to the image file to process. Required when --mode is 'file'.

  --mode <MODE>
      The operating mode.
      - file: Process a single image file.
      - clipboard: Watch the clipboard for new images.
      [default: clipboard]
      [possible values: file, clipboard]

  --refresh-timeout <REFRESH_TIMEOUT>
          The timeout in seconds for refreshing the clipboard. Only applicable when --mode is 'clipboard'.
          [default: 1]

  -h, --help
          Print help

  -V, --version
          Print version
```

### Examples

**Clipboard Mode (Default):**

Run the application without any arguments to start watching the clipboard. When you copy a new image, it will be processed, and the recognized text will be copied back to the clipboard.

```bash
./target/release/manga-ocr-rs
```

**File Mode:**

Process a single image and print the recognized text to the console.

```bash
./target/release/manga-ocr-rs --mode file --image /path/to/your/image.png
```

## Original Project Information

This project is based on the work and documentation from the original `manga-ocr`. The following sections are from the original `README.md` available at [Manga OCR](https://github.com/kha-white/manga-ocr).

---

Manga OCR can be used as a general purpose printed Japanese OCR, but its main goal was to provide a high quality
text recognition, robust against various scenarios specific to manga:
- both vertical and horizontal text
- text with furigana
- text overlaid on images
- wide variety of fonts and font styles
- low quality images

Unlike many OCR models, Manga OCR supports recognizing multi-line text in a single forward pass,
so that text bubbles found in manga can be processed at once, without splitting them into lines.

See also:
- [Poricom](https://github.com/bluaxees/Poricom), a GUI reader, which uses manga-ocr
- [mokuro](https://github.com/kha-white/mokuro), a tool, which uses manga-ocr to generate an HTML overlay for manga
- [Xelieu's guide](https://rentry.co/lazyXel), a comprehensive guide on setting up a reading and mining workflow with manga-ocr/mokuro (and many other useful tips)

### Usage tips

- OCR supports multi-line text, but the longer the text, the more likely some errors are to occur.
  If the recognition failed for some part of a longer text, you might try to run it on a smaller portion of the image.
- The model was trained specifically to handle manga well, but should do a decent job on other types of printed text,
  such as novels or video games. It probably won't be able to handle handwritten text though.
- The model always attempts to recognize some text on the image, even if there is none.
  Because it uses a transformer decoder (and therefore has some understanding of the Japanese language),
  it might even "dream up" some realistically looking sentences! This shouldn't be a problem for most use cases,
  but it might get improved in the next version.

## Acknowledgements

- **Original Author:** [kha-white](https://github.com/kha-white) for creating the original Manga OCR.
- **Fine-tuning:** [jzhang533](https://huggingface.co/jzhang533) for training the `manga-ocr-base-2025` model.
- The training of the models was done with the usage of:
  - [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
  - [CC-100](https://data.statmt.org/cc-100/) dataset
