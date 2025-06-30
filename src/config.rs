use std::path::PathBuf;

use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    pub decoder_start_token_id: u32,
    pub early_stopping: bool,
    pub eos_token_id: u32,
    pub length_penalty: f32,
    pub max_length: u32,
    pub no_repeat_ngram_size: u32,
    pub num_beams: u32,
    pub pad_token_id: u32,
    pub transformers_version: String,
}

impl GenerationConfig {
    pub fn from_file(path: &PathBuf) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let config: Self = serde_json::from_reader(file)?;
        Ok(config)
    }
}
