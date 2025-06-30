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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_correct_config() {
        let path = PathBuf::from("tests/fixtures/generation_config.json");
        let res = GenerationConfig::from_file(&path);
        assert!(res.is_ok());
        let config = res.unwrap();
        assert_eq!(config.decoder_start_token_id, 2);
        assert_eq!(config.early_stopping, true);
        assert_eq!(config.eos_token_id, 3);
        assert_eq!(config.length_penalty, 2.0);
        assert_eq!(config.max_length, 300);
        assert_eq!(config.no_repeat_ngram_size, 3);
        assert_eq!(config.num_beams, 4);
        assert_eq!(config.pad_token_id, 0);
        assert_eq!(config.transformers_version, "4.52.4");
    }

    #[test]
    fn test_load_incorrect_config() {
        let path = PathBuf::from("tests/fixtures/generation_config_incorrect.json");
        let res = GenerationConfig::from_file(&path);
        assert!(res.is_err());
    }
}
