use hf_hub::api::sync::{Api, ApiError};
use std::path::PathBuf;

pub fn pull_model(name: &str) -> Result<Vec<PathBuf>, ApiError> {
    let api = Api::new().unwrap();
    let repo = api.model(name.to_string());
    let info = repo.info()?;
    let mut local_paths = vec![];
    for item in info.siblings.iter() {
        local_paths.push(repo.get(&item.rfilename)?);
    }
    Ok(local_paths)
}

#[cfg(all(test, feature = "huggingface-tests"))]
mod tests {
    use super::*;

    #[test]
    fn test_pull_model_nonexistent() {
        let model_name = "~~nonexistent";
        let res = pull_model(model_name);
        assert!(res.is_err());
    }

    #[test]
    fn test_pull_model() {
        let model_name = "hf-tiny-model-private/tiny-random-YosoModel";
        let res = pull_model(model_name);
        assert!(res.is_ok());
        assert!(!res.unwrap().is_empty());
    }
}
