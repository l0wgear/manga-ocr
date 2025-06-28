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
