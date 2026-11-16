//! Model selection: the runtime is model-agnostic, so every model brings its own
//! tokenizer and artifacts through its own manifest. Nothing here names a model.
//!
//! Selection order for the active model manifest:
//! 1. `GLOBALMIND_MODEL_MANIFEST` (a path to a `.manifest` file).
//! 2. Otherwise the first candidate manifest in `manifests/` (sorted by name, admitted
//!    run manifests excluded) whose artifacts are all present on this machine.
//!
//! The tokenizer of a model is the artifact whose role is `tokenizer` in that model's
//! manifest entry, or `tokenizer.json` beside a model directory given by
//! `GLOBALMIND_MODEL_DIR`.

use std::path::{Path, PathBuf};

use crate::{ArtifactEntry, Manifest};

pub const MANIFEST_ENV: &str = "GLOBALMIND_MODEL_MANIFEST";
pub const MODEL_DIR_ENV: &str = "GLOBALMIND_MODEL_DIR";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("manifests")
}

fn all_present(manifest: &Manifest) -> bool {
    !manifest.models.is_empty()
        && manifest
            .models
            .iter()
            .all(|m| m.artifacts.iter().all(|a| Path::new(&a.path).is_file()))
}

/// The active model manifest path (see module docs).
pub fn model_manifest() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var(MANIFEST_ENV) {
        let path = PathBuf::from(path);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!("{MANIFEST_ENV} points at a missing file: {}", path.display()))
        };
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(manifest_dir())
        .map_err(|e| format!("read manifests/: {e}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            name.ends_with(".manifest") && !name.contains(".admitted.") && !name.starts_with("run")
        })
        .collect();
    candidates.sort();
    for path in candidates {
        let Some(text) = path.to_str() else { continue };
        if let Ok(manifest) = Manifest::load(text) {
            if all_present(&manifest) {
                return Ok(path);
            }
        }
    }
    Err(format!(
        "no model manifest in {} has all its artifacts on this machine; set {MANIFEST_ENV}",
        manifest_dir().display()
    ))
}

fn tokenizer_artifact(manifest: &Manifest) -> Option<&ArtifactEntry> {
    manifest
        .models
        .first()?
        .artifacts
        .iter()
        .find(|a| a.role == "tokenizer")
}

/// Path to the active model's own tokenizer.
pub fn model_tokenizer() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var(MODEL_DIR_ENV) {
        let path = Path::new(&dir).join("tokenizer.json");
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!("{MODEL_DIR_ENV} has no tokenizer.json: {dir}"))
        };
    }
    let manifest_path = model_manifest()?;
    let manifest = Manifest::load(manifest_path.to_str().ok_or("manifest path is not UTF-8")?)?;
    tokenizer_artifact(&manifest)
        .map(|a| PathBuf::from(&a.path))
        .ok_or_else(|| format!("{} lists no tokenizer artifact", manifest_path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_a_present_model_and_its_own_tokenizer() {
        let manifest = model_manifest().expect("a local model manifest is present");
        let tokenizer = model_tokenizer().expect("that model lists its tokenizer");
        assert!(tokenizer.is_file());
        let loaded = Manifest::load(manifest.to_str().unwrap()).unwrap();
        assert!(all_present(&loaded));
        assert_eq!(tokenizer_artifact(&loaded).map(|a| PathBuf::from(&a.path)), Some(tokenizer));
    }
}
