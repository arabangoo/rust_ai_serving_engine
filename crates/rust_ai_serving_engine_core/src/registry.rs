use crate::{EngineError, ModelFormat, ModelKind, ModelManifest, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

const COPY_BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedModel {
    pub manifest: ModelManifest,
    pub manifest_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct ModelRegistry {
    root: PathBuf,
}

impl ModelRegistry {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("manifests"))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn import_local(
        &self,
        id: impl Into<String>,
        weights: impl AsRef<Path>,
        kind: ModelKind,
        architecture: Option<String>,
        context_length: Option<u32>,
        chat_template: Option<String>,
    ) -> Result<ImportedModel> {
        let id = id.into();
        validate_model_id(&id)?;
        let weights = weights.as_ref();
        if !weights.is_file() {
            return Err(EngineError::ModelFileNotFound(
                weights.display().to_string(),
            ));
        }

        let manifest = ModelManifest {
            id: id.clone(),
            kind,
            format: format_from_path(weights)?,
            weights: weights.canonicalize()?.display().to_string(),
            sha256: sha256_file(weights)?,
            tokenizer: None,
            tokenizer_sha256: None,
            architecture,
            context_length,
            chat_template,
        };
        let manifest_path = self.manifest_path(&id);
        write_manifest_atomically(&manifest_path, &manifest)?;
        Ok(ImportedModel {
            manifest,
            manifest_path,
        })
    }

    /// Associates a Hugging Face `tokenizer.json` artifact with an existing model.
    pub fn attach_tokenizer(&self, id: &str, tokenizer: impl AsRef<Path>) -> Result<ModelManifest> {
        let tokenizer = tokenizer.as_ref();
        if !tokenizer.is_file() {
            return Err(EngineError::ModelFileNotFound(
                tokenizer.display().to_string(),
            ));
        }
        let mut manifest = self.get(id)?;
        manifest.tokenizer = Some(tokenizer.canonicalize()?.display().to_string());
        manifest.tokenizer_sha256 = Some(sha256_file(tokenizer)?);
        write_manifest_atomically(&self.manifest_path(id), &manifest)?;
        Ok(manifest)
    }

    pub fn get(&self, id: &str) -> Result<ModelManifest> {
        validate_model_id(id)?;
        let path = self.manifest_path(id);
        if !path.is_file() {
            return Err(EngineError::ModelNotFound(id.to_owned()));
        }
        Ok(toml::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn list(&self) -> Result<Vec<ModelManifest>> {
        let mut manifests: Vec<ModelManifest> = Vec::new();
        for entry in fs::read_dir(self.root.join("manifests"))? {
            let entry = entry?;
            if entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "toml")
            {
                manifests.push(toml::from_str(&fs::read_to_string(entry.path())?)?);
            }
        }
        manifests.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(manifests)
    }

    pub fn verify(&self, id: &str) -> Result<ModelManifest> {
        let manifest = self.get(id)?;
        let path = Path::new(&manifest.weights);
        if !path.is_file() {
            return Err(EngineError::ModelFileNotFound(manifest.weights));
        }
        let actual = sha256_file(path)?;
        if actual != manifest.sha256 {
            return Err(EngineError::IntegrityMismatch {
                id: manifest.id,
                expected: manifest.sha256,
                actual,
            });
        }
        if let (Some(tokenizer), Some(expected)) = (&manifest.tokenizer, &manifest.tokenizer_sha256)
        {
            let tokenizer_path = Path::new(tokenizer);
            if !tokenizer_path.is_file() {
                return Err(EngineError::ModelFileNotFound(tokenizer.clone()));
            }
            let actual = sha256_file(tokenizer_path)?;
            if &actual != expected {
                return Err(EngineError::IntegrityMismatch {
                    id: format!("{} tokenizer", manifest.id),
                    expected: expected.clone(),
                    actual,
                });
            }
        }
        Ok(manifest)
    }

    fn manifest_path(&self, id: &str) -> PathBuf {
        self.root.join("manifests").join(format!("{id}.toml"))
    }
}

fn validate_model_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(EngineError::InvalidModelId(id.to_owned()));
    }
    Ok(())
}

fn format_from_path(path: &Path) -> Result<ModelFormat> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("gguf") => Ok(ModelFormat::Gguf),
        Some("safetensors") => Ok(ModelFormat::Safetensors),
        _ => Err(EngineError::UnsupportedFormat(path.display().to_string())),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(COPY_BUFFER_SIZE, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; COPY_BUFFER_SIZE];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_manifest_atomically(path: &Path, manifest: &ModelManifest) -> Result<()> {
    let contents = toml::to_string_pretty(manifest)?;
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, contents)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn imports_lists_and_verifies_gguf_models() {
        let fixture = tempdir().unwrap();
        let model_path = fixture.path().join("fixture.gguf");
        fs::write(
            &model_path,
            b"not-a-real-model-but-a-valid-registry-fixture",
        )
        .unwrap();
        let registry = ModelRegistry::open(fixture.path().join("registry")).unwrap();

        let imported = registry
            .import_local(
                "fixture-model",
                &model_path,
                ModelKind::Generator,
                Some("qwen3".to_owned()),
                Some(8192),
                Some("chatml".to_owned()),
            )
            .unwrap();

        assert!(imported.manifest_path.is_file());
        assert_eq!(registry.list().unwrap(), vec![imported.manifest.clone()]);
        assert_eq!(registry.verify("fixture-model").unwrap(), imported.manifest);
    }

    #[test]
    fn detects_weight_changes_after_import() {
        let fixture = tempdir().unwrap();
        let model_path = fixture.path().join("fixture.safetensors");
        fs::write(&model_path, b"original").unwrap();
        let registry = ModelRegistry::open(fixture.path().join("registry")).unwrap();
        registry
            .import_local(
                "fixture",
                &model_path,
                ModelKind::Embedding,
                None,
                None,
                None,
            )
            .unwrap();
        fs::write(&model_path, b"modified").unwrap();

        assert!(matches!(
            registry.verify("fixture"),
            Err(EngineError::IntegrityMismatch { .. })
        ));
    }

    #[test]
    fn attaches_and_verifies_a_tokenizer() {
        let fixture = tempdir().unwrap();
        let weights = fixture.path().join("fixture.gguf");
        let tokenizer = fixture.path().join("tokenizer.json");
        fs::write(&weights, b"weights").unwrap();
        fs::write(&tokenizer, b"tokenizer").unwrap();
        let registry = ModelRegistry::open(fixture.path().join("registry")).unwrap();
        registry
            .import_local("fixture", &weights, ModelKind::Generator, None, None, None)
            .unwrap();

        let manifest = registry.attach_tokenizer("fixture", &tokenizer).unwrap();
        assert!(manifest.tokenizer.is_some());
        assert!(manifest.tokenizer_sha256.is_some());
        assert_eq!(registry.verify("fixture").unwrap(), manifest);
    }
}
