//! Loads a `presets/*.toml` file (SPEC §4.3, §9) into two `PipelineConfig`s
//! (K and V) plus the machine-readable evidence block.

use serde::Deserialize;

use crate::pipeline::PipelineConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct Evidence {
    pub source: String,
    pub calibration: String,
    #[serde(default)]
    pub paper_claims: Vec<String>,
    #[serde(default)]
    pub not_claimed: Vec<String>,
    #[serde(default)]
    pub seeds_in_paper: Option<u32>,
    #[serde(default)]
    pub models_in_paper: Vec<String>,
    pub our_status: String,
    #[serde(default)]
    pub parity_checked: bool,
    #[serde(default)]
    pub license_of_reference_impl: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Preset {
    pub name: String,
    pub k: PipelineConfig,
    pub v: PipelineConfig,
    pub evidence: Option<Evidence>,
}

impl Preset {
    pub fn from_toml_str(s: &str) -> anyhow::Result<Preset> {
        Ok(toml::from_str(s)?)
    }

    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Preset> {
        let s = std::fs::read_to_string(path)?;
        Preset::from_toml_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_preset() {
        let toml_str = r#"
            name = "toy"

            [k]
            rotation = "hadamard"
            scale = "none"
            quantizer = { kind = "sign_residual", bits = 4 }
            window = { sink = 4, recent = 8, dtype_bytes = 2 }

            [v]
            rotation = "none"
            scale = "none"
            quantizer = { kind = "group_rtn", bits = 2, axis = "per_token" }
            window = { sink = 4, recent = 8, dtype_bytes = 2 }
        "#;
        let preset = Preset::from_toml_str(toml_str).expect("parses");
        assert_eq!(preset.name, "toy");
    }
}
