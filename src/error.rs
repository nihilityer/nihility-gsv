use crate::text::num::{Rule};

pub type Result<T> = core::result::Result<T, NihilityGsvError>;

#[derive(thiserror::Error, Debug)]
pub enum NihilityGsvError {
    #[error(transparent)]
    Tch(#[from] tch::TchError),
    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),
    #[error(transparent)]
    Pest(#[from] pest::error::Error<Rule>),
    #[error("Feature Extraction Error: {0}")]
    FeatureExtraction(String),
    #[error("Infer Error: {0}")]
    Infer(String),
}
