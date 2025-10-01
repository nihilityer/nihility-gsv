use crate::error::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::{fs, io};
use tracing::info;

mod libtorch;

use crate::{BUFFER_SIZE, COMMON_DIR, LIB_DIR, MODEL_DIR, TMP_DIR};
pub use libtorch::init_libtorch;

fn init_preprocess() -> Result<()> {
    preprocess_dir(TMP_DIR)?;
    preprocess_dir(LIB_DIR)?;
    preprocess_dir(COMMON_DIR)?;
    preprocess_dir(MODEL_DIR)?;
    Ok(())
}

fn preprocess_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        fs::create_dir_all(path)?;
    } else if !path.is_dir() {
        return Err(NihilityGsvError::Init(format!(
            "{} is not a directory",
            path.display()
        )));
    }
    Ok(())
}

fn download<P: AsRef<Path>>(source_url: &str, target_file: P) -> Result<()> {
    info!(
        "Downloading source from {} to {}",
        source_url,
        target_file.as_ref().display()
    );
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let response = ureq::get(source_url).call()?;
    let response_code = response.status();
    if response_code != 200 {
        return Err(NihilityGsvError::Init(format!(
            "Unexpected response code {} for {}",
            response_code, source_url
        )));
    }
    let mut reader = response.into_body().into_reader();
    std::io::copy(&mut reader, &mut writer)?;
    info!("Downloaded source to {}", target_file.as_ref().display());
    Ok(())
}

pub fn check_file_sha256(buf_reader: &mut BufReader<File>, target_sha256: &str) -> Result<bool> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; BUFFER_SIZE];
    loop {
        let n = buf_reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(target_sha256 == hex::encode(hasher.finalize()))
}
