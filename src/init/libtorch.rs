use super::{check_file_sha256, download, init_preprocess};
use crate::error::*;
use crate::{BUFFER_SIZE, LIB_DIR, TMP_DIR};
use std::fs::File;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;
use std::{fs, io};
use tracing::{debug, info};

static LIBTORCH_TMP_FILE: &str = "libtorch-cpu.zip";
static SOURCE_URL_PREFIX: &str = "https://download.pytorch.org/libtorch/cpu/";
static LIBTORCH_ZIP_LIB_FILE_PREFIX: &str = "libtorch/lib/";

#[cfg(target_os = "windows")]
static LIBTORCH_SOURCE_URL: &str = "libtorch-win-shared-with-deps-2.8.0%2Bcpu.zip";
#[cfg(target_os = "windows")]
static LIBTORCH_ZIP_FILE_SHA256: &str =
    "bf0079104e068dcbd60b2f8de589dc725282cbe865a85016cc5b8aa46cab519d";
#[cfg(target_os = "linux")]
static LIBTORCH_SOURCE_URL: &str = "libtorch-shared-with-deps-2.8.0%2Bcpu.zip";
#[cfg(target_os = "macos")]
static LIBTORCH_SOURCE_URL: &str = "libtorch-macos-arm64-2.8.0.zip";

pub fn init_libtorch() -> Result<()> {
    init_preprocess()?;
    let lib_dir = Path::new(LIB_DIR);
    let libtorch_zip_file = format!("{TMP_DIR}/{LIBTORCH_TMP_FILE}");
    let mut libtorch_zip_buf_reader = if fs::exists(&libtorch_zip_file)? {
        info!("libtorch-zip already exists, check file sha256");
        let libtorch_zip = File::open(&libtorch_zip_file)?;
        let mut libtorch_zip_buf_reader = BufReader::with_capacity(BUFFER_SIZE, libtorch_zip);
        if check_file_sha256(&mut libtorch_zip_buf_reader, LIBTORCH_ZIP_FILE_SHA256)? {
            info!("libtorch zip file already exists");
        } else {
            return Err(NihilityGsvError::Init(format!(
                "libtorch zip file exists same file: {}",
                &libtorch_zip_file
            )));
        }
        libtorch_zip_buf_reader
    } else {
        download(
            &format!("{SOURCE_URL_PREFIX}{LIBTORCH_SOURCE_URL}"),
            &libtorch_zip_file,
        )?;
        let libtorch_zip = File::open(&libtorch_zip_file)?;
        let mut libtorch_zip_buf_reader = BufReader::with_capacity(BUFFER_SIZE, libtorch_zip);
        if check_file_sha256(&mut libtorch_zip_buf_reader, LIBTORCH_ZIP_FILE_SHA256)? {
            info!("libtorch zip file download failed");
        } else {
            return Err(NihilityGsvError::Init(format!(
                "Check libtorch zip sha256 fail: {}",
                &libtorch_zip_file
            )));
        }
        libtorch_zip_buf_reader
    };
    libtorch_zip_buf_reader.seek(SeekFrom::Start(0))?;
    let mut archive = zip::ZipArchive::new(libtorch_zip_buf_reader)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name().starts_with(LIBTORCH_ZIP_LIB_FILE_PREFIX) && file.is_file() {
            let lib_file_name = file.name().replace(LIBTORCH_ZIP_LIB_FILE_PREFIX, "");
            debug!("extract zip lib file: {}", lib_file_name);
            let out_path = lib_dir.join(lib_file_name);
            if fs::exists(&out_path)? {
                continue;
            }
            let mut outfile = File::create(&out_path)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    info!("extracted libtorch lib to {}", lib_dir.display());
    Ok(())
}
