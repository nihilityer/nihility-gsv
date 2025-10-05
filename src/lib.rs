pub mod error;
pub mod gsv;
pub mod ssl;
pub mod symbols;
pub mod text;

use crate::error::*;
use crate::gsv::Gsv;
use crate::ssl::SSL;
use crate::text::{G2PConfig, G2p};
use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
pub use tch;
use tch::Device;
use tracing::{error, info};
use wav_io::header::WavHeader;

const REF_PATH: &str = "ref.wav";
const REF_TEXT: &str = "ref.txt";
const GSV_MODEL: &str = "model.pt";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NihilityGsvConfig {
    pub g2p_en_model: String,
    pub g2p_zh_model: String,
    pub bert_model: String,
    pub ssl_model: String,
    pub gsv_dir: String,
    pub selected_model: String,
    pub output_dir: String,
}

pub struct NihilityGsv {
    g2p: G2p,
    gsv: Gsv,
    wav_header: WavHeader,
    output_path: PathBuf,
}

impl NihilityGsvConfig {
    pub fn init(self, device: Device) -> Result<NihilityGsv> {
        info!("use torch device: {:?}", device);
        if !fs::exists(&self.g2p_en_model)?
            || !fs::exists(&self.g2p_zh_model)?
            || !fs::exists(&self.bert_model)?
            || !fs::exists(&self.ssl_model)?
        {
            error!("GSV base model does not exist");
            return Err(NihilityGsvError::Infer(
                "GSV base model does not exist".into(),
            ));
        }
        let gsv_base_dir = Path::new(&self.gsv_dir).join(&self.selected_model);
        if !gsv_base_dir.exists() {
            error!("GSV model dir does not exist");
            return Err(NihilityGsvError::Infer(
                "GSV model dir does not exist".into(),
            ));
        }
        let output_path = Path::new(&self.output_dir).join(&self.selected_model);
        if !output_path.exists() {
            info!("Creating GSV output dir {:?}", output_path);
            fs::create_dir_all(&output_path)?;
        } else if !output_path.is_dir() {
            error!("Output path is not a directory");
            return Err(NihilityGsvError::Infer(
                "Output path is not a directory".into(),
            ));
        }

        let ref_path = gsv_base_dir.join(REF_PATH);
        if !ref_path.exists() {
            error!("Ref audio does not exist");
            return Err(NihilityGsvError::Infer("Ref audio does not exist".into()));
        }
        let ref_path = ref_path
            .to_str()
            .ok_or(NihilityGsvError::Infer("Ref audio exception".into()))?
            .to_string();

        let ref_text = gsv_base_dir.join(REF_TEXT);
        if !ref_text.exists() {
            error!("Ref text does not exist");
            return Err(NihilityGsvError::Infer("Ref text does not exist".into()));
        }
        let ref_text = fs::read_to_string(&ref_text)?;

        let gsv_model_path = gsv_base_dir.join(GSV_MODEL);
        if !gsv_model_path.exists() {
            error!("Gsv model does not exist");
            return Err(NihilityGsvError::Infer("Gsv model does not exist".into()));
        }
        let gsv_model_path = gsv_model_path
            .to_str()
            .expect("Gsv model path exception")
            .to_string();

        let g2p_conf = G2PConfig::new(self.g2p_en_model.clone())
            .with_chinese(self.g2p_zh_model.clone(), self.bert_model.clone());
        let g2p = g2p_conf.build(device)?;
        let file = fs::File::open(ref_path)?;
        let (head, mut ref_audio_samples) = wav_io::read_from_file(file)?;
        info!("ref wav file head: {:?}", head);
        if head.sample_rate != 32000 {
            info!("ref audio sample rate: {}, need 32000", head.sample_rate);
            ref_audio_samples =
                wav_io::resample::linear(ref_audio_samples, 1, head.sample_rate, 32000);
        }
        let (ref_seq, ref_bert) = text::get_phone_and_bert(&g2p, &ref_text)?;
        let ref_audio_32k = tch::Tensor::from_slice(&ref_audio_samples)
            .to_device(device)
            .unsqueeze(0);
        info!("load ref done");

        let _g = tch::no_grad_guard();

        let wav_header = wav_io::new_header(32000, 16, false, true);
        let ssl = SSL::new(&self.ssl_model, device)?;

        let ref_audio_16k = ssl.resample(&ref_audio_32k, 32000, 16000)?;
        let mut ssl_content = ssl.to_ssl_content(&ref_audio_16k)?;
        if ref_audio_32k.kind() == tch::Kind::Half {
            ssl_content = ssl_content.internal_cast_half(false);
        }
        let gsv = Gsv::new(
            &gsv_model_path,
            device,
            ssl_content,
            ref_audio_32k,
            ref_seq,
            ref_bert,
        )?;
        Ok(NihilityGsv {
            g2p,
            gsv,
            wav_header,
            output_path,
        })
    }
}

impl NihilityGsv {
    pub fn infer(&self, text: &str) -> Result<Vec<f32>> {
        info!("infer text: {}", text);
        let st = std::time::Instant::now();
        let (text_seq, text_bert) = text::get_phone_and_bert(&self.g2p, text)?;
        let audio = self.gsv.infer(&text_seq, &text_bert, 15)?;
        info!("infer done, cost: {:?}", st.elapsed());
        let audio_size = audio.size1().expect("Failed to get audio size") as usize;
        let mut samples = vec![0f32; audio_size];
        audio.f_copy_data(&mut samples, audio_size)?;
        Ok(samples)
    }

    pub fn infer_out_to_local(&self, text: &str) -> Result<()> {
        let samples = self.infer(text)?;
        let out_wav_name = Local::now().format("%Y-%m-%d-%H-%M-%S").to_string();
        let output = self.output_path.join(format!("{}.wav", out_wav_name));
        let mut file_out = fs::File::create(&output)?;
        wav_io::write_to_file(&mut file_out, &self.wav_header, &samples)?;
        info!("write output audio wav to {}", output.display());
        Ok(())
    }
}

impl Default for NihilityGsvConfig {
    fn default() -> Self {
        NihilityGsvConfig {
            g2p_en_model: "base/g2p-en.pt".to_string(),
            g2p_zh_model: "base/g2p-zh.pt".to_string(),
            bert_model: "base/bert.pt".to_string(),
            ssl_model: "base/ssl.pt".to_string(),
            gsv_dir: "model".to_string(),
            selected_model: "default".to_string(),
            output_dir: "output".to_string(),
        }
    }
}
