use crate::error::Result;
use tch::{Device, Tensor};

pub struct Gsv {
    pub model: tch::CModule,
    pub ssl_content: Tensor,
    pub ref_audio_32k: Tensor,
    pub ref_seq: Tensor,
    pub ref_bert: Tensor,
}

impl Gsv {
    pub fn new(
        model_path: &str,
        device: Device,
        ssl_content: Tensor,
        ref_audio_32k: Tensor,
        ref_seq: Tensor,
        ref_bert: Tensor,
    ) -> Result<Self> {
        let mut model = tch::CModule::load_on_device(model_path, device)?;
        model.set_eval();
        Ok(Gsv {
            model,
            ssl_content,
            ref_audio_32k,
            ref_seq,
            ref_bert,
        })
    }

    pub fn infer(&self, text_seq: &Tensor, text_bert: &Tensor, top_k: i64) -> Result<Tensor> {
        let audio = self.model.forward_ts(&[
            &self.ssl_content,
            &self.ref_audio_32k,
            &self.ref_seq,
            text_seq,
            &self.ref_bert,
            text_bert,
            &Tensor::from_slice(&[top_k]),
        ])?;
        let size = 32000.0 * 0.3;
        let zero = Tensor::zeros([size as i64], (tch::Kind::Float, audio.device()));
        Ok(Tensor::cat(&[audio, zero], 0))
    }
}
