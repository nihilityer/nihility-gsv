use tch::{Device, IValue, Tensor};

pub struct SSL {
    pub ssl: tch::CModule,
}

impl SSL {
    pub fn new(file_path: &str, device: Device) -> crate::error::Result<Self> {
        let mut ssl = tch::CModule::load_on_device(file_path, device)?;
        ssl.set_eval();
        Ok(SSL { ssl })
    }

    /// return: ssl_content
    pub fn to_ssl_content(&self, audio_16k: &Tensor) -> crate::error::Result<Tensor> {
        let r = self.ssl.forward_ts(&[audio_16k])?;
        Ok(r)
    }

    pub fn resample(
        &self,
        audio: &Tensor,
        sr: usize,
        target_sr: usize,
    ) -> crate::error::Result<Tensor> {
        tch::no_grad(|| {
            let resample = self.ssl.method_is(
                "resample",
                &[
                    &IValue::Tensor(audio.shallow_clone()),
                    &IValue::Int(sr as i64),
                    &IValue::Int(target_sr as i64),
                ],
            )?;
            match resample {
                IValue::Tensor(resample) => Ok(resample),
                _ => unreachable!(),
            }
        })
    }
}
