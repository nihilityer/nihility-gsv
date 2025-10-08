use crate::error::*;
use std::{collections::HashMap, fmt::Debug, sync::Arc};

static MONO_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_mono_chars.json");
static POLY_CHARS_DIST_STR: &str = include_str!("../../resource/g2pw/dict_poly_chars.json");
static LABELS: &str = include_str!("../../resource/g2pw/dict_poly_index_list.json");
pub(crate) static G2PW_TOKENIZER: &str = include_str!("../../resource/g2pw_tokenizer.json");

fn load_mono_chars() -> HashMap<char, MonoChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_mono_chars.json", dir))
            .expect("dict_mono_chars.json not found");
        serde_json::from_str(&s).expect("dict_mono_chars.json parse error")
    } else {
        serde_json::from_str(MONO_CHARS_DIST_STR).unwrap()
    }
}

fn load_poly_chars() -> HashMap<char, PolyChar> {
    if let Ok(dir) = std::env::var("G2PW_DIST_DIR") {
        let s = std::fs::read_to_string(format!("{}/dict_poly_chars.json", dir))
            .expect("dict_poly_chars.json not found");
        serde_json::from_str(&s).expect("dict_poly_chars.json parse error")
    } else {
        serde_json::from_str(POLY_CHARS_DIST_STR).unwrap()
    }
}

lazy_static::lazy_static! {
    static ref DICT_MONO_CHARS: HashMap<char, MonoChar> =load_mono_chars();
    static ref DICT_POLY_CHARS: HashMap<char, PolyChar> = load_poly_chars();
    static ref POLY_LABLES: Vec<String> = serde_json::from_str(LABELS).unwrap();
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PolyChar {
    index: usize,
    phones: Vec<(String, usize)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MonoChar {
    phone: String,
}

#[derive(Clone, Copy)]
pub enum G2PWOut {
    Pinyin(&'static str),
    RawChar(char),
}

impl Debug for G2PWOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pinyin(s) => write!(f, "\"{}\"", s),
            Self::RawChar(s) => write!(f, "\"{}\"", s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct G2PWConverter {
    model: Option<Arc<tch::CModule>>,
    tokenizers: Option<Arc<tokenizers::Tokenizer>>,
    device: tch::Device,
}

pub fn str_is_chinese(s: &str) -> bool {
    let mut r = true;
    for c in s.chars() {
        if !DICT_MONO_CHARS.contains_key(&c) && !DICT_POLY_CHARS.contains_key(&c) {
            r &= false;
        }
    }
    r
}

impl G2PWConverter {
    pub fn empty() -> Self {
        Self {
            model: None,
            tokenizers: None,
            device: tch::Device::Cpu,
        }
    }

    pub fn new(model_path: &str, tokenizer: Arc<tokenizers::Tokenizer>) -> Result<Self> {
        let device = tch::Device::Cpu;
        Self::new_with_device(model_path, tokenizer, device)
    }

    pub fn new_with_device(
        model_path: &str,
        tokenizer: Arc<tokenizers::Tokenizer>,
        mut device: tch::Device,
    ) -> Result<Self> {
        if device == tch::Device::Mps {
            device = tch::Device::Cpu;
        }

        let mut model = tch::CModule::load_on_device(model_path, device)?;

        model.set_eval();
        Ok(Self {
            model: Some(Arc::new(model)),
            tokenizers: Some(tokenizer),
            device,
        })
    }

    pub fn get_pinyin(&self, text: &str) -> Result<Vec<G2PWOut>> {
        if self.model.is_some() && self.tokenizers.is_some() {
            self.ml_get_pinyin(text)
        } else {
            Ok(self.simple_get_pinyin(text))
        }
    }

    pub fn simple_get_pinyin(&self, text: &str) -> Vec<G2PWOut> {
        let mut pre_data = vec![];
        for c in text.chars() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&mono.phone));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&poly.phones[0].0));
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }
        pre_data
    }

    fn ml_get_pinyin(&self, text: &str) -> Result<Vec<G2PWOut>> {
        let c = self.tokenizers.as_ref().unwrap().encode(text, true)?;
        let input_ids = c.get_ids().iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let token_type_ids = vec![0i64; input_ids.len()];
        let attention_mask = vec![1i64; input_ids.len()];

        let mut phoneme_masks = vec![];
        let mut pre_data = vec![];
        let mut query_id = vec![];
        let mut chars_id = vec![];

        for (i, c) in text.chars().enumerate() {
            if let Some(mono) = DICT_MONO_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(&mono.phone));
            } else if let Some(poly) = DICT_POLY_CHARS.get(&c) {
                pre_data.push(G2PWOut::Pinyin(""));
                // 这个位置是 tokens 的位置，它的前后添加了 '[CLS]' 和 '[SEP]' 两个特殊字符
                query_id.push(i + 1);
                chars_id.push(poly.index);
                let mut phoneme_mask = vec![0f32; POLY_LABLES.len()];
                for (_, i) in &poly.phones {
                    phoneme_mask[*i] = 1.0;
                }
                phoneme_masks.push(phoneme_mask);
            } else {
                pre_data.push(G2PWOut::RawChar(c));
            }
        }

        let input_ids = tch::Tensor::from_slice(&input_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let token_type_ids = tch::Tensor::from_slice(&token_type_ids)
            .unsqueeze(0)
            .to_device(self.device);
        let attention_mask = tch::Tensor::from_slice(&attention_mask)
            .unsqueeze(0)
            .to_device(self.device);

        for ((position_id, phoneme_mask), char_id) in query_id
            .iter()
            .zip(phoneme_masks.iter())
            .zip(chars_id.iter())
        {
            let phoneme_mask = tch::Tensor::from_slice(phoneme_mask)
                .unsqueeze(0)
                .to_device(self.device);
            let position_id_t =
                tch::Tensor::from_slice(&[*position_id as i64]).to_device(self.device);
            let char_id = tch::Tensor::from_slice(&[*char_id as i64]).to_device(self.device);

            let probs = tch::no_grad(|| {
                self.model.as_ref().unwrap().forward_ts(&[
                    &input_ids,
                    &token_type_ids,
                    &attention_mask,
                    &phoneme_mask,
                    &char_id,
                    &position_id_t,
                ])
            })?;

            let i = probs.argmax(-1, false).int64_value(&[]);

            pre_data[*position_id - 1] = G2PWOut::Pinyin(&POLY_LABLES[i as usize]);
        }

        Ok(pre_data)
    }
}
