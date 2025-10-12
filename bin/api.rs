use actix_web::{App, HttpResponse, HttpServer, post, web};
use nihility_gsv::error::*;
use nihility_gsv::{NihilityGsv, NihilityGsvConfig, NihilityGsvInferParam};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NihilityGsvApiConfig {
    server_addr: String,
    server_port: u16,
}

#[post("/infer")]
async fn infer(
    gsv: web::Data<Mutex<NihilityGsv>>,
    json: web::Json<NihilityGsvInferParam>,
) -> Result<HttpResponse> {
    let audio = gsv.lock().await.infer_out_to_wav(json.into_inner())?;
    Ok(HttpResponse::Ok().content_type("audio/wav").body(audio))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    nihility_log::init().expect("could not init log");
    let gsv_api_config =
        nihility_config::get_config::<NihilityGsvApiConfig>("nihility-gsv-api".to_string())
            .expect("could not get nihility gsv api config");
    let device = tch::Device::cuda_if_available();
    let gsv = nihility_config::get_config::<NihilityGsvConfig>(env!("CARGO_PKG_NAME").to_string())
        .expect("could not get inner config")
        .init(device)
        .expect("Failed to init gsv");
    let gsv = web::Data::new(Mutex::new(gsv));

    HttpServer::new(move || App::new().app_data(gsv.clone()).service(infer))
        .bind((gsv_api_config.server_addr, gsv_api_config.server_port))?
        .run()
        .await
}

impl Default for NihilityGsvApiConfig {
    fn default() -> Self {
        NihilityGsvApiConfig {
            server_addr: "127.0.0.1".to_string(),
            server_port: 8080,
        }
    }
}
