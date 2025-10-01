pub mod gsv;
pub mod symbols;
pub mod text;
pub mod error;
pub mod ssl;
pub mod init;

pub use tch;

static TMP_DIR: &str = ".tmp";
static LIB_DIR: &str = "lib";
static COMMON_DIR: &str = "common";
static MODEL_DIR: &str = "model";
// 4MB大小的缓冲区大小
const BUFFER_SIZE: usize = 4 * 1024 * 1024;