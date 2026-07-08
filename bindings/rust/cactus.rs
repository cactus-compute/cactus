use std::os::raw::{c_char, c_float, c_int, c_void};

pub type CactusModelT = *mut c_void;
pub type CactusIndexT = *mut c_void;
pub type CactusStreamTranscribeT = *mut c_void;
pub type CactusTokenCallback = Option<unsafe extern "C" fn(token: *const c_char, token_id: u32, user_data: *mut c_void)>;
pub type CactusLogCallback = Option<unsafe extern "C" fn(level: c_int, component: *const c_char, message: *const c_char, user_data: *mut c_void)>;

#[cfg_attr(target_os = "macos", link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "Metal", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "MetalPerformanceShaders", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "Foundation", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "Security", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "SystemConfiguration", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "CFNetwork", kind = "framework"))]
#[cfg_attr(target_os = "macos", link(name = "curl"))]
#[link(name = "cactus_engine", kind = "static")]
unsafe extern "C" {
    pub fn cactus_init(model_path: *const c_char, corpus_dir: *const c_char, cache_index: bool) -> CactusModelT;
    pub fn cactus_destroy(model: CactusModelT);
    pub fn cactus_reset(model: CactusModelT);
    pub fn cactus_stop(model: CactusModelT);

    pub fn cactus_complete(model: CactusModelT, messages_json: *const c_char, response_buffer: *mut c_char, buffer_size: usize, options_json: *const c_char, tools_json: *const c_char, callback: CactusTokenCallback, user_data: *mut c_void, pcm_buffer: *const u8, pcm_buffer_size: usize) -> c_int;
    pub fn cactus_prefill(model: CactusModelT, messages_json: *const c_char, response_buffer: *mut c_char, buffer_size: usize, options_json: *const c_char, tools_json: *const c_char, pcm_buffer: *const u8, pcm_buffer_size: usize) -> c_int;
    pub fn cactus_tokenize(model: CactusModelT, text: *const c_char, token_buffer: *mut u32, token_buffer_len: usize, out_token_len: *mut usize) -> c_int;
    pub fn cactus_score_window(model: CactusModelT, tokens: *const u32, token_len: usize, start: usize, end: usize, context: usize, response_buffer: *mut c_char, buffer_size: usize) -> c_int;
    pub fn cactus_transcribe(model: CactusModelT, audio_file_path: *const c_char, prompt: *const c_char, response_buffer: *mut c_char, buffer_size: usize, options_json: *const c_char, callback: CactusTokenCallback, user_data: *mut c_void, pcm_buffer: *const u8, pcm_buffer_size: usize) -> c_int;
    pub fn cactus_stream_transcribe_start(model: CactusModelT, options_json: *const c_char) -> CactusStreamTranscribeT;
    pub fn cactus_stream_transcribe_process(stream: CactusStreamTranscribeT, pcm_buffer: *const u8, pcm_buffer_size: usize, response_buffer: *mut c_char, buffer_size: usize) -> c_int;
    pub fn cactus_stream_transcribe_stop(stream: CactusStreamTranscribeT, response_buffer: *mut c_char, buffer_size: usize) -> c_int;

    pub fn cactus_embed(model: CactusModelT, text: *const c_char, embeddings_buffer: *mut c_float, buffer_size: usize, embedding_dim: *mut usize, normalize: bool) -> c_int;
    pub fn cactus_image_embed(model: CactusModelT, image_path: *const c_char, embeddings_buffer: *mut c_float, buffer_size: usize, embedding_dim: *mut usize) -> c_int;
    pub fn cactus_audio_embed(model: CactusModelT, audio_path: *const c_char, embeddings_buffer: *mut c_float, buffer_size: usize, embedding_dim: *mut usize) -> c_int;

    pub fn cactus_rag_query(model: CactusModelT, query: *const c_char, response_buffer: *mut c_char, buffer_size: usize, top_k: usize) -> c_int;

    pub fn cactus_index_init(index_dir: *const c_char, embedding_dim: usize) -> CactusIndexT;
    pub fn cactus_index_add(index: CactusIndexT, ids: *const c_int, documents: *const *const c_char, metadatas: *const *const c_char, embeddings: *const *const c_float, count: usize, embedding_dim: usize) -> c_int;
    pub fn cactus_index_delete(index: CactusIndexT, ids: *const c_int, ids_count: usize) -> c_int;
    pub fn cactus_index_get(index: CactusIndexT, ids: *const c_int, ids_count: usize, document_buffers: *mut *mut c_char, document_buffer_sizes: *mut usize, metadata_buffers: *mut *mut c_char, metadata_buffer_sizes: *mut usize, embedding_buffers: *mut *mut c_float, embedding_buffer_sizes: *mut usize) -> c_int;
    pub fn cactus_index_query(index: CactusIndexT, embeddings: *const *const c_float, embeddings_count: usize, embedding_dim: usize, options_json: *const c_char, id_buffers: *mut *mut c_int, id_buffer_sizes: *mut usize, score_buffers: *mut *mut c_float, score_buffer_sizes: *mut usize) -> c_int;
    pub fn cactus_index_compact(index: CactusIndexT) -> c_int;
    pub fn cactus_index_destroy(index: CactusIndexT);

    pub fn cactus_get_last_error() -> *const c_char;
    pub fn cactus_log_set_level(level: c_int);
    pub fn cactus_log_set_callback(callback: CactusLogCallback, user_data: *mut c_void);
    pub fn cactus_set_telemetry_environment(framework: *const c_char, cache_location: *const c_char, version: *const c_char);
    pub fn cactus_set_app_id(app_id: *const c_char);
    pub fn cactus_telemetry_flush();
    pub fn cactus_telemetry_shutdown();
}
