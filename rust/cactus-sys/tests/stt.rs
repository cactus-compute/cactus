use std::ffi::CString;
use std::ptr;

fn model_path() -> Option<CString> {
    std::env::var("CACTUS_STT_MODEL_PATH")
        .ok()
        .map(|p| CString::new(p).unwrap())
}

#[test]
fn init_transcribe_destroy() {
    let Some(path) = model_path() else { return };

    let m = unsafe { cactus_sys::cactus_init(path.as_ptr(), ptr::null(), false) };
    assert!(!m.is_null());

    let prompt = CString::new("").unwrap();
    let mut buf = vec![0i8; 2048];
    let pcm: Vec<u8> = vec![0u8; 3200]; // 0.1s of silence at 16kHz
    let rc = unsafe {
        cactus_sys::cactus_transcribe(
            m,
            ptr::null(),
            prompt.as_ptr(),
            buf.as_mut_ptr(),
            buf.len() as _,
            ptr::null(),
            None,
            ptr::null_mut(),
            pcm.as_ptr(),
            pcm.len() as _,
        )
    };
    assert_eq!(rc, 0);

    unsafe { cactus_sys::cactus_destroy(m) };
}
