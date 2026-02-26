use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

fn model_path() -> Option<CString> {
    std::env::var("CACTUS_STT_MODEL_PATH")
        .ok()
        .map(|p| CString::new(p).unwrap())
}

fn audio_path() -> Option<CString> {
    std::env::var("CACTUS_STT_AUDIO_PATH")
        .ok()
        .map(|p| CString::new(p).unwrap())
}

#[test]
fn init_transcribe_destroy() {
    let Some(model) = model_path() else { return };
    let Some(audio) = audio_path() else { return };

    let m = unsafe { cactus_sys::cactus_init(model.as_ptr(), ptr::null(), false) };
    assert!(!m.is_null());

    let prompt = CString::new("<|startoftranscript|><|en|><|transcribe|><|notimestamps|>").unwrap();
    let mut buf = vec![0u8; 4096];
    let rc = unsafe {
        cactus_sys::cactus_transcribe(
            m,
            audio.as_ptr(),
            prompt.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as _,
            ptr::null(),
            None,
            ptr::null_mut(),
            ptr::null(),
            0,
        )
    };

    assert!(rc > 0, "cactus_transcribe failed with rc={}", rc);

    let null_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let response = std::str::from_utf8(&buf[..null_pos]).expect("invalid utf8");
    assert!(
        !response.is_empty(),
        "expected transcription text, got empty response"
    );
    println!("transcription: {}", response);

    unsafe { cactus_sys::cactus_destroy(m) };
}
