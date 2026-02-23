use std::ffi::CString;
use std::ptr;

fn model_path() -> Option<CString> {
    std::env::var("CACTUS_MODEL_PATH")
        .ok()
        .map(|p| CString::new(p).unwrap())
}

#[test]
fn complete_generates_text() {
    let Some(path) = model_path() else { return };

    let m = unsafe { cactus_sys::cactus_init(path.as_ptr(), ptr::null(), false) };
    assert!(!m.is_null());

    unsafe { cactus_sys::cactus_destroy(m) };
}
