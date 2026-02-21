use std::ffi::{CStr, CString};
use std::ptr;

fn model_path() -> Option<CString> {
    std::env::var("CACTUS_MODEL_PATH")
        .ok()
        .map(|p| CString::new(p).unwrap())
}

#[test]
fn init_bad_path_returns_null() {
    let path = CString::new("/no/such/model").unwrap();
    let m = unsafe { cactus_sys::cactus_init(path.as_ptr(), ptr::null(), false) };
    assert!(m.is_null());
}

#[test]
fn last_error_set_after_failed_init() {
    let path = CString::new("/no/such/model").unwrap();
    unsafe { cactus_sys::cactus_init(path.as_ptr(), ptr::null(), false) };
    let err = unsafe { CStr::from_ptr(cactus_sys::cactus_get_last_error()) };
    assert!(!err.to_bytes().is_empty());
}

#[test]
fn init_tokenize_complete_destroy() {
    let Some(path) = model_path() else { return };

    let m = unsafe { cactus_sys::cactus_init(path.as_ptr(), ptr::null(), false) };
    assert!(!m.is_null());

    let text = CString::new("Hello").unwrap();
    let mut tokens = vec![0u32; 64];
    let mut n = 0usize;
    let rc = unsafe {
        cactus_sys::cactus_tokenize(m, text.as_ptr(), tokens.as_mut_ptr(), tokens.len(), &mut n)
    };
    assert_eq!(rc, 0);
    assert!(n > 0);

    let msgs = CString::new(r#"[{"role":"user","content":"Hi"}]"#).unwrap();
    let opts = CString::new(r#"{"max_tokens":8}"#).unwrap();
    let mut buf = vec![0i8; 2048];
    let rc = unsafe {
        cactus_sys::cactus_complete(
            m,
            msgs.as_ptr(),
            buf.as_mut_ptr(),
            buf.len() as _,
            opts.as_ptr(),
            ptr::null(),
            None,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0);
    assert!(!unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_bytes()
        .is_empty());

    unsafe { cactus_sys::cactus_destroy(m) };
}
