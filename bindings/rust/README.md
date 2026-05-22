# Rust Bindings

Raw `extern "C"` declarations for `cactus_engine.h`.

## Integration

```bash
cactus build
```

1. Copy `cactus.rs` into your project
2. Link against `libcactus.a` in your `build.rs`:

```rust
println!("cargo:rustc-link-lib=static=cactus");
println!("cargo:rustc-link-search=native=/path/to/cactus/cactus/build");
```

## Usage

```rust
use std::ffi::CString;
use std::os::raw::c_char;

mod cactus;

unsafe {
    let path = CString::new("/path/to/model").unwrap();
    let model = cactus::cactus_init(path.as_ptr(), std::ptr::null(), false);
    let mut buf = vec![0i8; 65536];
    cactus::cactus_complete(model, msgs.as_ptr(), buf.as_mut_ptr() as *mut c_char, buf.len(), std::ptr::null(), std::ptr::null(), None, std::ptr::null_mut(), std::ptr::null(), 0);
    cactus::cactus_destroy(model);
}
```
