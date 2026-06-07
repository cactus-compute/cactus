# Rust Bindings

Raw `extern "C"` declarations for `cactus_engine.h`.

## Integration

<!-- --8<-- [start:install] -->
```bash
cactus build
```
<!-- --8<-- [end:install] -->

Copy [`cactus.rs`](cactus.rs) into your project (it carries its own `#[link]`
attributes) and point Cargo at the build directory:

```rust
// build.rs
println!("cargo:rustc-link-search=native=/path/to/cactus/cactus-engine/build");
```

## Usage

```rust
use std::ffi::CString;
use std::os::raw::c_char;

mod cactus;

fn main() {
    unsafe {
        let path = CString::new("/path/to/model").unwrap();
        let model = cactus::cactus_init(path.as_ptr(), std::ptr::null(), false);

        let messages = CString::new(r#"[{"role":"user","content":"Hello"}]"#).unwrap();
        let mut response = vec![0u8; 65536];
        cactus::cactus_complete(
            model,
            messages.as_ptr(),
            response.as_mut_ptr() as *mut c_char,
            response.len(),
            std::ptr::null(), std::ptr::null(),
            None, std::ptr::null_mut(),
            std::ptr::null(), 0,
        );

        cactus::cactus_destroy(model);
    }
}
```
