//! The packed `.moe` and safetensors headers, through a real file.
//!
//! Slower than the others because the store maps a path rather than a slice, but
//! it is the parser whose offsets are dereferenced straight into a memory map, so
//! it is the one where a bounds mistake is worst.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let path = std::env::temp_dir().join(format!("moe-fuzz-{}.moe", std::process::id()));
    if std::fs::write(&path, data).is_err() {
        return;
    }
    if let Ok(store) = moe::Store::open(&path) {
        let _ = store.bytes();
        let names: Vec<String> = store.names().map(String::from).collect();
        for n in &names {
            let _ = store.get(n);
            let _ = store.view(n, usize::MAX, 0..1);
        }
        let _ = moe::Model::load(store);
    }
});
