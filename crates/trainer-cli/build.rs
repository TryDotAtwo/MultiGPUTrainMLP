fn main() {
    println!("cargo:rerun-if-env-changed=MGT_SINGLE_GPU_NATIVE_LIB_DIR");
    if let Ok(directory) = std::env::var("MGT_SINGLE_GPU_NATIVE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={directory}");
        println!("cargo:rustc-link-lib=dylib=mgt_single_gpu_trainer");
    }
}
