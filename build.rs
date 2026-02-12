fn main() {
    #[cfg(feature = "c-reference")]
    {
        cc::Build::new()
            .file("c_reference/rscope_shim.c")
            .warnings(true)
            .compile("rscope_shim");
    }
}
