# resamplescope-rs

Reverse-engineer the resampling filter used by any image resizer.

Port of [ResampleScope](http://entropymine.com/resamplescope/) by Jason Summers
(~1400 lines of C) to Rust. The original tool generates test images with known
pixel patterns, has the resizer process them, then reconstructs the filter kernel
shape from the output.

This port replaces that file-based workflow with a callback API: you provide a
resize closure, the crate handles everything in-memory. It adds scoring against
reference filters, SSIM comparison, and edge handling detection.

## How it works

1. Generates test patterns (a dot grid for downscale, a single bright column for upscale)
2. Passes them through your resize function
3. Reconstructs the filter kernel from the output pixel values
4. Scores the result against known filters (Box, Triangle, Hermite, Mitchell, Lanczos, etc.)
5. Returns the best match with correlation, RMS error, and detected support radius

## Usage

```rust
use imgref::{ImgRef, ImgVec};
use resamplescope::{AnalysisConfig, KnownFilter};

// Wrap your resizer to accept grayscale u8 images.
fn my_resize(src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize) -> ImgVec<u8> {
    // ... your resize implementation ...
    # resamplescope::perfect_resize(src, dst_w, dst_h, KnownFilter::Lanczos3)
}

let config = AnalysisConfig {
    srgb: false,       // set true if your resizer linearizes before filtering
    detect_edges: true,
};

let result = resamplescope::analyze(&my_resize, &config).unwrap();

if let Some(best) = result.best_match() {
    println!("Detected filter: {} (r={:.4})", best.filter, best.correlation);
}

// Render a 600x300 scope graph as ImgVec<RGB8>.
let graph = result.render_graph();
// Or with a reference filter overlay:
let graph = result.render_graph_with_reference(KnownFilter::Lanczos3);
```

## What you get back

`AnalysisResult` contains:

- **`downscale_curve`** -- reconstructed filter from the dot pattern (557->555 downscale)
- **`upscale_curve`** -- reconstructed filter from the line pattern (15->555 upscale)
- **`scores`** -- sorted best-first by Pearson correlation against known filters
- **`edge_mode`** -- detected edge handling (Clamp, Reflect, Wrap, Zero, or Unknown)

Each `FilterScore` includes correlation, RMS error, max error, and detected vs. expected support radius.

## Known filters

Box, Triangle, Hermite, Catmull-Rom, Mitchell, B-Spline, Lanczos2, Lanczos3, Lanczos4,
and arbitrary Mitchell-Netravali(B, C) parameterization.

## Reference resize

The `reference` module provides `perfect_resize` and `compute_weights` for generating
mathematically exact resize output for any built-in filter. Useful for SSIM comparison
against your resizer's output.

## No I/O

This crate has no image codec dependencies. It works with `ImgVec<u8>` (grayscale)
and `ImgVec<RGB8>` (graph output). You handle your own PNG/JPEG encoding.

## Original work

- **Author**: Jason Summers
- **Website**: http://entropymine.com/resamplescope/
- **License of original**: GPL-3.0-or-later

The test pattern generation and filter reconstruction algorithms are ported from
the original C source.

The reference filter math uses standard mathematical definitions (sinc,
Mitchell-Netravali, etc.). The authoritative, optimized implementations of these
filters live in [imageflow](https://github.com/imazen/imageflow).

## License

AGPL-3.0-or-later.
