# resamplescope-rs [![CI](https://img.shields.io/github/actions/workflow/status/imazen/resamplescope-rs/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/resamplescope-rs/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/resamplescope-rs?style=flat-square)](https://crates.io/crates/resamplescope-rs) [![lib.rs](https://img.shields.io/crates/v/resamplescope-rs?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/resamplescope-rs) [![docs.rs](https://img.shields.io/docsrs/resamplescope-rs?style=flat-square)](https://docs.rs/resamplescope-rs) [![license](https://img.shields.io/crates/l/resamplescope-rs?style=flat-square)](https://github.com/imazen/resamplescope-rs#license)

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

## Add to Cargo.toml

The public API hands you `imgref` and `rgb` types directly (your resize closure
takes an `ImgRef<'_, u8>` and returns an `ImgVec<u8>`; graphs come back as
`ImgVec<RGB8>`), so add both as direct dependencies alongside this crate. Match
the versions this crate is built against:

```toml
[dependencies]
resamplescope-rs = "0.1.0"
imgref = "1"
rgb = { version = "0.8", default-features = false }
```

`rgb` is pulled in with `default-features = false` here because only the plain
`RGB8` struct is needed for graph output — no extra `rgb` features are required.

(The library crate is named `resamplescope` even though the package is
`resamplescope-rs`, so imports read `use resamplescope::...`.)

## Usage

```rust
use imgref::{ImgRef, ImgVec};
use rgb::RGB8;
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
let graph: ImgVec<RGB8> = result.render_graph();
// Or with a reference filter overlay:
let graph = result.render_graph_with_reference(KnownFilter::Lanczos3);
```

`AnalysisConfig` has exactly two fields (it is not `#[non_exhaustive]`, so you
can construct it with a struct literal as above, or start from
`AnalysisConfig::default()`):

```rust
pub struct AnalysisConfig {
    /// `true` if the resizer works in linear light — i.e. it converts sRGB to
    /// linear *before* filtering and back *after*. The analyzer then linearizes
    /// the probe output the same way before reconstructing the kernel. Set
    /// `false` for a resizer that filters directly on sRGB-encoded samples.
    pub srgb: bool,
    /// Whether to run edge-handling detection (fills in `edge_mode`).
    pub detect_edges: bool,
}
```

The default is `srgb: false`, `detect_edges: true`.

## What you get back

`analyze()` returns `Result<AnalysisResult, Error>`. All of the following are
plain public **fields** (not accessor methods), so you read them directly. None
of these structs is `#[non_exhaustive]`.

```rust
pub struct AnalysisResult {
    /// Filter reconstructed from the dot pattern (557->555 downscale). `None` if
    /// reconstruction found no usable points.
    pub downscale_curve: Option<FilterCurve>,
    /// Filter reconstructed from the line pattern (15->555 upscale). `None` if
    /// reconstruction found no usable points.
    pub upscale_curve: Option<FilterCurve>,
    /// One entry per known filter, sorted best-first by Pearson correlation.
    pub scores: Vec<FilterScore>,
    /// Detected edge handling. `None` when `AnalysisConfig::detect_edges` is false.
    pub edge_mode: Option<EdgeMode>,
}

pub struct FilterScore {
    /// Which reference filter this score is against. Implements `Display`
    /// (e.g. "Lanczos3", or "Mitchell-Netravali(B=0.333, C=0.333)").
    pub filter: KnownFilter,
    /// Pearson correlation coefficient, in -1.0..=1.0. 1.0 is a perfect match.
    pub correlation: f64,
    /// Root-mean-square error between reconstructed and reference weights
    /// (filter-weight units; lower is better).
    pub rms_error: f64,
    /// Largest single absolute weight difference (filter-weight units).
    pub max_error: f64,
    /// Support radius detected from the reconstructed curve, in source pixels
    /// (outermost offset where |weight| > 0.005).
    pub detected_support: f64,
    /// Support radius the reference filter is defined to have, in source pixels.
    pub expected_support: f64,
}

pub struct FilterCurve {
    /// (offset, weight) sample points. Offset is in source-pixel units
    /// (distance from the filter center); weight is the normalized filter value.
    pub points: Vec<(f64, f64)>,
    /// Integral of the filter (~1.0 for a normalized filter).
    pub area: f64,
    /// Scale factor used: dst_width / src_width.
    pub scale_factor: f64,
    /// True for the dot pattern (scattered points), false for the line pattern
    /// (one connected curve).
    pub is_scatter: bool,
}
```

`EdgeMode` is an enum: `Clamp`, `Reflect`, `Wrap`, `Zero`, or `Unknown` (also
`Display`).

### Picking the best match

`AnalysisResult::best_match() -> Option<&FilterScore>` returns the entry with the
highest Pearson correlation (the first element of the already-sorted `scores`),
but only when its `correlation` exceeds `0.99` — otherwise it returns `None`. Use
the `filter` field (which is `Display`) to name it:

```rust
if let Some(best) = result.best_match() {
    println!("Detected: {} (r={:.4})", best.filter, best.correlation);
}
```

If you want the top candidate regardless of confidence, read `scores[0]`
directly instead.

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
