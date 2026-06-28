<!-- GENERATED FROM README.md by zenutils gen-readme-crates.sh — DO NOT EDIT. -->

# resamplescope-rs

resamplescope-rs reverse-engineers the resampling filter behind any image resizer. Hand it a resize closure and it reconstructs the filter kernel's shape, names the closest match (Box, Triangle, Hermite, Catmull-Rom, Mitchell, B-Spline, Lanczos2/3/4), reports correlation and support radius, detects the edge-handling mode, and renders a scope graph. A Rust port of Jason Summers' [ResampleScope](http://entropymine.com/resamplescope/), trading the original's file-based round-trip for an in-memory callback API. Pure Rust, `#![forbid(unsafe_code)]`, no image-codec dependencies.

## Quick start

The public API hands you `imgref` and `rgb` types directly: your resize closure takes an `ImgRef<'_, u8>` and returns an `ImgVec<u8>`, and graphs come back as `ImgVec<RGB8>`. Add both alongside this crate, matching the versions it builds against:

```toml
[dependencies]
resamplescope-rs = "0.1.0"
imgref = "1"
rgb = { version = "0.8", default-features = false }
```

```rust
use imgref::{ImgRef, ImgVec};
use resamplescope::{analyze, AnalysisConfig};

// Wrap your resizer so it takes a grayscale source and target dimensions.
fn my_resize(src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize) -> ImgVec<u8> {
    // ...call your real resizer here, returning a dst_w x dst_h grayscale image...
    todo!()
}

let result = analyze(&my_resize, &AnalysisConfig::default()).unwrap();

// best_match() names the filter only when correlation > 0.99.
if let Some(best) = result.best_match() {
    println!("Detected filter: {} (r={:.4})", best.filter, best.correlation);
}

// Render a 600x300 scope graph as ImgVec<RGB8> (encode it with your own codec).
let graph = result.render_graph();
```

The package is `resamplescope-rs`; the library is `resamplescope`, so imports read `use resamplescope::...`. `rgb` uses `default-features = false` because only the plain `RGB8` struct is needed for graph output.

## How it works

1. Generates test patterns (a dot grid for downscale, a single bright line for upscale).
2. Passes them through your resize function.
3. Reconstructs the filter kernel from the output pixel values.
4. Scores the result against known filters (Box, Triangle, Hermite, Mitchell, Lanczos, ...).
5. Returns the best match with correlation, RMS error, and detected support radius.

`analyze` runs both directions; `analyze_downscale` (dot pattern, 557->555) and `analyze_upscale` (line pattern, 15->555) run just one when that's all you need.

## Configuration

`AnalysisConfig` has exactly two fields. It is not `#[non_exhaustive]`, so you can build it with a struct literal or start from `AnalysisConfig::default()`:

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

`analyze()` returns `Result<AnalysisResult, Error>`. Everything below is a plain public **field** (not an accessor method), so you read it directly. None of these structs is `#[non_exhaustive]`. `Error` is `WrongDimensions { .. }` (your closure returned the wrong output size) or `NoData` (reconstruction found no usable points).

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

`EdgeMode` is an enum: `Clamp`, `Reflect`, `Wrap`, `Zero`, or `Unknown` (all `Display`).

### Picking the best match

`AnalysisResult::best_match() -> Option<&FilterScore>` returns the entry with the highest Pearson correlation (the first element of the already-sorted `scores`), but only when its `correlation` exceeds `0.99` — otherwise it returns `None`. Use the `filter` field (which is `Display`) to name it:

```rust
if let Some(best) = result.best_match() {
    println!("Detected: {} (r={:.4})", best.filter, best.correlation);
}
```

If you want the top candidate regardless of confidence, read `scores[0]` directly instead.

## Graphs

`AnalysisResult::render_graph() -> ImgVec<RGB8>` draws a 600x300 scope plot of the reconstructed curve(s). `render_graph_with_reference(KnownFilter)` overlays a named reference filter so you can eyeball the fit:

```rust
use resamplescope::KnownFilter;

let graph = result.render_graph_with_reference(KnownFilter::Lanczos3);
// graph is ImgVec<RGB8> — encode it with zenpng, the png crate, etc.
```

## Known filters

`KnownFilter`: `Box`, `Triangle`, `Hermite`, `CatmullRom`, `Mitchell`, `BSpline`, `Lanczos2`, `Lanczos3`, `Lanczos4`, plus an arbitrary `MitchellNetravali { b, c }` parameterization. Scoring (`KnownFilter::all_named()`) compares against the nine fixed filters.

## Reference resize and SSIM

The `reference` module produces mathematically exact resize output for any built-in filter — useful both as a stand-in resizer and as the ground truth in an SSIM comparison:

```rust
use resamplescope::{compute_weights, perfect_resize, ssim, KnownFilter};

// Exact separable resize (clamp edges) with a chosen filter:
let resized = perfect_resize(src.as_ref(), 555, 15, KnownFilter::Lanczos3);

// The 1D weight table behind it (per output pixel: source pixels + weights):
let weights = compute_weights(KnownFilter::Lanczos3, 15, 555);

// Block-based SSIM between two equal-size grayscale buffers:
let score = ssim(a, b, width, height); // -> f64, 1.0 == identical
```

## No I/O

This crate has no image-codec dependencies. It works in `ImgVec<u8>` (grayscale) and `ImgVec<RGB8>` (graph output); you bring your own PNG/JPEG encoding.

## Original work

- **Author**: Jason Summers
- **Website**: <http://entropymine.com/resamplescope/>
- **License of original**: GPL-3.0-or-later

The test-pattern generation and filter-reconstruction algorithms are ported from the original C source. The reference filter math uses standard mathematical definitions (sinc, Mitchell-Netravali, etc.); the authoritative, optimized implementations of these filters live in [imageflow](https://github.com/imazen/imageflow).

## License

AGPL-3.0-or-later. See [LICENSE](https://github.com/imazen/resamplescope-rs/blob/main/LICENSE). As a derivative of the GPL-3.0-or-later original, distribution stays under copyleft.

## Image tech I maintain

| | |
|:--|:--|
| **Codecs** ¹ | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] · [zenjxl] · [zenbitmaps] · [heic] · [zentiff] · [zenpdf] · [zensvg] · [zenjp2] · [zenraw] · [ultrahdr] |
| Codec internals | [zenjxl-decoder] · [jxl-encoder] · [zenrav1e] · [rav1d-safe] · [zenavif-parse] · [zenavif-serialize] |
| Compression | [zenflate] · [zenzop] · [zenzstd] |
| Processing | [zenresize] · [zenquant] · [zenblend] · [zenfilters] · [zensally] · [zentone] |
| Pixels & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline & framework | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] · [zenwasm] · [zentract] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [zenmetrics] · **resamplescope-rs** |
| Pickers & ML | [zenanalyze] · [zenpredict] · [zenpicker] |
| Products | [Imageflow] image engine ([.NET][imageflow-dotnet] · [Node][imageflow-node] · [Go][imageflow-go]) · [Imageflow Server] · [ImageResizer] (C#) |

<sub>¹ pure-Rust, `#![forbid(unsafe_code)]` codecs, as of 2026</sub>

### General Rust awesomeness

[zenbench] · [archmage] · [magetypes] · [enough] · [whereat] · [cargo-copter]

[Open source](https://www.imazen.io/open-source) · [@imazen](https://github.com/imazen) · [@lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith)

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic
[zentiff]: https://github.com/imazen/zentiff
[zenpdf]: https://github.com/imazen/zenpdf
[zensvg]: https://github.com/imazen/zenextras
[zenjp2]: https://github.com/imazen/zenextras
[zenraw]: https://github.com/imazen/zenraw
[ultrahdr]: https://github.com/imazen/ultrahdr
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenrav1e]: https://github.com/imazen/zenrav1e
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenzstd]: https://github.com/imazen/zenzstd
[zenresize]: https://github.com/imazen/zenresize
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zenfilters]: https://github.com/imazen/zenfilters
[zensally]: https://github.com/imazen/zensally
[zentone]: https://github.com/imazen/zentone
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[zenwasm]: https://github.com/imazen/zenwasm
[zentract]: https://github.com/imazen/zentract
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenmetrics]: https://github.com/imazen/zenmetrics
[zenanalyze]: https://github.com/imazen/zenanalyze
[zenpredict]: https://github.com/imazen/zenanalyze
[zenpicker]: https://github.com/imazen/zenanalyze
[zenbench]: https://github.com/imazen/zenbench
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[cargo-copter]: https://github.com/imazen/cargo-copter
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[ImageResizer]: https://github.com/imazen/resizer
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
