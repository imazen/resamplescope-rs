#![forbid(unsafe_code)]

//! Reverse-engineer the resampling filter used by any image resizer.
//!
//! Port of [ResampleScope](http://entropymine.com/resamplescope/) by Jason Summers
//! (~1400 lines of C, GPL-3.0-or-later). The original tool generates test PNG images,
//! has the resizer process them externally, then reconstructs the filter kernel shape
//! from the output.
//!
//! This port replaces that file-based workflow with a **callback API**: you provide a
//! resize closure, and the crate handles everything in-memory. It adds scoring against
//! reference filters, SSIM comparison against perfect reference weight tables, and edge
//! handling detection.
//!
//! ## License
//!
//! AGPL-3.0-or-later. See [`LICENSE`](https://github.com/imazen/resamplescope-rs/blob/main/LICENSE).
//!
//! ## Original work
//!
//! - **Author**: Jason Summers
//! - **Website**: <http://entropymine.com/resamplescope/>
//! - **License**: GPL-3.0-or-later
//!
//! This crate is a derivative work. The test pattern generation and filter
//! reconstruction algorithms are ported from the original C source.
//!
//! ## Reference resize implementation
//!
//! The [`reference`] module (weight table computation, separable 2D resize) is
//! derived from [imageflow](https://github.com/imazen/imageflow) by Imazen.
//! The reference filter math in [`filters`] uses standard mathematical definitions
//! (sinc, Mitchell-Netravali, etc.).

pub mod analyze;
pub mod edge;
pub mod filters;
pub mod graph;
pub mod pattern;
pub mod reference;
pub mod score;

use imgref::{ImgRef, ImgVec};
use rgb::RGB8;

pub use analyze::FilterCurve;
pub use edge::EdgeMode;
pub use filters::KnownFilter;
pub use reference::{PixelWeights, WeightEntry};
pub use score::FilterScore;

/// The resize callback type: takes a grayscale source image and target dimensions,
/// returns the resized grayscale image.
pub type ResizeFn = dyn Fn(ImgRef<'_, u8>, usize, usize) -> ImgVec<u8>;

/// Configuration for analysis.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Whether the resizer operates in sRGB colorspace (converts to linear before resize).
    pub srgb: bool,
    /// Whether to detect edge handling mode.
    pub detect_edges: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            srgb: false,
            detect_edges: true,
        }
    }
}

/// Complete analysis result from probing a resizer.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Reconstructed filter from dot pattern (downscale, 557->555).
    pub downscale_curve: Option<FilterCurve>,
    /// Reconstructed filter from line pattern (upscale, 15->555).
    pub upscale_curve: Option<FilterCurve>,
    /// Scores against known reference filters, sorted best-first by correlation.
    pub scores: Vec<FilterScore>,
    /// Detected edge handling mode, if requested.
    pub edge_mode: Option<EdgeMode>,
}

impl AnalysisResult {
    /// Returns the best-matching filter if correlation exceeds 0.99.
    pub fn best_match(&self) -> Option<&FilterScore> {
        self.scores.first().filter(|s| s.correlation > 0.99)
    }

    /// Render a scope graph showing the reconstructed filter curve(s).
    pub fn render_graph(&self) -> ImgVec<RGB8> {
        graph::render(
            self.downscale_curve.as_ref(),
            self.upscale_curve.as_ref(),
            None,
        )
    }

    /// Render a scope graph with a reference filter overlay.
    pub fn render_graph_with_reference(&self, filter: KnownFilter) -> ImgVec<RGB8> {
        graph::render(
            self.downscale_curve.as_ref(),
            self.upscale_curve.as_ref(),
            Some(filter),
        )
    }
}

/// Error type for analysis operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "resize callback returned wrong dimensions: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
    )]
    WrongDimensions {
        expected_w: usize,
        expected_h: usize,
        actual_w: usize,
        actual_h: usize,
    },
    #[error("analysis produced no usable data")]
    NoData,
}

fn check_dimensions(img: &ImgVec<u8>, expected_w: usize, expected_h: usize) -> Result<(), Error> {
    if img.width() != expected_w || img.height() != expected_h {
        return Err(Error::WrongDimensions {
            expected_w,
            expected_h,
            actual_w: img.width(),
            actual_h: img.height(),
        });
    }
    Ok(())
}

/// Run both downscale and upscale analysis, score against known filters,
/// and optionally detect edge handling.
pub fn analyze(resize: &ResizeFn, config: &AnalysisConfig) -> Result<AnalysisResult, Error> {
    // Downscale analysis (dot pattern).
    let dot_src = pattern::generate_dot_pattern();
    let (dot_w, dot_h) = analyze::dot_target();
    let dot_resized = resize(dot_src.as_ref(), dot_w, dot_h);
    check_dimensions(&dot_resized, dot_w, dot_h)?;
    let downscale_curve = analyze::analyze_dot(&dot_resized.as_ref(), config.srgb);

    // Upscale analysis (line pattern).
    let line_src = pattern::generate_line_pattern();
    let (line_w, line_h) = analyze::line_target();
    let line_resized = resize(line_src.as_ref(), line_w, line_h);
    check_dimensions(&line_resized, line_w, line_h)?;
    let upscale_curve = analyze::analyze_line(&line_resized.as_ref(), config.srgb);

    // Score using the upscale curve (higher resolution, cleaner data).
    // Fall back to downscale if upscale has no points.
    let scoring_curve = if !upscale_curve.points.is_empty() {
        &upscale_curve
    } else if !downscale_curve.points.is_empty() {
        &downscale_curve
    } else {
        return Err(Error::NoData);
    };

    let scores = score::score_against_all(scoring_curve);

    // Edge detection.
    let edge_mode = if config.detect_edges {
        Some(edge::detect(resize))
    } else {
        None
    };

    Ok(AnalysisResult {
        downscale_curve: Some(downscale_curve),
        upscale_curve: Some(upscale_curve),
        scores,
        edge_mode,
    })
}

/// Run only the downscale analysis (dot pattern, 557->555).
pub fn analyze_downscale(
    resize: &ResizeFn,
    config: &AnalysisConfig,
) -> Result<AnalysisResult, Error> {
    let dot_src = pattern::generate_dot_pattern();
    let (dot_w, dot_h) = analyze::dot_target();
    let dot_resized = resize(dot_src.as_ref(), dot_w, dot_h);
    check_dimensions(&dot_resized, dot_w, dot_h)?;
    let downscale_curve = analyze::analyze_dot(&dot_resized.as_ref(), config.srgb);

    if downscale_curve.points.is_empty() {
        return Err(Error::NoData);
    }

    let scores = score::score_against_all(&downscale_curve);

    Ok(AnalysisResult {
        downscale_curve: Some(downscale_curve),
        upscale_curve: None,
        scores,
        edge_mode: None,
    })
}

/// Run only the upscale analysis (line pattern, 15->555).
pub fn analyze_upscale(
    resize: &ResizeFn,
    config: &AnalysisConfig,
) -> Result<AnalysisResult, Error> {
    let line_src = pattern::generate_line_pattern();
    let (line_w, line_h) = analyze::line_target();
    let line_resized = resize(line_src.as_ref(), line_w, line_h);
    check_dimensions(&line_resized, line_w, line_h)?;
    let upscale_curve = analyze::analyze_line(&line_resized.as_ref(), config.srgb);

    if upscale_curve.points.is_empty() {
        return Err(Error::NoData);
    }

    let scores = score::score_against_all(&upscale_curve);

    let edge_mode = if config.detect_edges {
        Some(edge::detect(resize))
    } else {
        None
    };

    Ok(AnalysisResult {
        downscale_curve: None,
        upscale_curve: Some(upscale_curve),
        scores,
        edge_mode,
    })
}

// Re-export convenience functions from pattern.
pub use pattern::{generate_dot_pattern, generate_line_pattern};

// Re-export reference resize functions.
pub use reference::{compute_weights, perfect_resize};

// Re-export SSIM.
pub use score::ssim;
