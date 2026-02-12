use imgref::ImgRef;

use crate::pattern::{
    BRIGHT, DARK, DOT_DST_HEIGHT, DOT_DST_WIDTH, DOT_HCENTER, DOT_HPIXELSPAN, DOT_NUM_STRIPS,
    DOT_SRC_WIDTH, DOT_STRIP_HEIGHT, LINE_DST_HEIGHT, LINE_DST_WIDTH, LINE_SRC_WIDTH,
};

/// A reconstructed filter curve from analysis.
#[derive(Debug, Clone)]
pub struct FilterCurve {
    /// (offset, weight) sample points. Offset is in source-pixel units
    /// (distance from the filter center). Weight is the normalized filter value.
    pub points: Vec<(f64, f64)>,
    /// Integral of the filter (should be ~1.0 for a normalized filter).
    pub area: f64,
    /// Scale factor used: dst_width / src_width.
    pub scale_factor: f64,
    /// True for dot pattern (scatter), false for line pattern (connected).
    pub is_scatter: bool,
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Read a pixel value, optionally applying sRGB correction.
/// Returns a value in the range where DARK=50 and BRIGHT=250.
fn read_pixel(img: &ImgRef<'_, u8>, x: usize, y: usize, srgb: bool) -> f64 {
    let raw = img.buf()[y * img.stride() + x] as f64;
    if srgb {
        let srgb50_lin = srgb_to_linear(50.0 / 255.0);
        let srgb250_lin = srgb_to_linear(250.0 / 255.0);
        let v_lin = srgb_to_linear(raw / 255.0);
        (v_lin - srgb50_lin) * ((BRIGHT as f64 - DARK as f64) / (srgb250_lin - srgb50_lin))
            + DARK as f64
    } else {
        raw
    }
}

/// Reconstruct the filter curve from a resized dot pattern image (downscale analysis).
///
/// The dot pattern has 25 strips, each with bright dots at phase-offset positions.
/// By analyzing where each output pixel falls relative to the nearest dot,
/// we reconstruct the filter kernel as a scatter plot.
pub fn analyze_dot(img: &ImgRef<'_, u8>, srgb: bool) -> FilterCurve {
    let w = img.width();
    let h = img.height();
    let scale_factor = w as f64 / DOT_SRC_WIDTH as f64;

    assert_eq!(
        h, DOT_DST_HEIGHT,
        "dot image height must be {DOT_DST_HEIGHT}"
    );

    let mut points = Vec::new();

    for strip in 0..DOT_NUM_STRIPS {
        for dstpos in 0..w {
            // Find nearest zero-point for this strip and output pixel.
            let mut offset = 10000.0_f64;

            let mut k = DOT_HCENTER + strip;
            while k < DOT_SRC_WIDTH - DOT_HCENTER {
                // Convert source dot position to target image coordinates.
                let zp = scale_factor * (k as f64 + 0.5 - DOT_SRC_WIDTH as f64 / 2.0)
                    + (w as f64 / 2.0)
                    - 0.5;

                let tmp_offset = dstpos as f64 - zp;

                if tmp_offset.abs() < offset.abs() {
                    offset = tmp_offset;
                }

                k += DOT_HPIXELSPAN;
            }

            // Skip points too far from any dot.
            if offset.abs() > scale_factor * DOT_HCENTER as f64 {
                continue;
            }

            // Sum vertically across the strip to undo vertical blur.
            let mut tot = 0.0;
            for row in 0..DOT_STRIP_HEIGHT {
                let y = DOT_STRIP_HEIGHT * strip + row;
                let v = read_pixel(img, dstpos, y, srgb);
                tot += v - DARK as f64;
            }

            // Convert to normalized weight.
            let mut weight = tot / (BRIGHT as f64 - DARK as f64);

            if scale_factor < 1.0 {
                // Downscale: compensate for pixel size reduction.
                weight /= scale_factor;
            } else {
                // Upscale: convert offset to source-pixel units.
                offset /= scale_factor;
            }

            points.push((offset, weight));
        }
    }

    FilterCurve {
        points,
        area: 0.0, // Not well-defined for scatter data
        scale_factor,
        is_scatter: true,
    }
}

/// Reconstruct the filter curve from a resized line pattern image (upscale analysis).
///
/// The line pattern is a single bright column that, when upscaled, directly reveals
/// the filter kernel shape as a connected curve.
pub fn analyze_line(img: &ImgRef<'_, u8>, srgb: bool) -> FilterCurve {
    let w = img.width();
    let h = img.height();
    let scale_factor = w as f64 / LINE_SRC_WIDTH as f64;
    let scanline = h / 2;

    let mut points = Vec::new();
    let mut tot = 0.0;

    for i in 0..w {
        // Read from cycling scanlines (±1 row) as a consistency check,
        // matching the C source behavior.
        let y = if h >= 3 {
            let cycle_offset = (i % 3) as isize - 1;
            (scanline as isize + cycle_offset).clamp(0, h as isize - 1) as usize
        } else {
            scanline
        };

        let v = read_pixel(img, i, y, srgb);
        let mut weight = (v - DARK as f64) / (BRIGHT as f64 - DARK as f64);
        tot += weight;

        let mut offset = 0.5 + i as f64 - (w as f64 / 2.0);

        if scale_factor < 1.0 {
            weight /= scale_factor;
        } else {
            offset /= scale_factor;
        }

        points.push((offset, weight));
    }

    let area = tot / scale_factor;

    FilterCurve {
        points,
        area,
        scale_factor,
        is_scatter: false,
    }
}

/// Expected target dimensions for the dot pattern resize.
pub fn dot_target() -> (usize, usize) {
    (DOT_DST_WIDTH, DOT_DST_HEIGHT)
}

/// Expected target dimensions for the line pattern resize.
pub fn line_target() -> (usize, usize) {
    (LINE_DST_WIDTH, LINE_DST_HEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern;
    use imgref::ImgVec;

    /// Nearest-neighbor resize for testing.
    fn nn_resize(src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize) -> ImgVec<u8> {
        let mut dst = vec![0u8; dst_w * dst_h];
        for y in 0..dst_h {
            let sy = ((y as f64 + 0.5) * src.height() as f64 / dst_h as f64 - 0.5)
                .round()
                .clamp(0.0, (src.height() - 1) as f64) as usize;
            for x in 0..dst_w {
                let sx = ((x as f64 + 0.5) * src.width() as f64 / dst_w as f64 - 0.5)
                    .round()
                    .clamp(0.0, (src.width() - 1) as f64) as usize;
                dst[y * dst_w + x] = src.buf()[sy * src.stride() + sx];
            }
        }
        ImgVec::new(dst, dst_w, dst_h)
    }

    #[test]
    fn dot_analysis_produces_points() {
        let dot = pattern::generate_dot_pattern();
        let (tw, th) = dot_target();
        let resized = nn_resize(dot.as_ref(), tw, th);
        let curve = analyze_dot(&resized.as_ref(), false);
        assert!(!curve.points.is_empty());
        assert!(curve.scale_factor < 1.0);
    }

    #[test]
    fn line_analysis_produces_curve() {
        let line = pattern::generate_line_pattern();
        let (tw, th) = line_target();
        let resized = nn_resize(line.as_ref(), tw, th);
        let curve = analyze_line(&resized.as_ref(), false);
        assert_eq!(curve.points.len(), tw);
        assert!(curve.scale_factor > 1.0);
        // Area should be roughly 1.0 for a normalized filter
        assert!((curve.area - 1.0).abs() < 0.5, "area = {}", curve.area);
    }
}
