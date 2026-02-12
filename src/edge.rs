use crate::pattern::{self, BRIGHT, DARK, LINE_DST_WIDTH, LINE_SRC_WIDTH};

/// Detected edge handling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// Clamps to the nearest edge pixel.
    Clamp,
    /// Reflects across the edge.
    Reflect,
    /// Wraps around to the opposite edge.
    Wrap,
    /// Treats out-of-bounds as zero (black).
    Zero,
    /// Could not determine edge handling.
    Unknown,
}

impl std::fmt::Display for EdgeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clamp => f.write_str("Clamp"),
            Self::Reflect => f.write_str("Reflect"),
            Self::Wrap => f.write_str("Wrap"),
            Self::Zero => f.write_str("Zero"),
            Self::Unknown => f.write_str("Unknown"),
        }
    }
}

/// Detect the edge handling mode used by a resizer.
///
/// Generates a test image with a bright column at x=1 (near the left edge),
/// resizes it, then analyzes the asymmetry of the filter response near the
/// boundary to classify the edge handling strategy.
pub fn detect(resize: &crate::ResizeFn) -> EdgeMode {
    let edge_img = pattern::generate_edge_pattern();
    let dst_w = LINE_DST_WIDTH;
    let dst_h = edge_img.height();
    let resized = resize(edge_img.as_ref(), dst_w, dst_h);

    if resized.width() != dst_w || resized.height() != dst_h {
        return EdgeMode::Unknown;
    }

    let scale_factor = dst_w as f64 / LINE_SRC_WIDTH as f64;
    let scanline = resized.height() / 2;
    let row = &resized.buf()[scanline * resized.stride()..][..dst_w];

    // Convert to normalized weights.
    let weights: Vec<f64> = row
        .iter()
        .map(|&v| (v as f64 - DARK as f64) / (BRIGHT as f64 - DARK as f64))
        .collect();

    // Find the peak (should be near x=1 * scale_factor).
    let expected_peak = ((1.0 + 0.5) * scale_factor - 0.5) as usize;
    let search_start = expected_peak.saturating_sub(5);
    let search_end = (expected_peak + 6).min(dst_w);
    let peak_idx = (search_start..search_end)
        .max_by(|&a, &b| weights[a].partial_cmp(&weights[b]).unwrap())
        .unwrap_or(expected_peak);

    // Compute energy on each side of the peak.
    // Left side: from pixel 0 to peak (edge-influenced).
    // Right side: mirror of the left side, away from edge (clean interior).
    let left_extent = peak_idx;
    let right_extent = (dst_w - 1 - peak_idx).min(left_extent);

    // Use matching extents for fair comparison.
    let extent = left_extent.min(right_extent).min(dst_w / 4);

    if extent < 3 {
        return EdgeMode::Unknown;
    }

    let left_energy: f64 = (1..=extent).map(|d| weights[peak_idx - d].abs()).sum();
    let right_energy: f64 = (1..=extent).map(|d| weights[peak_idx + d].abs()).sum();

    // Check for wrap: energy at the far-right side of the image.
    // If wrap is active, the bright column at x=1 wraps to near x=14,
    // which maps to the far-right of the output.
    let far_right_start = dst_w.saturating_sub((2.0 * scale_factor) as usize);
    let far_energy: f64 = (far_right_start..dst_w)
        .map(|i| weights[i].abs())
        .sum::<f64>()
        / (dst_w - far_right_start) as f64;

    // Check for negative values on left side (indicator of zero padding).
    let left_has_negative = (0..peak_idx).any(|i| weights[i] < -0.03);

    // Energy ratio: left/right. Values close to 1.0 mean symmetric (clamp-like).
    let energy_ratio = if right_energy > 1e-6 {
        left_energy / right_energy
    } else {
        1.0
    };

    // Classify based on observed patterns:
    if left_has_negative || energy_ratio < 0.5 {
        // Zero padding creates missing contributions or negative artifacts.
        EdgeMode::Zero
    } else if far_energy > 0.02 {
        // Wrap causes energy at the far end of the image.
        EdgeMode::Wrap
    } else if energy_ratio > 1.5 {
        // Reflect doubles the bright column's contribution on the left side.
        EdgeMode::Reflect
    } else if energy_ratio > 0.7 {
        // Clamp preserves the filter shape (dark background extends naturally).
        EdgeMode::Clamp
    } else {
        EdgeMode::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use imgref::{ImgRef, ImgVec};

    /// Clamp-based nearest-neighbor resize.
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
    fn nn_produces_some_result() {
        let mode = detect(&nn_resize);
        // Nearest-neighbor has zero filter extent, so edge detection
        // results are not meaningful. Just verify it doesn't panic.
        let _ = mode;
    }
}
