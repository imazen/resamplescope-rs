use imgref::{ImgRef, ImgVec};

use crate::filters::KnownFilter;

/// A single weight entry: which source pixel contributes and by how much.
#[derive(Debug, Clone)]
pub struct WeightEntry {
    pub src_pixel: usize,
    pub weight: f64,
}

/// The computed weights for a single output pixel.
#[derive(Debug, Clone)]
pub struct PixelWeights {
    pub entries: Vec<WeightEntry>,
}

/// Compute the exact pixel weight table for a 1D resize operation.
///
/// For each output pixel, returns the list of source pixels and their
/// normalized weights. Uses clamp edge handling (repeats the edge pixel
/// for out-of-bounds accesses).
pub fn compute_weights(filter: KnownFilter, src_size: usize, dst_size: usize) -> Vec<PixelWeights> {
    let scale = dst_size as f64 / src_size as f64;
    let filter_scale = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = filter.support() * filter_scale;

    let mut result = Vec::with_capacity(dst_size);

    for dst_x in 0..dst_size {
        // Center of this output pixel in source coordinates.
        let center = (dst_x as f64 + 0.5) / scale - 0.5;

        let left = (center - support).ceil() as isize;
        let right = (center + support).floor() as isize;

        let mut entries = Vec::new();
        let mut total = 0.0;

        for src_x in left..=right {
            // Clamp to valid range.
            let clamped = src_x.clamp(0, src_size as isize - 1) as usize;
            let distance = (src_x as f64 - center) / filter_scale;
            let w = filter.evaluate(distance);

            if w.abs() > 1e-12 {
                // Merge with existing entry for same clamped pixel.
                if let Some(existing) = entries
                    .iter_mut()
                    .find(|e: &&mut WeightEntry| e.src_pixel == clamped)
                {
                    existing.weight += w;
                } else {
                    entries.push(WeightEntry {
                        src_pixel: clamped,
                        weight: w,
                    });
                }
                total += w;
            }
        }

        // Normalize so weights sum to 1.
        if total.abs() > 1e-12 {
            for e in &mut entries {
                e.weight /= total;
            }
        }

        result.push(PixelWeights { entries });
    }

    result
}

/// Apply 1D weights to a row of source pixels, producing one output row.
fn apply_weights_row(weights: &[PixelWeights], src_row: &[u8]) -> Vec<u8> {
    weights
        .iter()
        .map(|pw| {
            let val: f64 = pw
                .entries
                .iter()
                .map(|e| src_row[e.src_pixel] as f64 * e.weight)
                .sum();
            val.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Generate the mathematically perfect resize output for a given filter.
///
/// Uses separable 2D resize: horizontal pass then vertical pass.
/// Edge handling is clamp (repeat edge pixel).
pub fn perfect_resize(
    src: ImgRef<'_, u8>,
    dst_width: usize,
    dst_height: usize,
    filter: KnownFilter,
) -> ImgVec<u8> {
    let h_weights = compute_weights(filter, src.width(), dst_width);

    // Horizontal pass: resize each row.
    let mut temp = vec![0u8; dst_width * src.height()];
    for y in 0..src.height() {
        let src_row = &src.buf()[y * src.stride()..][..src.width()];
        let dst_row = apply_weights_row(&h_weights, src_row);
        temp[y * dst_width..][..dst_width].copy_from_slice(&dst_row);
    }

    // Vertical pass (only if height changes).
    if dst_height == src.height() {
        return ImgVec::new(temp, dst_width, dst_height);
    }

    let v_weights = compute_weights(filter, src.height(), dst_height);
    let mut result = vec![0u8; dst_width * dst_height];

    for x in 0..dst_width {
        // Extract the column from temp.
        let col: Vec<u8> = (0..src.height()).map(|y| temp[y * dst_width + x]).collect();

        for (y, pw) in v_weights.iter().enumerate() {
            let val: f64 = pw
                .entries
                .iter()
                .map(|e| col[e.src_pixel] as f64 * e.weight)
                .sum();
            result[y * dst_width + x] = val.round().clamp(0.0, 255.0) as u8;
        }
    }

    ImgVec::new(result, dst_width, dst_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern;

    #[test]
    fn weights_sum_to_one() {
        for filter in crate::filters::KnownFilter::all_named() {
            let weights = compute_weights(*filter, 15, 555);
            for (i, pw) in weights.iter().enumerate() {
                let sum: f64 = pw.entries.iter().map(|e| e.weight).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "{}: pixel {i} weights sum to {sum}",
                    filter.name()
                );
            }
        }
    }

    #[test]
    fn weights_sum_to_one_downscale() {
        for filter in crate::filters::KnownFilter::all_named() {
            let weights = compute_weights(*filter, 557, 555);
            for (i, pw) in weights.iter().enumerate() {
                let sum: f64 = pw.entries.iter().map(|e| e.weight).sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "{}: pixel {i} weights sum to {sum}",
                    filter.name()
                );
            }
        }
    }

    #[test]
    fn perfect_resize_preserves_uniform() {
        // Resizing a uniform image should produce a uniform image.
        let src = ImgVec::new(vec![128u8; 15 * 15], 15, 15);
        let dst = perfect_resize(src.as_ref(), 555, 15, KnownFilter::Lanczos3);
        for &v in dst.buf() {
            assert_eq!(v, 128, "uniform image not preserved");
        }
    }

    #[test]
    fn perfect_resize_line_pattern() {
        let src = pattern::generate_line_pattern();
        let dst = perfect_resize(src.as_ref(), 555, 15, KnownFilter::Lanczos3);
        assert_eq!(dst.width(), 555);
        assert_eq!(dst.height(), 15);
        // Middle pixel should have the peak value.
        let mid = dst.buf()[(15 / 2) * 555 + 555 / 2];
        assert!(mid > 200, "peak should be bright, got {mid}");
    }
}
