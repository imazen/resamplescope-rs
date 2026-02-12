use crate::analyze::FilterCurve;
use crate::filters::KnownFilter;

/// Scoring result for one reference filter compared against a reconstructed curve.
#[derive(Debug, Clone)]
pub struct FilterScore {
    pub filter: KnownFilter,
    pub correlation: f64,
    pub rms_error: f64,
    pub max_error: f64,
    pub detected_support: f64,
    pub expected_support: f64,
}

impl std::fmt::Display for FilterScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: r={:.4} rms={:.4} max={:.4} support={:.1}/{:.1}",
            self.filter,
            self.correlation,
            self.rms_error,
            self.max_error,
            self.detected_support,
            self.expected_support
        )
    }
}

/// Bin scatter data into uniform intervals and average.
fn bin_scatter(points: &[(f64, f64)], bin_width: f64) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }

    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);

    let n_bins = ((max_x - min_x) / bin_width).ceil() as usize + 1;
    let mut sums = vec![0.0_f64; n_bins];
    let mut counts = vec![0_u32; n_bins];

    for &(x, y) in points {
        let bin = ((x - min_x) / bin_width).floor() as usize;
        let bin = bin.min(n_bins - 1);
        sums[bin] += y;
        counts[bin] += 1;
    }

    sums.iter()
        .zip(counts.iter())
        .enumerate()
        .filter(|&(_, (_, c))| *c > 0)
        .map(|(i, (&s, &c))| {
            let center = min_x + (i as f64 + 0.5) * bin_width;
            (center, s / c as f64)
        })
        .collect()
}

/// Compute Pearson correlation coefficient between two equal-length vectors.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }

    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    cov / denom
}

/// Detect the support radius from a curve (outermost offset where |weight| > threshold).
fn detect_support(points: &[(f64, f64)], threshold: f64) -> f64 {
    points
        .iter()
        .filter(|(_, w)| w.abs() > threshold)
        .map(|(x, _)| x.abs())
        .fold(0.0_f64, f64::max)
}

/// Score a reconstructed curve against a single reference filter.
pub fn score_against(curve: &FilterCurve, filter: KnownFilter) -> FilterScore {
    // For scatter data, bin first; for connected data, use directly.
    let comparison_points = if curve.is_scatter {
        bin_scatter(&curve.points, 0.02)
    } else {
        curve.points.clone()
    };

    if comparison_points.is_empty() {
        return FilterScore {
            filter,
            correlation: 0.0,
            rms_error: f64::INFINITY,
            max_error: f64::INFINITY,
            detected_support: 0.0,
            expected_support: filter.support(),
        };
    }

    // Evaluate reference filter at the same offsets.
    let actual: Vec<f64> = comparison_points.iter().map(|p| p.1).collect();
    let reference: Vec<f64> = comparison_points
        .iter()
        .map(|p| filter.evaluate(p.0))
        .collect();

    let correlation = pearson(&actual, &reference);

    // RMS error
    let n = actual.len() as f64;
    let rms_error = (actual
        .iter()
        .zip(reference.iter())
        .map(|(a, r)| (a - r).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();

    // Max error
    let max_error = actual
        .iter()
        .zip(reference.iter())
        .map(|(a, r)| (a - r).abs())
        .fold(0.0_f64, f64::max);

    let detected_support = detect_support(&comparison_points, 0.005);

    FilterScore {
        filter,
        correlation,
        rms_error,
        max_error,
        detected_support,
        expected_support: filter.support(),
    }
}

/// Score a curve against all built-in filters, returning results sorted by correlation (best first).
pub fn score_against_all(curve: &FilterCurve) -> Vec<FilterScore> {
    let mut scores: Vec<FilterScore> = KnownFilter::all_named()
        .iter()
        .map(|&f| score_against(curve, f))
        .collect();

    scores.sort_by(|a, b| {
        b.correlation
            .partial_cmp(&a.correlation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scores
}

/// Compute SSIM between two equal-sized grayscale images.
/// Uses 8x8 block-based comparison with standard SSIM constants.
pub fn ssim(a: &[u8], b: &[u8], width: usize, height: usize) -> f64 {
    const K1: f64 = 0.01;
    const K2: f64 = 0.03;
    const L: f64 = 255.0;
    let c1 = (K1 * L) * (K1 * L);
    let c2 = (K2 * L) * (K2 * L);
    const BLOCK: usize = 8;

    assert_eq!(a.len(), width * height);
    assert_eq!(b.len(), width * height);

    if width < BLOCK || height < BLOCK {
        // Fall back to global SSIM for very small images.
        let mean_a: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / a.len() as f64;
        let mean_b: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / b.len() as f64;
        let var_a: f64 =
            a.iter().map(|&v| (v as f64 - mean_a).powi(2)).sum::<f64>() / a.len() as f64;
        let var_b: f64 =
            b.iter().map(|&v| (v as f64 - mean_b).powi(2)).sum::<f64>() / b.len() as f64;
        let cov: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&va, &vb)| (va as f64 - mean_a) * (vb as f64 - mean_b))
            .sum::<f64>()
            / a.len() as f64;

        let num = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
        let den = (mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2);
        return num / den;
    }

    let blocks_x = width / BLOCK;
    let blocks_y = height / BLOCK;
    let mut total_ssim = 0.0;
    let mut count = 0;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum_a = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let mut sum_aa = 0.0_f64;
            let mut sum_bb = 0.0_f64;
            let mut sum_ab = 0.0_f64;
            let n = (BLOCK * BLOCK) as f64;

            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let y = by * BLOCK + dy;
                    let x = bx * BLOCK + dx;
                    let va = a[y * width + x] as f64;
                    let vb = b[y * width + x] as f64;
                    sum_a += va;
                    sum_b += vb;
                    sum_aa += va * va;
                    sum_bb += vb * vb;
                    sum_ab += va * vb;
                }
            }

            let mean_a = sum_a / n;
            let mean_b = sum_b / n;
            let var_a = sum_aa / n - mean_a * mean_a;
            let var_b = sum_bb / n - mean_b * mean_b;
            let cov = sum_ab / n - mean_a * mean_b;

            let num = (2.0 * mean_a * mean_b + c1) * (2.0 * cov + c2);
            let den = (mean_a.powi(2) + mean_b.powi(2) + c1) * (var_a + var_b + c2);
            total_ssim += num / den;
            count += 1;
        }
    }

    if count == 0 {
        return 1.0;
    }
    total_ssim / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_negative_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn ssim_identical_images() {
        let img = vec![128u8; 64 * 64];
        let s = ssim(&img, &img, 64, 64);
        assert!((s - 1.0).abs() < 1e-6, "ssim of identical images: {s}");
    }

    #[test]
    fn ssim_different_images() {
        let a = vec![0u8; 64 * 64];
        let b = vec![255u8; 64 * 64];
        let s = ssim(&a, &b, 64, 64);
        assert!(s < 0.1, "ssim of opposite images should be low: {s}");
    }

    #[test]
    fn score_triangle_against_triangle() {
        // Generate a synthetic triangle filter curve
        let points: Vec<(f64, f64)> = (-100..=100)
            .map(|i| {
                let x = i as f64 / 100.0;
                (x, KnownFilter::Triangle.evaluate(x))
            })
            .collect();

        let curve = FilterCurve {
            points,
            area: 1.0,
            scale_factor: 37.0,
            is_scatter: false,
        };

        let score = score_against(&curve, KnownFilter::Triangle);
        assert!(
            score.correlation > 0.999,
            "correlation: {}",
            score.correlation
        );
        assert!(score.rms_error < 0.001, "rms: {}", score.rms_error);
    }

    #[test]
    fn score_all_sorts_correctly() {
        let points: Vec<(f64, f64)> = (-300..=300)
            .map(|i| {
                let x = i as f64 / 100.0;
                (x, KnownFilter::Lanczos3.evaluate(x))
            })
            .collect();

        let curve = FilterCurve {
            points,
            area: 1.0,
            scale_factor: 37.0,
            is_scatter: false,
        };

        let scores = score_against_all(&curve);
        assert_eq!(scores[0].filter, KnownFilter::Lanczos3);
        assert!(scores[0].correlation > 0.999);
    }
}
