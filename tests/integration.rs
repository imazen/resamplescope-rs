use imgref::{ImgRef, ImgVec};
use resamplescope::{AnalysisConfig, KnownFilter};

/// Simple box (nearest-neighbor) resize for testing.
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

/// Bilinear resize for testing.
fn bilinear_resize(src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize) -> ImgVec<u8> {
    let mut dst = vec![0u8; dst_w * dst_h];
    let src_w = src.width();
    let src_h = src.height();

    for y in 0..dst_h {
        let sy = (y as f64 + 0.5) * src_h as f64 / dst_h as f64 - 0.5;
        let sy0 = (sy.floor() as isize).clamp(0, src_h as isize - 1) as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let fy = sy - sy.floor();

        for x in 0..dst_w {
            let sx = (x as f64 + 0.5) * src_w as f64 / dst_w as f64 - 0.5;
            let sx0 = (sx.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let fx = sx - sx.floor();

            let p00 = src.buf()[sy0 * src.stride() + sx0] as f64;
            let p10 = src.buf()[sy0 * src.stride() + sx1] as f64;
            let p01 = src.buf()[sy1 * src.stride() + sx0] as f64;
            let p11 = src.buf()[sy1 * src.stride() + sx1] as f64;

            let val = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            dst[y * dst_w + x] = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    ImgVec::new(dst, dst_w, dst_h)
}

#[test]
fn box_filter_detection() {
    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze(&nn_resize, &config).unwrap();

    assert!(result.downscale_curve.is_some());
    assert!(result.upscale_curve.is_some());
    assert!(!result.scores.is_empty());

    let best = &result.scores[0];
    assert_eq!(best.filter, KnownFilter::Box, "expected Box, got {}", best);
    assert!(
        best.correlation > 0.99,
        "correlation too low: {}",
        best.correlation
    );
}

#[test]
fn triangle_filter_detection() {
    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze(&bilinear_resize, &config).unwrap();

    let best = &result.scores[0];
    assert_eq!(
        best.filter,
        KnownFilter::Triangle,
        "expected Triangle, got {}",
        best
    );
    assert!(
        best.correlation > 0.99,
        "correlation too low: {}",
        best.correlation
    );
}

#[test]
fn perfect_resize_detected_correctly() {
    // Use the reference Lanczos3 resize as the callback.
    let resize_fn = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Lanczos3)
    };

    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze(&resize_fn, &config).unwrap();

    let best = &result.scores[0];
    assert_eq!(
        best.filter,
        KnownFilter::Lanczos3,
        "expected Lanczos3, got {}",
        best
    );
    assert!(
        best.correlation > 0.999,
        "correlation too low: {}",
        best.correlation
    );
}

#[test]
fn ssim_perfect_resize_against_itself() {
    let line = resamplescope::generate_line_pattern();
    let resized = resamplescope::perfect_resize(line.as_ref(), 555, 15, KnownFilter::Lanczos3);
    let s = resamplescope::ssim(resized.buf(), resized.buf(), 555, 15);
    assert!((s - 1.0).abs() < 1e-6, "ssim of identical: {s}");
}

#[test]
fn ssim_perfect_vs_nn() {
    let line = resamplescope::generate_line_pattern();
    let perfect = resamplescope::perfect_resize(line.as_ref(), 555, 15, KnownFilter::Lanczos3);
    let nn = nn_resize(line.as_ref(), 555, 15);
    let s = resamplescope::ssim(perfect.buf(), nn.buf(), 555, 15);
    // NN and Lanczos3 should differ significantly.
    assert!(s < 0.95, "ssim should differ: {s}");
}

#[test]
fn graph_rendering() {
    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze(&nn_resize, &config).unwrap();

    let graph = result.render_graph();
    assert_eq!(graph.width(), 600);
    assert_eq!(graph.height(), 300);

    // Also test with reference overlay.
    let graph_ref = result.render_graph_with_reference(KnownFilter::Box);
    assert_eq!(graph_ref.width(), 600);
    assert_eq!(graph_ref.height(), 300);
}

#[test]
fn weight_table_correctness() {
    // Verify that applying computed weights to a constant image produces the same constant.
    let weights = resamplescope::compute_weights(KnownFilter::Lanczos3, 15, 555);
    let src = [100u8; 15];
    let dst: Vec<u8> = weights
        .iter()
        .map(|pw| {
            let val: f64 = pw
                .entries
                .iter()
                .map(|e| src[e.src_pixel] as f64 * e.weight)
                .sum();
            val.round().clamp(0.0, 255.0) as u8
        })
        .collect();
    for (i, &v) in dst.iter().enumerate() {
        assert_eq!(v, 100, "pixel {i}: expected 100, got {v}");
    }
}

#[test]
fn upscale_only_analysis() {
    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze_upscale(&bilinear_resize, &config).unwrap();

    assert!(result.downscale_curve.is_none());
    assert!(result.upscale_curve.is_some());

    let best = &result.scores[0];
    assert_eq!(best.filter, KnownFilter::Triangle);
}

#[test]
fn downscale_only_analysis() {
    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result = resamplescope::analyze_downscale(&nn_resize, &config).unwrap();

    assert!(result.downscale_curve.is_some());
    assert!(result.upscale_curve.is_none());
}
