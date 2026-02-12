#![cfg(feature = "c-reference")]

//! Comparison tests between the Rust port and the original C implementation.
//!
//! Run with: cargo test --features c-reference --test c_comparison

use imgref::{ImgRef, ImgVec};
use resamplescope::KnownFilter;

// FFI declarations for the C shim.
unsafe extern "C" {
    fn rs_generate_dot_pattern(buf: *mut u8);
    fn rs_generate_line_pattern(buf: *mut u8);

    fn rs_analyze_dot(
        resized: *const u8,
        w: i32,
        h: i32,
        srgb: i32,
        offsets: *mut f64,
        weights: *mut f64,
        out_count: *mut i32,
    ) -> f64;

    fn rs_analyze_line(
        resized: *const u8,
        w: i32,
        h: i32,
        srgb: i32,
        offsets: *mut f64,
        weights: *mut f64,
        out_area: *mut f64,
    ) -> f64;

    fn rs_dot_src_width() -> i32;
    fn rs_dot_src_height() -> i32;
    fn rs_dot_dst_width() -> i32;
    fn rs_dot_dst_height() -> i32;
    fn rs_line_src_width() -> i32;
    fn rs_line_src_height() -> i32;
    fn rs_line_dst_width() -> i32;
    fn rs_line_dst_height() -> i32;
}

// ---------- Constants ----------

#[test]
fn constants_match() {
    unsafe {
        assert_eq!(rs_dot_src_width(), 557);
        assert_eq!(rs_dot_src_height(), 275);
        assert_eq!(rs_dot_dst_width(), 555);
        assert_eq!(rs_dot_dst_height(), 275);
        assert_eq!(rs_line_src_width(), 15);
        assert_eq!(rs_line_src_height(), 15);
        assert_eq!(rs_line_dst_width(), 555);
        assert_eq!(rs_line_dst_height(), 15);
    }
}

// ---------- Pattern generation ----------

#[test]
fn dot_pattern_pixel_identical() {
    let rust_img = resamplescope::generate_dot_pattern();

    let mut c_buf = vec![0u8; 557 * 275];
    unsafe { rs_generate_dot_pattern(c_buf.as_mut_ptr()) };

    assert_eq!(
        rust_img.buf().len(),
        c_buf.len(),
        "buffer lengths differ: rust={} c={}",
        rust_img.buf().len(),
        c_buf.len()
    );

    let mut diffs = 0;
    let mut first_diff = None;
    for (i, (&r, &c)) in rust_img.buf().iter().zip(c_buf.iter()).enumerate() {
        if r != c {
            diffs += 1;
            if first_diff.is_none() {
                let x = i % 557;
                let y = i / 557;
                first_diff = Some((x, y, r, c));
            }
        }
    }

    if let Some((x, y, r, c)) = first_diff {
        panic!(
            "dot patterns differ at {diffs} pixels; first diff at ({x}, {y}): rust={r} c={c}"
        );
    }
}

#[test]
fn line_pattern_pixel_identical() {
    let rust_img = resamplescope::generate_line_pattern();

    let mut c_buf = vec![0u8; 15 * 15];
    unsafe { rs_generate_line_pattern(c_buf.as_mut_ptr()) };

    assert_eq!(rust_img.buf().len(), c_buf.len());

    for (i, (&r, &c)) in rust_img.buf().iter().zip(c_buf.iter()).enumerate() {
        assert_eq!(
            r, c,
            "line pattern pixel {i}: rust={r} c={c}"
        );
    }
}

// ---------- Analysis ----------

/// Nearest-neighbor resize for deterministic test input.
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
fn dot_analysis_matches_c() {
    // Generate and resize the dot pattern.
    let dot = resamplescope::generate_dot_pattern();
    let resized = nn_resize(dot.as_ref(), 555, 275);

    // Run Rust analysis.
    let config = resamplescope::AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let rust_result = resamplescope::analyze_downscale(&nn_resize, &config).unwrap();
    let rust_curve = rust_result.downscale_curve.as_ref().unwrap();

    // Run C analysis.
    let max_points = 555 * 25; // w * DOTIMG_NUMSTRIPS
    let mut c_offsets = vec![0.0f64; max_points];
    let mut c_weights = vec![0.0f64; max_points];
    let mut c_count: i32 = 0;

    let c_scale = unsafe {
        rs_analyze_dot(
            resized.buf().as_ptr(),
            555,
            275,
            0,
            c_offsets.as_mut_ptr(),
            c_weights.as_mut_ptr(),
            &mut c_count,
        )
    };

    // Compare scale factors.
    assert!(
        (rust_curve.scale_factor - c_scale).abs() < 1e-10,
        "scale factors differ: rust={} c={}",
        rust_curve.scale_factor,
        c_scale
    );

    // Compare point counts.
    let c_count = c_count as usize;
    assert_eq!(
        rust_curve.points.len(),
        c_count,
        "point counts differ: rust={} c={}",
        rust_curve.points.len(),
        c_count
    );

    // Compare each point.
    let mut max_offset_err = 0.0f64;
    let mut max_weight_err = 0.0f64;
    for i in 0..c_count {
        let (r_off, r_wt) = rust_curve.points[i];
        let c_off = c_offsets[i];
        let c_wt = c_weights[i];

        let off_err = (r_off - c_off).abs();
        let wt_err = (r_wt - c_wt).abs();
        max_offset_err = max_offset_err.max(off_err);
        max_weight_err = max_weight_err.max(wt_err);

        assert!(
            off_err < 1e-10,
            "dot point {i}: offset differs: rust={r_off} c={c_off} err={off_err}"
        );
        assert!(
            wt_err < 1e-10,
            "dot point {i}: weight differs: rust={r_wt} c={c_wt} err={wt_err}"
        );
    }

    println!(
        "dot analysis: {c_count} points, max offset err={max_offset_err:.2e}, max weight err={max_weight_err:.2e}"
    );
}

#[test]
fn line_analysis_matches_c() {
    // Generate and resize the line pattern.
    let line = resamplescope::generate_line_pattern();
    let resized = nn_resize(line.as_ref(), 555, 15);

    // Run Rust analysis.
    let config = resamplescope::AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let rust_result = resamplescope::analyze_upscale(&nn_resize, &config).unwrap();
    let rust_curve = rust_result.upscale_curve.as_ref().unwrap();

    // Run C analysis.
    let mut c_offsets = vec![0.0f64; 555];
    let mut c_weights = vec![0.0f64; 555];
    let mut c_area: f64 = 0.0;

    let c_scale = unsafe {
        rs_analyze_line(
            resized.buf().as_ptr(),
            555,
            15,
            0,
            c_offsets.as_mut_ptr(),
            c_weights.as_mut_ptr(),
            &mut c_area,
        )
    };

    // Compare scale factors.
    assert!(
        (rust_curve.scale_factor - c_scale).abs() < 1e-10,
        "scale factors differ: rust={} c={}",
        rust_curve.scale_factor,
        c_scale
    );

    // Compare areas.
    assert!(
        (rust_curve.area - c_area).abs() < 1e-10,
        "areas differ: rust={} c={}",
        rust_curve.area,
        c_area
    );

    // Compare point counts.
    assert_eq!(
        rust_curve.points.len(),
        555,
        "rust point count: {}",
        rust_curve.points.len()
    );

    // Compare each point.
    let mut max_offset_err = 0.0f64;
    let mut max_weight_err = 0.0f64;
    for i in 0..555 {
        let (r_off, r_wt) = rust_curve.points[i];
        let c_off = c_offsets[i];
        let c_wt = c_weights[i];

        let off_err = (r_off - c_off).abs();
        let wt_err = (r_wt - c_wt).abs();
        max_offset_err = max_offset_err.max(off_err);
        max_weight_err = max_weight_err.max(wt_err);

        assert!(
            off_err < 1e-10,
            "line point {i}: offset differs: rust={r_off} c={c_off} err={off_err}"
        );
        assert!(
            wt_err < 1e-10,
            "line point {i}: weight differs: rust={r_wt} c={c_wt} err={wt_err}"
        );
    }

    println!(
        "line analysis: 555 points, area rust={:.6} c={:.6}, max offset err={max_offset_err:.2e}, max weight err={max_weight_err:.2e}",
        rust_curve.area, c_area
    );
}

#[test]
fn line_analysis_with_srgb_matches_c() {
    // Use perfect_resize for a non-trivial sRGB test.
    let line = resamplescope::generate_line_pattern();
    let resized = resamplescope::perfect_resize(line.as_ref(), 555, 15, KnownFilter::Lanczos3);

    // Rust analysis with sRGB.
    let config = resamplescope::AnalysisConfig {
        srgb: true,
        detect_edges: false,
    };
    let resize_fn = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Lanczos3)
    };
    let rust_result = resamplescope::analyze_upscale(&resize_fn, &config).unwrap();
    let rust_curve = rust_result.upscale_curve.as_ref().unwrap();

    // C analysis with sRGB.
    let mut c_offsets = vec![0.0f64; 555];
    let mut c_weights = vec![0.0f64; 555];
    let mut c_area: f64 = 0.0;

    let c_scale = unsafe {
        rs_analyze_line(
            resized.buf().as_ptr(),
            555,
            15,
            1,
            c_offsets.as_mut_ptr(),
            c_weights.as_mut_ptr(),
            &mut c_area,
        )
    };

    assert!(
        (rust_curve.scale_factor - c_scale).abs() < 1e-10,
        "scale factors differ"
    );

    // sRGB conversion uses pow(), so allow slightly more tolerance for
    // floating-point differences between C and Rust math libraries.
    let mut max_offset_err = 0.0f64;
    let mut max_weight_err = 0.0f64;
    for i in 0..555 {
        let (r_off, r_wt) = rust_curve.points[i];
        let off_err = (r_off - c_offsets[i]).abs();
        let wt_err = (r_wt - c_weights[i]).abs();
        max_offset_err = max_offset_err.max(off_err);
        max_weight_err = max_weight_err.max(wt_err);
    }

    println!(
        "sRGB line analysis: max offset err={max_offset_err:.2e}, max weight err={max_weight_err:.2e}"
    );

    // pow() implementations may differ between C libm and Rust; allow small epsilon.
    assert!(
        max_offset_err < 1e-6,
        "sRGB offset error too large: {max_offset_err:.2e}"
    );
    assert!(
        max_weight_err < 1e-6,
        "sRGB weight error too large: {max_weight_err:.2e}"
    );
}

/// Test with perfect_resize (non-trivial pixel values) to stress both paths.
#[test]
fn dot_analysis_perfect_resize_matches_c() {
    let dot = resamplescope::generate_dot_pattern();
    let resized = resamplescope::perfect_resize(dot.as_ref(), 555, 275, KnownFilter::Lanczos3);

    // Rust analysis.
    let resize_fn = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Lanczos3)
    };
    let config = resamplescope::AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let rust_result = resamplescope::analyze_downscale(&resize_fn, &config).unwrap();
    let rust_curve = rust_result.downscale_curve.as_ref().unwrap();

    // C analysis.
    let max_points = 555 * 25;
    let mut c_offsets = vec![0.0f64; max_points];
    let mut c_weights = vec![0.0f64; max_points];
    let mut c_count: i32 = 0;

    unsafe {
        rs_analyze_dot(
            resized.buf().as_ptr(),
            555,
            275,
            0,
            c_offsets.as_mut_ptr(),
            c_weights.as_mut_ptr(),
            &mut c_count,
        );
    }

    let c_count = c_count as usize;
    assert_eq!(rust_curve.points.len(), c_count);

    let mut max_weight_err = 0.0f64;
    for i in 0..c_count {
        let (r_off, r_wt) = rust_curve.points[i];
        let off_err = (r_off - c_offsets[i]).abs();
        let wt_err = (r_wt - c_weights[i]).abs();
        max_weight_err = max_weight_err.max(wt_err);

        assert!(
            off_err < 1e-10,
            "Lanczos3 dot point {i}: offset err={off_err:.2e}"
        );
        assert!(
            wt_err < 1e-10,
            "Lanczos3 dot point {i}: weight err={wt_err:.2e}"
        );
    }

    println!(
        "Lanczos3 dot analysis: {c_count} points, max weight err={max_weight_err:.2e}"
    );
}
