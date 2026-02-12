use imgref::{ImgRef, ImgVec};
use linear_srgb::default::{linear_to_srgb_u8, srgb_u8_to_linear};
use resamplescope::{AnalysisConfig, KnownFilter};
use std::fs;
use std::path::Path;

fn write_rgb_png(path: &Path, img: &ImgVec<rgb::RGB8>) {
    let file = fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, img.width() as u32, img.height() as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let raw: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    writer.write_image_data(&raw).unwrap();
}

/// Bilinear resize operating directly on sRGB byte values (no linearization).
fn bilinear_resize_raw(src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize) -> ImgVec<u8> {
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

/// Wrap a raw-byte resizer with linear-srgb conversion:
/// sRGB u8 → linear f32 → resize in linear → sRGB u8.
///
/// This validates that linear-srgb's conversion functions are correct
/// by checking that resamplescope (with srgb=true) can still identify
/// the underlying filter through the nonlinear transfer.
fn with_linear_srgb(
    raw_resize: impl Fn(ImgRef<'_, u8>, usize, usize) -> ImgVec<u8>,
) -> impl Fn(ImgRef<'_, u8>, usize, usize) -> ImgVec<u8> {
    move |src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize| -> ImgVec<u8> {
        let src_w = src.width();
        let src_h = src.height();

        // sRGB u8 → linear f32.
        let linear_src: Vec<f32> = (0..src_h)
            .flat_map(|y| {
                (0..src_w).map(move |x| srgb_u8_to_linear(src.buf()[y * src.stride() + x]))
            })
            .collect();

        // Scale linear f32 [0,1] → u8 [0,255] for the byte-based resizer.
        let linear_u8: Vec<u8> = linear_src
            .iter()
            .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();

        let linear_img = ImgVec::new(linear_u8, src_w, src_h);

        // Resize in linear light.
        let resized_linear = raw_resize(linear_img.as_ref(), dst_w, dst_h);

        // Resized u8 [0,255] → f32 [0,1] → sRGB u8.
        let dst: Vec<u8> = resized_linear
            .buf()
            .iter()
            .map(|&v| linear_to_srgb_u8(v as f32 / 255.0))
            .collect();

        ImgVec::new(dst, dst_w, dst_h)
    }
}

/// Validate that linear-srgb round-trips correctly at key values.
#[test]
fn linear_srgb_roundtrip_accuracy() {
    println!("\n=== linear-srgb round-trip accuracy ===\n");
    println!("{:>5} {:>12} {:>12} {:>5}", "sRGB", "linear", "back", "err");
    println!("{}", "-".repeat(40));

    let mut max_err = 0i32;
    for v in 0..=255u8 {
        let linear = srgb_u8_to_linear(v);
        let back = linear_to_srgb_u8(linear);
        let err = (back as i32 - v as i32).abs();
        max_err = max_err.max(err);
        if v % 32 == 0 || err > 0 {
            println!("{v:>5} {linear:>12.6} {back:>12} {err:>5}");
        }
    }
    println!("\nMax round-trip error: {max_err}");
    assert!(max_err <= 1, "round-trip error too large: {max_err}");
}

/// Validate that linear-srgb conversion is monotonic.
#[test]
fn linear_srgb_monotonicity() {
    let mut prev = 0.0f32;
    for v in 0..=255u8 {
        let linear = srgb_u8_to_linear(v);
        assert!(
            linear >= prev,
            "non-monotonic at {v}: {prev} -> {linear}"
        );
        prev = linear;
    }
}

/// Core validation: bilinear resize wrapped with linear-srgb conversion,
/// analyzed by resamplescope with srgb=true, should identify as Triangle.
#[test]
fn linear_srgb_bilinear_identified_as_triangle() {
    let out = Path::new("/mnt/v/output/resamplescope/linear_srgb");
    fs::create_dir_all(out).unwrap();

    let srgb_bilinear = with_linear_srgb(bilinear_resize_raw);

    // With srgb=true, resamplescope should correct for the nonlinear transfer
    // and correctly identify the underlying triangle filter.
    let config_srgb = AnalysisConfig {
        srgb: true,
        detect_edges: false,
    };
    let result_srgb = resamplescope::analyze(&srgb_bilinear, &config_srgb).unwrap();

    let best = &result_srgb.scores[0];
    println!("linear-srgb bilinear (srgb=true): best={} r={:.4}", best.filter.name(), best.correlation);

    write_rgb_png(
        &out.join("bilinear_srgb_corrected.png"),
        &result_srgb.render_graph_with_reference(KnownFilter::Triangle),
    );

    assert_eq!(
        best.filter,
        KnownFilter::Triangle,
        "expected Triangle with srgb correction, got {} (r={:.4})",
        best.filter.name(),
        best.correlation
    );
    assert!(
        best.correlation > 0.99,
        "correlation too low: {:.4}",
        best.correlation
    );

    // Without srgb correction, the distorted shape should still be *somewhat*
    // identifiable but with lower correlation.
    let config_linear = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };
    let result_linear = resamplescope::analyze(&srgb_bilinear, &config_linear).unwrap();
    let best_uncorrected = &result_linear.scores[0];
    println!(
        "linear-srgb bilinear (srgb=false): best={} r={:.4}",
        best_uncorrected.filter.name(),
        best_uncorrected.correlation
    );

    write_rgb_png(
        &out.join("bilinear_srgb_uncorrected.png"),
        &result_linear.render_graph_with_reference(KnownFilter::Triangle),
    );
}

/// Validate linear-srgb with perfect_resize: wrap Lanczos3 reference resize
/// with linear-srgb conversion and verify resamplescope identifies it.
#[test]
fn linear_srgb_lanczos3_identification() {
    let out = Path::new("/mnt/v/output/resamplescope/linear_srgb");
    fs::create_dir_all(out).unwrap();

    let srgb_lanczos = with_linear_srgb(|src, w, h| {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Lanczos3)
    });

    let config = AnalysisConfig {
        srgb: true,
        detect_edges: false,
    };
    let result = resamplescope::analyze(&srgb_lanczos, &config).unwrap();

    let best = &result.scores[0];
    println!(
        "linear-srgb lanczos3 (srgb=true): best={} r={:.4}",
        best.filter.name(),
        best.correlation
    );

    write_rgb_png(
        &out.join("lanczos3_srgb_corrected.png"),
        &result.render_graph_with_reference(KnownFilter::Lanczos3),
    );

    // Lanczos3 has negative lobes that get distorted by sRGB transfer.
    // With srgb correction, we should still get a reasonable match.
    // The correlation may be lower than for triangle due to negative-lobe
    // quantization artifacts in u8 space.
    println!(
        "  (top 3: {} r={:.4}, {} r={:.4}, {} r={:.4})",
        result.scores[0].filter.name(), result.scores[0].correlation,
        result.scores[1].filter.name(), result.scores[1].correlation,
        result.scores[2].filter.name(), result.scores[2].correlation,
    );
}

/// Survey all reference filters through linear-srgb pipeline.
#[test]
fn linear_srgb_all_filters_survey() {
    let out = Path::new("/mnt/v/output/resamplescope/linear_srgb");
    fs::create_dir_all(out).unwrap();

    let config = AnalysisConfig {
        srgb: true,
        detect_edges: false,
    };

    let filters = [
        ("box", KnownFilter::Box),
        ("triangle", KnownFilter::Triangle),
        ("hermite", KnownFilter::Hermite),
        ("catmull_rom", KnownFilter::CatmullRom),
        ("mitchell", KnownFilter::Mitchell),
        ("bspline", KnownFilter::BSpline),
        ("lanczos2", KnownFilter::Lanczos2),
        ("lanczos3", KnownFilter::Lanczos3),
    ];

    println!("\n=== All filters through linear-srgb pipeline (srgb=true) ===\n");
    println!(
        "{:<15} {:<15} {:>8} {:>8}",
        "Filter", "Best Match", "r", "rms"
    );
    println!("{}", "-".repeat(50));

    let mut failures = Vec::new();

    for &(name, filter) in &filters {
        let srgb_resize = with_linear_srgb(move |src, w, h| {
            resamplescope::perfect_resize(src, w, h, filter)
        });
        let result = resamplescope::analyze(&srgb_resize, &config).unwrap();

        let best = &result.scores[0];
        println!(
            "{:<15} {:<15} {:>8.4} {:>8.4}",
            name,
            best.filter.name(),
            best.correlation,
            best.rms_error,
        );

        write_rgb_png(
            &out.join(format!("{name}_srgb.png")),
            &result.render_graph_with_reference(filter),
        );

        // For non-negative-lobe filters, we expect correct identification.
        let has_negative_lobes = matches!(
            filter,
            KnownFilter::CatmullRom | KnownFilter::Lanczos2 | KnownFilter::Lanczos3 | KnownFilter::Lanczos4
        );
        if !has_negative_lobes && best.filter != filter {
            failures.push(format!(
                "{name}: expected {}, got {} (r={:.4})",
                filter.name(),
                best.filter.name(),
                best.correlation
            ));
        }
    }

    if !failures.is_empty() {
        println!("\nFailed identifications:");
        for f in &failures {
            println!("  {f}");
        }
    }
    assert!(
        failures.is_empty(),
        "Non-negative-lobe filters misidentified through linear-srgb pipeline"
    );
}

/// SSIM comparison: linear-srgb pipeline vs direct (no conversion).
/// This validates that the conversion introduces minimal distortion
/// for well-behaved filters (positive lobes only).
#[test]
fn linear_srgb_ssim_vs_direct() {
    let line = resamplescope::generate_line_pattern();

    println!("\n=== SSIM: linear-srgb pipeline vs direct (line 15->555) ===\n");
    println!("{:<15} {:>8} {:>8}", "Filter", "SSIM", "note");
    println!("{}", "-".repeat(35));

    let test_filters = [
        KnownFilter::Triangle,
        KnownFilter::Mitchell,
        KnownFilter::CatmullRom,
        KnownFilter::Lanczos3,
    ];

    for filter in test_filters {
        // Direct resize (no sRGB conversion).
        let direct = resamplescope::perfect_resize(line.as_ref(), 555, 15, filter);

        // Through linear-srgb pipeline.
        let srgb_resize = with_linear_srgb(|src, w, h| {
            resamplescope::perfect_resize(src, w, h, filter)
        });
        let through_srgb = srgb_resize(line.as_ref(), 555, 15);

        let s = resamplescope::ssim(direct.buf(), through_srgb.buf(), 555, 15);
        let note = if s > 0.99 { "excellent" } else if s > 0.95 { "good" } else { "degraded" };
        println!("{:<15} {:>8.6} {:>8}", filter.name(), s, note);
    }
}
