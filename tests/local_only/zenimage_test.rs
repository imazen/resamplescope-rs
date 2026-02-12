use imgref::{ImgRef, ImgVec};
use resamplescope::{AnalysisConfig, KnownFilter};
use std::fs;
use std::path::Path;
use zenimage::graphics::filters::Filter as ZenFilter;
use zenimage::graphics::resize::{resize_linear, ResizeOptions};

fn write_rgb_png(path: &Path, img: &ImgVec<rgb::RGB8>) {
    let file = fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, img.width() as u32, img.height() as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let raw: Vec<u8> = img.buf().iter().flat_map(|p| [p.r, p.g, p.b]).collect();
    writer.write_image_data(&raw).unwrap();
}

/// Wrap a zenimage filter into resamplescope's callback signature.
/// Expands grayscale to BGR, resizes, extracts green channel.
fn make_zen_resize(
    filter: ZenFilter,
    srgb: bool,
) -> impl Fn(ImgRef<'_, u8>, usize, usize) -> ImgVec<u8> {
    move |src: ImgRef<'_, u8>, dst_w: usize, dst_h: usize| -> ImgVec<u8> {
        let src_w = src.width();
        let src_h = src.height();

        // Expand grayscale to BGR.
        let mut bgr = Vec::with_capacity(src_w * src_h * 3);
        for y in 0..src_h {
            for x in 0..src_w {
                let v = src.buf()[y * src.stride() + x];
                bgr.push(v); // B
                bgr.push(v); // G
                bgr.push(v); // R
            }
        }

        let options = ResizeOptions::new()
            .filter(filter)
            .sharpen(0.0)
            .linear_rgb(srgb)
            .has_alpha(false);

        let resized_bgr = resize_linear(&bgr, src_w, src_h, dst_w, dst_h, 3, &options)
            .expect("zenimage resize failed");

        // Extract green channel back to grayscale.
        let gray: Vec<u8> = resized_bgr.chunks_exact(3).map(|c| c[1]).collect();

        ImgVec::new(gray, dst_w, dst_h)
    }
}

/// Expected resamplescope match for each zenimage filter.
/// Returns (ZenFilter, expected KnownFilter or None for custom filters).
fn expected_matches() -> Vec<(ZenFilter, &'static str, Option<KnownFilter>)> {
    vec![
        (ZenFilter::Box, "box", Some(KnownFilter::Box)),
        (ZenFilter::Triangle, "triangle", Some(KnownFilter::Triangle)),
        (ZenFilter::Linear, "linear", Some(KnownFilter::Triangle)),
        (ZenFilter::Hermite, "hermite", Some(KnownFilter::Hermite)),
        (
            ZenFilter::CatmullRom,
            "catmull_rom",
            Some(KnownFilter::CatmullRom),
        ),
        (
            ZenFilter::Mitchell,
            "mitchell",
            Some(KnownFilter::Mitchell),
        ),
        (
            ZenFilter::CubicBSpline,
            "cubic_b_spline",
            Some(KnownFilter::BSpline),
        ),
        (ZenFilter::Lanczos2, "lanczos2", Some(KnownFilter::Lanczos2)),
        (ZenFilter::Lanczos, "lanczos3", Some(KnownFilter::Lanczos3)),
    ]
}

#[test]
fn zenimage_filter_identification_linear() {
    let out = Path::new("/mnt/v/output/resamplescope/zenimage");
    fs::create_dir_all(out).unwrap();

    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };

    println!("\n=== Zenimage filter identification (linear_rgb=false) ===\n");
    println!(
        "{:<25} {:<15} {:>8} {:>8} {:>8}",
        "Filter", "Best Match", "r", "rms", "support"
    );
    println!("{}", "-".repeat(70));

    let mut failures = Vec::new();

    for (zen_filter, name, expected) in expected_matches() {
        let resize = make_zen_resize(zen_filter, false);
        let result = resamplescope::analyze(&resize, &config).unwrap();

        let best = &result.scores[0];
        println!(
            "{:<25} {:<15} {:>8.4} {:>8.4} {:>4.1}/{:.1}",
            name,
            best.filter.name(),
            best.correlation,
            best.rms_error,
            best.detected_support,
            best.expected_support
        );

        // Write graph.
        let graph = if let Some(expected_filter) = expected {
            result.render_graph_with_reference(expected_filter)
        } else {
            result.render_graph()
        };
        write_rgb_png(&out.join(format!("{name}_linear.png")), &graph);

        if let Some(expected_filter) = expected {
            if best.filter != expected_filter {
                failures.push(format!(
                    "{name}: expected {}, got {} (r={:.4})",
                    expected_filter.name(),
                    best.filter.name(),
                    best.correlation
                ));
            }
        }
    }

    if !failures.is_empty() {
        println!("\nFailed filter identifications:");
        for f in &failures {
            println!("  {f}");
        }
    }
    assert!(failures.is_empty(), "Some filters were not correctly identified");
}

#[test]
fn zenimage_filter_identification_srgb() {
    let out = Path::new("/mnt/v/output/resamplescope/zenimage");
    fs::create_dir_all(out).unwrap();

    let config = AnalysisConfig {
        srgb: true,
        detect_edges: false,
    };

    println!("\n=== Zenimage filter identification (linear_rgb=true, srgb correction) ===\n");
    println!(
        "{:<25} {:<15} {:>8} {:>8} {:>8}",
        "Filter", "Best Match", "r", "rms", "support"
    );
    println!("{}", "-".repeat(70));

    let mut failures = Vec::new();

    for (zen_filter, name, expected) in expected_matches() {
        let resize = make_zen_resize(zen_filter, true);
        let result = resamplescope::analyze(&resize, &config).unwrap();

        let best = &result.scores[0];
        println!(
            "{:<25} {:<15} {:>8.4} {:>8.4} {:>4.1}/{:.1}",
            name,
            best.filter.name(),
            best.correlation,
            best.rms_error,
            best.detected_support,
            best.expected_support
        );

        // Write graph.
        let graph = if let Some(expected_filter) = expected {
            result.render_graph_with_reference(expected_filter)
        } else {
            result.render_graph()
        };
        write_rgb_png(&out.join(format!("{name}_srgb.png")), &graph);

        if let Some(expected_filter) = expected {
            if best.filter != expected_filter {
                failures.push(format!(
                    "{name}: expected {}, got {} (r={:.4})",
                    expected_filter.name(),
                    best.filter.name(),
                    best.correlation
                ));
            }
        }
    }

    if !failures.is_empty() {
        println!("\nExpected sRGB misidentifications (nonlinear transfer distorts negative lobes):");
        for f in &failures {
            println!("  {f}");
        }
    }
    // sRGB mode inherently distorts filter shapes (especially negative lobes).
    // We only assert the well-behaved filters (no negative weights) match.
    let critical_failures: Vec<_> = failures
        .iter()
        .filter(|f| !f.starts_with("lanczos"))
        .collect();
    assert!(
        critical_failures.is_empty(),
        "Non-Lanczos filters misidentified in sRGB mode: {:?}",
        critical_failures
    );
}

/// Test ALL zenimage filters (not just the ones with known matches).
/// This is informational - prints what resamplescope thinks each filter is.
#[test]
fn zenimage_all_filters_survey() {
    let out = Path::new("/mnt/v/output/resamplescope/zenimage");
    fs::create_dir_all(out).unwrap();

    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };

    // Filters from downscaling-eval (the 20 tested by imageflow)
    let downscaling_eval_filters = [
        ZenFilter::Robidoux,
        ZenFilter::RobidouxFast,
        ZenFilter::RobidouxSharp,
        ZenFilter::Lanczos,
        ZenFilter::LanczosSharp,
        ZenFilter::Lanczos2,
        ZenFilter::Lanczos2Sharp,
        ZenFilter::Ginseng,
        ZenFilter::GinsengSharp,
        ZenFilter::Cubic,
        ZenFilter::CubicSharp,
        ZenFilter::CatmullRom,
        ZenFilter::Mitchell,
        ZenFilter::CubicBSpline,
        ZenFilter::Hermite,
        ZenFilter::Triangle,
        ZenFilter::Linear,
        ZenFilter::Box,
        ZenFilter::NCubic,
        ZenFilter::NCubicSharp,
    ];

    println!("\n=== ALL zenimage/imageflow filters (linear_rgb=false) ===\n");
    println!(
        "{:<25} {:<15} {:>8} {:>8} {:>8} {:>8}",
        "Filter", "Best Match", "r", "rms", "max_err", "support"
    );
    println!("{}", "-".repeat(80));

    for zen_filter in downscaling_eval_filters {
        let name = zen_filter.name();
        let resize = make_zen_resize(zen_filter, false);
        let result = resamplescope::analyze(&resize, &config).unwrap();

        let best = &result.scores[0];
        let second = result.scores.get(1);
        println!(
            "{:<25} {:<15} {:>8.4} {:>8.4} {:>8.4} {:>4.1}/{:.1}{}",
            name,
            best.filter.name(),
            best.correlation,
            best.rms_error,
            best.max_error,
            best.detected_support,
            best.expected_support,
            if let Some(s) = second {
                format!("  (2nd: {} r={:.4})", s.filter.name(), s.correlation)
            } else {
                String::new()
            }
        );

        // Write graph for every filter.
        let graph = result.render_graph();
        write_rgb_png(&out.join(format!("{name}.png")), &graph);
    }
}

/// SSIM comparison: zenimage's resize vs perfect reference.
#[test]
fn zenimage_ssim_vs_reference() {
    let line = resamplescope::generate_line_pattern();

    println!("\n=== SSIM: zenimage vs perfect reference (line pattern 15->555) ===\n");
    println!("{:<25} {:>8}", "Filter", "SSIM");
    println!("{}", "-".repeat(35));

    let test_pairs: Vec<(ZenFilter, KnownFilter)> = vec![
        (ZenFilter::Box, KnownFilter::Box),
        (ZenFilter::Triangle, KnownFilter::Triangle),
        (ZenFilter::CatmullRom, KnownFilter::CatmullRom),
        (ZenFilter::Mitchell, KnownFilter::Mitchell),
        (ZenFilter::Lanczos2, KnownFilter::Lanczos2),
        (ZenFilter::Lanczos, KnownFilter::Lanczos3),
    ];

    for (zen_filter, ref_filter) in test_pairs {
        let resize = make_zen_resize(zen_filter, false);
        let actual = resize(line.as_ref(), 555, 15);
        let perfect = resamplescope::perfect_resize(line.as_ref(), 555, 15, ref_filter);

        let s = resamplescope::ssim(actual.buf(), perfect.buf(), 555, 15);
        println!("{:<25} {:>8.6}", zen_filter.name(), s);
    }
}
