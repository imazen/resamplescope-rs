use imgref::{ImgRef, ImgVec};
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

fn write_gray_png(path: &Path, img: &ImgVec<u8>) {
    let file = fs::File::create(path).unwrap();
    let mut encoder = png::Encoder::new(file, img.width() as u32, img.height() as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(img.buf()).unwrap();
}

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
fn write_visual_output() {
    let out = Path::new("/mnt/v/output/resamplescope");
    fs::create_dir_all(out).unwrap();

    let config = AnalysisConfig {
        srgb: false,
        detect_edges: false,
    };

    // Write test patterns.
    write_gray_png(&out.join("dot_pattern.png"), &resamplescope::generate_dot_pattern());
    write_gray_png(
        &out.join("line_pattern.png"),
        &resamplescope::generate_line_pattern(),
    );

    // Bilinear (Triangle) analysis.
    let result = resamplescope::analyze(&bilinear_resize, &config).unwrap();
    write_rgb_png(&out.join("bilinear_graph.png"), &result.render_graph());
    write_rgb_png(
        &out.join("bilinear_graph_ref.png"),
        &result.render_graph_with_reference(KnownFilter::Triangle),
    );

    println!("Bilinear best match: {}", result.scores[0]);
    for s in result.scores.iter().take(3) {
        println!("  {s}");
    }

    // Lanczos3 via perfect_resize.
    let lanczos_resize = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Lanczos3)
    };
    let result = resamplescope::analyze(&lanczos_resize, &config).unwrap();
    write_rgb_png(&out.join("lanczos3_graph.png"), &result.render_graph());
    write_rgb_png(
        &out.join("lanczos3_graph_ref.png"),
        &result.render_graph_with_reference(KnownFilter::Lanczos3),
    );

    println!("Lanczos3 best match: {}", result.scores[0]);
    for s in result.scores.iter().take(3) {
        println!("  {s}");
    }

    // Mitchell via perfect_resize.
    let mitchell_resize = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::Mitchell)
    };
    let result = resamplescope::analyze(&mitchell_resize, &config).unwrap();
    write_rgb_png(&out.join("mitchell_graph.png"), &result.render_graph());
    write_rgb_png(
        &out.join("mitchell_graph_ref.png"),
        &result.render_graph_with_reference(KnownFilter::Mitchell),
    );

    println!("Mitchell best match: {}", result.scores[0]);

    // CatmullRom via perfect_resize.
    let catrom_resize = |src: ImgRef<'_, u8>, w: usize, h: usize| -> ImgVec<u8> {
        resamplescope::perfect_resize(src, w, h, KnownFilter::CatmullRom)
    };
    let result = resamplescope::analyze(&catrom_resize, &config).unwrap();
    write_rgb_png(&out.join("catmullrom_graph.png"), &result.render_graph());
    write_rgb_png(
        &out.join("catmullrom_graph_ref.png"),
        &result.render_graph_with_reference(KnownFilter::CatmullRom),
    );

    println!("CatmullRom best match: {}", result.scores[0]);
}
