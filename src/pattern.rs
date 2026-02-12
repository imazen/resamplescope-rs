use imgref::ImgVec;

// Dot pattern constants (matching C source exactly)
pub const DOT_SRC_WIDTH: usize = 557;
pub const DOT_HPIXELSPAN: usize = 25;
pub const DOT_NUM_STRIPS: usize = DOT_HPIXELSPAN;
pub const DOT_HCENTER: usize = (DOT_HPIXELSPAN - 1) / 2; // 12
pub const DOT_STRIP_HEIGHT: usize = 11;
pub const DOT_VCENTER: usize = (DOT_STRIP_HEIGHT - 1) / 2; // 5
pub const DOT_SRC_HEIGHT: usize = DOT_NUM_STRIPS * DOT_STRIP_HEIGHT; // 275

pub const DOT_DST_WIDTH: usize = DOT_SRC_WIDTH - 2; // 555
pub const DOT_DST_HEIGHT: usize = DOT_SRC_HEIGHT; // 275

// Line pattern constants
pub const LINE_SRC_WIDTH: usize = 15;
pub const LINE_SRC_HEIGHT: usize = 15;
pub const LINE_DST_WIDTH: usize = 555;
pub const LINE_DST_HEIGHT: usize = LINE_SRC_HEIGHT; // 15

pub const DARK: u8 = 50;
pub const BRIGHT: u8 = 250;

/// Generate the dot test pattern for downscale analysis.
/// 557x275 grayscale image with bright dots at phase-offset positions per strip.
pub fn generate_dot_pattern() -> ImgVec<u8> {
    let mut pixels = vec![DARK; DOT_SRC_WIDTH * DOT_SRC_HEIGHT];

    for j in 0..DOT_SRC_HEIGHT {
        let strip = j / DOT_STRIP_HEIGHT;
        let strip_row = j % DOT_STRIP_HEIGHT;

        if strip_row != DOT_VCENTER {
            continue;
        }

        for i in DOT_HCENTER..(DOT_SRC_WIDTH - DOT_HCENTER) {
            // Each strip shifts the dot positions by 1 pixel.
            // When i < strip, the C code produces a negative modulus which never
            // equals DOT_HCENTER, so no dot is placed. We replicate that by only
            // checking when i >= strip.
            if i >= strip && (i - strip) % DOT_HPIXELSPAN == DOT_HCENTER {
                pixels[j * DOT_SRC_WIDTH + i] = BRIGHT;
            }
        }
    }

    ImgVec::new(pixels, DOT_SRC_WIDTH, DOT_SRC_HEIGHT)
}

/// Generate the line test pattern for upscale analysis.
/// 15x15 grayscale image with a single bright column at the center (x=7).
pub fn generate_line_pattern() -> ImgVec<u8> {
    let middle = LINE_SRC_WIDTH / 2; // 7
    let mut pixels = vec![DARK; LINE_SRC_WIDTH * LINE_SRC_HEIGHT];

    for y in 0..LINE_SRC_HEIGHT {
        pixels[y * LINE_SRC_WIDTH + middle] = BRIGHT;
    }

    ImgVec::new(pixels, LINE_SRC_WIDTH, LINE_SRC_HEIGHT)
}

/// Generate the edge test pattern for edge handling detection.
/// 15x15 grayscale image with a bright column at x=1 (near left edge).
pub fn generate_edge_pattern() -> ImgVec<u8> {
    let mut pixels = vec![DARK; LINE_SRC_WIDTH * LINE_SRC_HEIGHT];

    for y in 0..LINE_SRC_HEIGHT {
        pixels[y * LINE_SRC_WIDTH + 1] = BRIGHT;
    }

    ImgVec::new(pixels, LINE_SRC_WIDTH, LINE_SRC_HEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_pattern_dimensions() {
        let img = generate_dot_pattern();
        assert_eq!(img.width(), DOT_SRC_WIDTH);
        assert_eq!(img.height(), DOT_SRC_HEIGHT);
    }

    #[test]
    fn dot_pattern_has_bright_pixels() {
        let img = generate_dot_pattern();
        let bright_count = img.buf().iter().filter(|&&v| v == BRIGHT).count();
        // Each strip has roughly (SRC_WIDTH - 2*HCENTER) / HPIXELSPAN dots
        // = (557 - 24) / 25 = 21.32 → 22 dots per strip × 25 strips
        assert!(bright_count > 500, "only {bright_count} bright pixels");
        assert!(
            bright_count < 600,
            "{bright_count} bright pixels (too many)"
        );
    }

    #[test]
    fn dot_pattern_center_row_of_strip0() {
        let img = generate_dot_pattern();
        // Strip 0, center row is y = DOT_VCENTER = 5
        let row = &img.buf()[DOT_VCENTER * DOT_SRC_WIDTH..][..DOT_SRC_WIDTH];
        // First dot at x = DOT_HCENTER = 12 (since strip=0, (12-0)%25==12)
        assert_eq!(row[DOT_HCENTER], BRIGHT);
        assert_eq!(row[DOT_HCENTER - 1], DARK);
        assert_eq!(row[DOT_HCENTER + 1], DARK);
        // Next dot at x = 37
        assert_eq!(row[DOT_HCENTER + DOT_HPIXELSPAN], BRIGHT);
    }

    #[test]
    fn line_pattern_dimensions() {
        let img = generate_line_pattern();
        assert_eq!(img.width(), LINE_SRC_WIDTH);
        assert_eq!(img.height(), LINE_SRC_HEIGHT);
    }

    #[test]
    fn line_pattern_center_column() {
        let img = generate_line_pattern();
        for y in 0..LINE_SRC_HEIGHT {
            for x in 0..LINE_SRC_WIDTH {
                let expected = if x == 7 { BRIGHT } else { DARK };
                assert_eq!(
                    img.buf()[y * LINE_SRC_WIDTH + x],
                    expected,
                    "mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn edge_pattern_column_at_x1() {
        let img = generate_edge_pattern();
        for y in 0..LINE_SRC_HEIGHT {
            assert_eq!(img.buf()[y * LINE_SRC_WIDTH + 0], DARK);
            assert_eq!(img.buf()[y * LINE_SRC_WIDTH + 1], BRIGHT);
            assert_eq!(img.buf()[y * LINE_SRC_WIDTH + 2], DARK);
        }
    }
}
