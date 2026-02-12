use imgref::ImgVec;
use rgb::RGB8;

use crate::analyze::FilterCurve;
use crate::filters::KnownFilter;

const WIDTH: usize = 600;
const HEIGHT: usize = 300;
const ZERO_X: f64 = 230.0;
const UNIT_X: f64 = 90.0;
const ZERO_Y: f64 = 220.0;
const UNIT_Y: f64 = -200.0; // Negative: positive values go up

const WHITE: RGB8 = RGB8 {
    r: 255,
    g: 255,
    b: 255,
};
const BLACK: RGB8 = RGB8 { r: 0, g: 0, b: 0 };
const GRID_GRAY: RGB8 = RGB8 {
    r: 192,
    g: 192,
    b: 192,
};
const BORDER_GREEN: RGB8 = RGB8 {
    r: 144,
    g: 192,
    b: 144,
};
const SCATTER_BLUE: RGB8 = RGB8 { r: 0, g: 0, b: 255 };
const LINE_RED: RGB8 = RGB8 {
    r: 224,
    g: 64,
    b: 64,
};
const REF_LIGHT: RGB8 = RGB8 {
    r: 180,
    g: 180,
    b: 180,
};

fn xcoord(ix: f64) -> i32 {
    (0.5 + ZERO_X + ix * UNIT_X) as i32
}

fn ycoord(iy: f64) -> i32 {
    (0.5 + ZERO_Y + iy * UNIT_Y) as i32
}

fn set_pixel(buf: &mut [RGB8], x: i32, y: i32, color: RGB8) {
    if x >= 0 && x < WIDTH as i32 && y >= 0 && y < HEIGHT as i32 {
        buf[y as usize * WIDTH + x as usize] = color;
    }
}

fn draw_line(buf: &mut [RGB8], x0: i32, y0: i32, x1: i32, y1: i32, color: RGB8) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        set_pixel(buf, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

fn draw_dashed_line(buf: &mut [RGB8], x0: i32, y0: i32, x1: i32, y1: i32, color: RGB8) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let mut step = 0u32;

    loop {
        // 4 on, 4 off pattern matching gdImageDashedLine
        if step % 8 < 4 {
            set_pixel(buf, x, y, color);
        }
        step += 1;
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

fn draw_grid(buf: &mut [RGB8]) {
    // Dashed lines at half-integers
    for i in -10..=10 {
        let hx = xcoord(0.5 + i as f64);
        draw_dashed_line(buf, hx, 0, hx, HEIGHT as i32 - 1, GRID_GRAY);
        let hy = ycoord(0.5 + i as f64);
        draw_dashed_line(buf, 0, hy, WIDTH as i32 - 1, hy, GRID_GRAY);
    }

    // Solid lines at integers
    for i in -10..=10 {
        let ix = xcoord(i as f64);
        draw_line(buf, ix, 0, ix, HEIGHT as i32 - 1, GRID_GRAY);
        let iy = ycoord(i as f64);
        draw_line(buf, 0, iy, WIDTH as i32 - 1, iy, GRID_GRAY);
    }

    // Axes
    let ax = xcoord(0.0);
    draw_line(buf, ax, 0, ax, HEIGHT as i32 - 1, BLACK);
    let ay = ycoord(0.0);
    draw_line(buf, 0, ay, WIDTH as i32 - 1, ay, BLACK);
}

fn draw_border(buf: &mut [RGB8], color: RGB8) {
    let w = WIDTH as i32;
    let h = HEIGHT as i32;
    draw_line(buf, 0, 0, w - 1, 0, color);
    draw_line(buf, 0, h - 1, w - 1, h - 1, color);
    draw_line(buf, 0, 0, 0, h - 1, color);
    draw_line(buf, w - 1, 0, w - 1, h - 1, color);
}

fn plot_scatter(buf: &mut [RGB8], points: &[(f64, f64)], color: RGB8) {
    for &(x, y) in points {
        let px = xcoord(x);
        let py = ycoord(y);
        set_pixel(buf, px, py, color);
    }
}

fn plot_connected(buf: &mut [RGB8], points: &[(f64, f64)], color: RGB8) {
    let mut last: Option<(i32, i32)> = None;
    for &(x, y) in points {
        let px = xcoord(x);
        let py = ycoord(y);
        if let Some((lx, ly)) = last {
            draw_line(buf, lx, ly, px, py, color);
        }
        last = Some((px, py));
    }
}

fn plot_reference(buf: &mut [RGB8], filter: KnownFilter, color: RGB8) {
    // Sample the reference filter densely across the visible range.
    let x_min = -ZERO_X / UNIT_X; // leftmost visible logical x
    let x_max = (WIDTH as f64 - ZERO_X) / UNIT_X; // rightmost visible logical x

    let steps = WIDTH * 2;
    let mut last: Option<(i32, i32)> = None;
    for i in 0..=steps {
        let x = x_min + (x_max - x_min) * i as f64 / steps as f64;
        let y = filter.evaluate(x);
        let px = xcoord(x);
        let py = ycoord(y);
        if let Some((lx, ly)) = last {
            draw_line(buf, lx, ly, px, py, color);
        }
        last = Some((px, py));
    }
}

/// Render a scope graph showing the reconstructed filter curve(s).
pub fn render(
    downscale: Option<&FilterCurve>,
    upscale: Option<&FilterCurve>,
    reference: Option<KnownFilter>,
) -> ImgVec<RGB8> {
    let mut buf = vec![WHITE; WIDTH * HEIGHT];

    draw_grid(&mut buf);

    if let Some(filter) = reference {
        plot_reference(&mut buf, filter, REF_LIGHT);
    }

    if let Some(ds) = downscale {
        plot_scatter(&mut buf, &ds.points, SCATTER_BLUE);
    }

    if let Some(us) = upscale {
        plot_connected(&mut buf, &us.points, LINE_RED);
    }

    draw_border(&mut buf, BORDER_GREEN);

    ImgVec::new(buf, WIDTH, HEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_empty_graph() {
        let img = render(None, None, None);
        assert_eq!(img.width(), WIDTH);
        assert_eq!(img.height(), HEIGHT);
        // Check corners are border color
        assert_eq!(img.buf()[0], BORDER_GREEN);
    }

    #[test]
    fn coordinate_system() {
        assert_eq!(xcoord(0.0), 230);
        assert_eq!(xcoord(1.0), 320);
        assert_eq!(ycoord(0.0), 220);
        assert_eq!(ycoord(1.0), 20);
    }
}
