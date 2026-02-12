use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KnownFilter {
    Box,
    Triangle,
    Hermite,
    CatmullRom,
    Mitchell,
    BSpline,
    Lanczos2,
    Lanczos3,
    Lanczos4,
    MitchellNetravali { b: f64, c: f64 },
}

impl KnownFilter {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Box => "Box",
            Self::Triangle => "Triangle",
            Self::Hermite => "Hermite",
            Self::CatmullRom => "Catmull-Rom",
            Self::Mitchell => "Mitchell",
            Self::BSpline => "B-Spline",
            Self::Lanczos2 => "Lanczos2",
            Self::Lanczos3 => "Lanczos3",
            Self::Lanczos4 => "Lanczos4",
            Self::MitchellNetravali { .. } => "Mitchell-Netravali",
        }
    }

    pub fn support(&self) -> f64 {
        match self {
            Self::Box => 0.5,
            Self::Triangle | Self::Hermite => 1.0,
            Self::CatmullRom | Self::Mitchell | Self::BSpline | Self::MitchellNetravali { .. } => {
                2.0
            }
            Self::Lanczos2 => 2.0,
            Self::Lanczos3 => 3.0,
            Self::Lanczos4 => 4.0,
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Box => box_filter(x),
            Self::Triangle => triangle(x),
            Self::Hermite => hermite(x),
            Self::CatmullRom => mitchell_netravali(x, 0.0, 0.5),
            Self::Mitchell => mitchell_netravali(x, 1.0 / 3.0, 1.0 / 3.0),
            Self::BSpline => mitchell_netravali(x, 1.0, 0.0),
            Self::MitchellNetravali { b, c } => mitchell_netravali(x, *b, *c),
            Self::Lanczos2 => lanczos(x, 2),
            Self::Lanczos3 => lanczos(x, 3),
            Self::Lanczos4 => lanczos(x, 4),
        }
    }

    /// All built-in named filters for scoring.
    pub fn all_named() -> &'static [KnownFilter] {
        &[
            KnownFilter::Box,
            KnownFilter::Triangle,
            KnownFilter::Hermite,
            KnownFilter::CatmullRom,
            KnownFilter::Mitchell,
            KnownFilter::BSpline,
            KnownFilter::Lanczos2,
            KnownFilter::Lanczos3,
            KnownFilter::Lanczos4,
        ]
    }
}

impl std::fmt::Display for KnownFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MitchellNetravali { b, c } => write!(f, "Mitchell-Netravali(B={b:.3}, C={c:.3})"),
            other => f.write_str(other.name()),
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

fn box_filter(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 0.5 {
        1.0
    } else if (ax - 0.5).abs() < 1e-10 {
        0.5
    } else {
        0.0
    }
}

fn triangle(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0 { 1.0 - ax } else { 0.0 }
}

fn hermite(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0 {
        (2.0 * ax - 3.0) * ax * ax + 1.0
    } else {
        0.0
    }
}

fn mitchell_netravali(x: f64, b: f64, c: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax * ax * ax
            + (-18.0 + 12.0 * b + 6.0 * c) * ax * ax
            + (6.0 - 2.0 * b))
            / 6.0
    } else if ax < 2.0 {
        ((-b - 6.0 * c) * ax * ax * ax
            + (6.0 * b + 30.0 * c) * ax * ax
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

fn lanczos(x: f64, n: u32) -> f64 {
    let ax = x.abs();
    if ax < n as f64 {
        sinc(x) * sinc(x / n as f64)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_values_at_zero() {
        // Interpolating filters have f(0) = 1.
        for f in &[
            KnownFilter::Box,
            KnownFilter::Triangle,
            KnownFilter::Hermite,
            KnownFilter::CatmullRom,
            KnownFilter::Lanczos2,
            KnownFilter::Lanczos3,
            KnownFilter::Lanczos4,
        ] {
            let v = f.evaluate(0.0);
            assert!((v - 1.0).abs() < 1e-10, "{}: f(0) = {v}", f.name());
        }
        // Non-interpolating (approximating) filters have f(0) < 1.
        let m = KnownFilter::Mitchell.evaluate(0.0);
        assert!((m - 8.0 / 9.0).abs() < 1e-10, "Mitchell f(0) = {m}");
        let bs = KnownFilter::BSpline.evaluate(0.0);
        assert!((bs - 2.0 / 3.0).abs() < 1e-10, "B-Spline f(0) = {bs}");
    }

    #[test]
    fn filters_zero_outside_support() {
        for f in KnownFilter::all_named() {
            let s = f.support();
            assert!(
                f.evaluate(s + 0.5).abs() < 1e-10,
                "{}: f({}) = {}",
                f.name(),
                s + 0.5,
                f.evaluate(s + 0.5)
            );
        }
    }

    #[test]
    fn triangle_at_known_points() {
        assert!((KnownFilter::Triangle.evaluate(0.0) - 1.0).abs() < 1e-10);
        assert!((KnownFilter::Triangle.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((KnownFilter::Triangle.evaluate(1.0)).abs() < 1e-10);
    }

    #[test]
    fn catmull_rom_interpolating() {
        // CatmullRom passes through data points: f(0)=1, f(1)=0
        assert!((KnownFilter::CatmullRom.evaluate(0.0) - 1.0).abs() < 1e-10);
        assert!((KnownFilter::CatmullRom.evaluate(1.0)).abs() < 1e-10);
        assert!((KnownFilter::CatmullRom.evaluate(2.0)).abs() < 1e-10);
    }

    #[test]
    fn lanczos_symmetry() {
        for &x in &[0.3, 0.7, 1.5, 2.5] {
            let pos = KnownFilter::Lanczos3.evaluate(x);
            let neg = KnownFilter::Lanczos3.evaluate(-x);
            assert!(
                (pos - neg).abs() < 1e-10,
                "Lanczos3 not symmetric at {x}: {pos} vs {neg}"
            );
        }
    }

    #[test]
    fn bspline_partition_of_unity() {
        // B-spline should sum to 1 across integer shifts
        let x = 0.3;
        let sum: f64 = (-3..=3)
            .map(|k| KnownFilter::BSpline.evaluate(x - k as f64))
            .sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "B-spline partition of unity: sum = {sum}"
        );
    }
}
