// Extracted core algorithms from ResampleScope by Jason Summers.
// Original: http://entropymine.com/resamplescope/
// License: GPL-3.0-or-later
//
// This shim removes all gd/file/Windows dependencies and operates
// on raw pixel buffers, so it can be compiled with cc and linked
// into Rust tests for comparison against the Rust port.

#include <stdint.h>
#include <math.h>
#include <string.h>

// ---------- Constants (matching rscope.c exactly) ----------

#define DOTIMG_SRC_WIDTH    557
#define DOTIMG_HPIXELSPAN   25
#define DOTIMG_NUMSTRIPS    DOTIMG_HPIXELSPAN
#define DOTIMG_HCENTER      ((DOTIMG_HPIXELSPAN-1)/2)
#define DOTIMG_STRIPHEIGHT  11
#define DOTIMG_VCENTER      ((DOTIMG_STRIPHEIGHT-1)/2)
#define DOTIMG_SRC_HEIGHT   (DOTIMG_NUMSTRIPS*DOTIMG_STRIPHEIGHT)
#define DOTIMG_DST_WIDTH    (DOTIMG_SRC_WIDTH-2)
#define DOTIMG_DST_HEIGHT   DOTIMG_SRC_HEIGHT

#define LINEIMG_SRC_WIDTH   15
#define LINEIMG_SRC_HEIGHT  15
#define LINEIMG_DST_WIDTH   555
#define LINEIMG_DST_HEIGHT  LINEIMG_SRC_HEIGHT

#define DARK  50
#define BRIGHT 250

// ---------- Pattern generation ----------

// Generate the dot test pattern into a pre-allocated buffer.
// Buffer must be DOT_SRC_WIDTH * DOT_SRC_HEIGHT bytes.
// Pixel layout: row-major, grayscale u8.
void rs_generate_dot_pattern(uint8_t *buf) {
    int i, j;

    // Fill with dark gray.
    memset(buf, DARK, DOTIMG_SRC_WIDTH * DOTIMG_SRC_HEIGHT);

    for (j = 0; j < DOTIMG_SRC_HEIGHT; j++) {
        for (i = 0; i < DOTIMG_SRC_WIDTH; i++) {
            // This replicates the exact logic from rscope.c gen_dotimg_image():
            //   if ((j%DOTIMG_STRIPHEIGHT==DOTIMG_VCENTER) &&
            //       (i>=DOTIMG_HCENTER) &&
            //       (i<DOTIMG_SRC_WIDTH-DOTIMG_HCENTER)) {
            //       if ((i-j/DOTIMG_STRIPHEIGHT)%DOTIMG_HPIXELSPAN == DOTIMG_HCENTER)
            //           pixel = BRIGHT;
            //   }
            if ((j % DOTIMG_STRIPHEIGHT == DOTIMG_VCENTER) &&
                (i >= DOTIMG_HCENTER) &&
                (i < DOTIMG_SRC_WIDTH - DOTIMG_HCENTER)) {
                // C modulus of (i - j/DOTIMG_STRIPHEIGHT) can be negative when
                // i < j/DOTIMG_STRIPHEIGHT, but the comparison with DOTIMG_HCENTER
                // (which is positive) will never match for negative values, so
                // the dot won't be placed. This matches the C behavior exactly.
                int strip = j / DOTIMG_STRIPHEIGHT;
                int val = (i - strip) % DOTIMG_HPIXELSPAN;
                // C modulus for negative dividend: result has sign of dividend.
                // DOTIMG_HCENTER is 12, always positive, so negative val never matches.
                if (val == DOTIMG_HCENTER) {
                    buf[j * DOTIMG_SRC_WIDTH + i] = BRIGHT;
                }
            }
        }
    }
}

// Generate the line test pattern into a pre-allocated buffer.
// Buffer must be LINEIMG_SRC_WIDTH * LINEIMG_SRC_HEIGHT bytes.
void rs_generate_line_pattern(uint8_t *buf) {
    int j;
    int middle = LINEIMG_SRC_WIDTH / 2;

    memset(buf, DARK, LINEIMG_SRC_WIDTH * LINEIMG_SRC_HEIGHT);

    for (j = 0; j < LINEIMG_SRC_HEIGHT; j++) {
        buf[j * LINEIMG_SRC_WIDTH + middle] = BRIGHT;
    }
}

// ---------- sRGB conversion ----------

static double c_srgb_to_linear(double v_srgb) {
    if (v_srgb <= 0.04045) {
        return v_srgb / 12.92;
    } else {
        return pow((v_srgb + 0.055) / 1.055, 2.4);
    }
}

static double read_pixel(const uint8_t *buf, int stride, int x, int y, int srgb) {
    double raw = (double)buf[y * stride + x];
    if (srgb) {
        double srgb50_lin = c_srgb_to_linear(50.0 / 255.0);
        double srgb250_lin = c_srgb_to_linear(250.0 / 255.0);
        double v_lin = c_srgb_to_linear(raw / 255.0);
        return (v_lin - srgb50_lin) * ((250.0 - 50.0) / (srgb250_lin - srgb50_lin)) + 50.0;
    }
    return raw;
}

// ---------- Dot analysis (downscale) ----------

// Extract (offset, weight) scatter points from a resized dot pattern.
// This is a direct port of plot_strip() from rscope.c.
//
// Parameters:
//   resized   - the resized dot pattern image (grayscale, row-major)
//   w, h      - dimensions of resized image
//   srgb      - apply sRGB correction (0 or 1)
//   offsets   - output array for x-offsets (must hold w * DOTIMG_NUMSTRIPS entries)
//   weights   - output array for weights (must hold w * DOTIMG_NUMSTRIPS entries)
//   out_count - number of valid points written
//
// Returns scale_factor used.
double rs_analyze_dot(const uint8_t *resized, int w, int h,
                      int srgb,
                      double *offsets, double *weights, int *out_count) {
    double scale_factor = (double)w / (double)DOTIMG_SRC_WIDTH;
    int count = 0;
    int strip, dstpos, k, row;

    (void)h; // height must be DOTIMG_DST_HEIGHT, validated by caller

    for (strip = 0; strip < DOTIMG_NUMSTRIPS; strip++) {
        for (dstpos = 0; dstpos < w; dstpos++) {
            double offset = 10000.0;
            double tmp_offset;
            double tot, weight;
            double zp;

            // Find nearest zero-point (matching rscope.c plot_strip exactly).
            for (k = DOTIMG_HCENTER + strip;
                 k < DOTIMG_SRC_WIDTH - DOTIMG_HCENTER;
                 k += DOTIMG_HPIXELSPAN) {
                zp = scale_factor * (((double)k) + 0.5 - ((double)DOTIMG_SRC_WIDTH) / 2.0)
                     + ((double)w) / 2.0 - 0.5;
                tmp_offset = ((double)dstpos) - zp;
                if (fabs(tmp_offset) < fabs(offset)) {
                    offset = tmp_offset;
                }
            }

            // Skip points too far from any dot.
            if (fabs(offset) > scale_factor * DOTIMG_HCENTER) continue;

            // Sum vertically across the strip.
            tot = 0.0;
            for (row = 0; row < DOTIMG_STRIPHEIGHT; row++) {
                int y = DOTIMG_STRIPHEIGHT * strip + row;
                double v = read_pixel(resized, w, dstpos, y, srgb);
                tot += (v - 50.0);
            }

            // Normalize to weight.
            weight = tot / 200.0;

            if (scale_factor < 1.0) {
                weight /= scale_factor;
            } else {
                offset /= scale_factor;
            }

            offsets[count] = offset;
            weights[count] = weight;
            count++;
        }
    }

    *out_count = count;
    return scale_factor;
}

// ---------- Line analysis (upscale) ----------

// Extract (offset, weight) connected curve from a resized line pattern.
// This is a direct port of run_lineimg_1file() + gr_lineimg_graph_main() from rscope.c.
//
// Parameters:
//   resized   - the resized line pattern image (grayscale, row-major)
//   w, h      - dimensions of resized image
//   srgb      - apply sRGB correction (0 or 1)
//   offsets   - output array for x-offsets (must hold w entries)
//   weights   - output array for weights (must hold w entries)
//   out_area  - computed area (sum / scale_factor)
//
// Returns scale_factor used.
double rs_analyze_line(const uint8_t *resized, int w, int h,
                       int srgb,
                       double *offsets, double *weights, double *out_area) {
    double scale_factor = (double)w / (double)LINEIMG_SRC_WIDTH;
    int scanline = h / 2;
    double tot = 0.0;
    int i;

    // Read samples from cycling scanlines (matching rscope.c).
    // The C code reads: scanline + (i%3) - 1
    for (i = 0; i < w; i++) {
        int y;
        double v, weight, offset;

        if (h >= 3) {
            y = scanline + (i % 3) - 1;
            if (y < 0) y = 0;
            if (y >= h) y = h - 1;
        } else {
            y = scanline;
        }

        v = read_pixel(resized, w, i, y, srgb);
        weight = (v - 50.0) / 200.0;
        tot += weight;

        offset = 0.5 + (double)i - ((double)w) / 2.0;

        if (scale_factor < 1.0) {
            weight /= scale_factor;
        } else {
            offset /= scale_factor;
        }

        offsets[i] = offset;
        weights[i] = weight;
    }

    *out_area = tot / scale_factor;
    return scale_factor;
}

// ---------- Dimension queries ----------

int rs_dot_src_width(void)  { return DOTIMG_SRC_WIDTH; }
int rs_dot_src_height(void) { return DOTIMG_SRC_HEIGHT; }
int rs_dot_dst_width(void)  { return DOTIMG_DST_WIDTH; }
int rs_dot_dst_height(void) { return DOTIMG_DST_HEIGHT; }

int rs_line_src_width(void)  { return LINEIMG_SRC_WIDTH; }
int rs_line_src_height(void) { return LINEIMG_SRC_HEIGHT; }
int rs_line_dst_width(void)  { return LINEIMG_DST_WIDTH; }
int rs_line_dst_height(void) { return LINEIMG_DST_HEIGHT; }
