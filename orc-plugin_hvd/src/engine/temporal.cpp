// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/temporal.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hvd {

namespace {

// Same box-blur convention as motion.cpp/holographic_init.cpp: zero-padded
// 'same' convolution normalised by the full kernel length (numpy.convolve
// equivalent). Duplicated locally rather than shared, matching how the rest
// of the engine already does this (see motion.cpp's own copy) — small
// enough that a shared header isn't worth the extra indirection yet.
void BoxBlur1DRows(Plane* a, int r) {
  const int h = a->height();
  const int w = a->width();
  const float norm = 1.0F / static_cast<float>(2 * r + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int x = 0; x < w; ++x) {
    std::vector<float> col(h);
    for (int y = 0; y < h; ++y) col[y] = a->at(y, x);
    for (int y = 0; y < h; ++y) {
      float acc = 0.0F;
      for (int k = -r; k <= r; ++k) {
        const int yy = y + k;
        if (yy >= 0 && yy < h) acc += col[yy];
      }
      a->at(y, x) = acc * norm;
    }
  }
}

void BoxBlur1DCols(Plane* a, int r) {
  const int h = a->height();
  const int w = a->width();
  const float norm = 1.0F / static_cast<float>(2 * r + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    std::vector<float> row(w);
    for (int x = 0; x < w; ++x) row[x] = a->at(y, x);
    for (int x = 0; x < w; ++x) {
      float acc = 0.0F;
      for (int k = -r; k <= r; ++k) {
        const int xx = x + k;
        if (xx >= 0 && xx < w) acc += row[xx];
      }
      a->at(y, x) = acc * norm;
    }
  }
}

Plane BoxBlur(Plane a, int r) {
  BoxBlur1DRows(&a, r);
  BoxBlur1DCols(&a, r);
  return a;
}

// 3x3 component-wise median, edge-padded. th/tw < 3 returns a copy of V
// unchanged (matches the reference: too small a tile grid for a 3x3
// neighbourhood to mean anything).
Plane Median3(const Plane& v) {
  const int th = v.height();
  const int tw = v.width();
  if (th < 3 || tw < 3) return v;
  Plane out(th, tw);
  for (int y = 0; y < th; ++y) {
    for (int x = 0; x < tw; ++x) {
      float vals[9];
      int k = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int yy = std::clamp(y + dy, 0, th - 1);
          const int xx = std::clamp(x + dx, 0, tw - 1);
          vals[k++] = v.at(yy, xx);
        }
      }
      std::nth_element(vals, vals + 4, vals + 9);
      out.at(y, x) = vals[4];
    }
  }
  return out;
}

}  // namespace

PerPixelMotion VectorsPerPixel(const Plane& mdy_in, const Plane& mdx_in,
                               int tile, int out_h, int out_w) {
  const int th = mdy_in.height();
  const int tw = mdy_in.width();

  // Outlier-snap: a vector disagreeing with its own 3x3 median by more than
  // 3px is replaced by that median (isolated flat-tile garbage squashed
  // before interpolation can spread it; coherent motion clusters pass
  // through untouched).
  const Plane my = Median3(mdy_in);
  const Plane mx = Median3(mdx_in);
  Plane mdy = mdy_in;
  Plane mdx = mdx_in;
  for (size_t i = 0; i < mdy.size(); ++i) {
    const float disagreement = std::fabs(mdy[i] - my[i]) + std::fabs(mdx[i] - mx[i]);
    if (disagreement > 3.0F) {
      mdy[i] = my[i];
      mdx[i] = mx[i];
    }
  }

  // Bracket pixel row/col y/x between tile centres cy[t] = (t + 0.5) * tile
  // (an evenly-spaced arithmetic sequence, so the bracketing index reduces
  // to a floor-and-clip instead of the reference's general searchsorted).
  auto bracket = [tile](int coord, int n_tiles, int* lo, float* frac) {
    if (n_tiles > 1) {
      const float rel = static_cast<float>(coord) / static_cast<float>(tile) - 0.5F;
      *lo = std::clamp(static_cast<int>(std::floor(rel)), 0, n_tiles - 2);
      const float centre_lo = (static_cast<float>(*lo) + 0.5F) * static_cast<float>(tile);
      *frac = std::clamp((static_cast<float>(coord) - centre_lo) / static_cast<float>(tile),
                         0.0F, 1.0F);
    } else {
      *lo = 0;
      *frac = 0.0F;
    }
  };

  auto interp = [&](const Plane& v) {
    Plane out(out_h, out_w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < out_h; ++y) {
      int iy;
      float fy;
      bracket(y, th, &iy, &fy);
      const int iy1 = std::min(iy + 1, th - 1);
      for (int x = 0; x < out_w; ++x) {
        int ix;
        float fx;
        bracket(x, tw, &ix, &fx);
        const int ix1 = std::min(ix + 1, tw - 1);
        const float v00 = v.at(iy, ix);
        const float v10 = v.at(iy1, ix);
        const float v01 = v.at(iy, ix1);
        const float v11 = v.at(iy1, ix1);
        out.at(y, x) = v00 * (1 - fy) * (1 - fx) + v10 * fy * (1 - fx) +
                      v01 * (1 - fy) * fx + v11 * fy * fx;
      }
    }
    return out;
  };

  PerPixelMotion out;
  out.dy = interp(mdy);
  out.dx = interp(mdx);
  return out;
}

Plane WarpByTiles(const Plane& a, const Plane& mdy, const Plane& mdx, int tile) {
  const int h = a.height();
  const int w = a.width();
  const PerPixelMotion v = VectorsPerPixel(mdy, mdx, tile, h, w);
  Plane out(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int sy = std::clamp(y - static_cast<int>(std::lround(v.dy.at(y, x))), 0, h - 1);
      const int sx = std::clamp(x - static_cast<int>(std::lround(v.dx.at(y, x))), 0, w - 1);
      out.at(y, x) = a.at(sy, sx);
    }
  }
  return out;
}

Plane WarpBilinearTiles(const Plane& a, const Plane& dyf, const Plane& dxf,
                        int tile, float row_offset, int out_h, int out_w) {
  const PerPixelMotion v = VectorsPerPixel(dyf, dxf, tile, out_h, out_w);
  const int hs = a.height();
  const int ws = a.width();
  Plane out(out_h, out_w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      float sy = static_cast<float>(y) - v.dy.at(y, x) + row_offset;
      float sx = static_cast<float>(x) - v.dx.at(y, x);
      // Clip leaves room for the +1 taps below without a second clamp
      // (matches the reference's "hs - 1.001" epsilon trick exactly).
      sy = std::clamp(sy, 0.0F, static_cast<float>(hs) - 1.001F);
      sx = std::clamp(sx, 0.0F, static_cast<float>(ws) - 1.001F);
      const int y0 = static_cast<int>(sy);
      const int x0 = static_cast<int>(sx);
      const float fy = sy - static_cast<float>(y0);
      const float fx = sx - static_cast<float>(x0);
      const int y1 = std::min(y0 + 1, hs - 1);
      const int x1 = std::min(x0 + 1, ws - 1);
      out.at(y, x) = a.at(y0, x0) * (1 - fy) * (1 - fx) +
                    a.at(y1, x0) * fy * (1 - fx) +
                    a.at(y0, x1) * (1 - fy) * fx + a.at(y1, x1) * fy * fx;
    }
  }
  return out;
}

Envelope EnvelopeOf(const Plane& s, const ComplexPlane& carrier) {
  const int h = s.height();
  const int w = s.width();
  const int ho = h - 1;

  Envelope e;
  e.luma = Plane(std::max(0, ho), w);
  e.chroma = ComplexPlane(std::max(0, ho), w);
  if (ho <= 0) return e;

  Plane cb(ho, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < ho; ++y) {
    for (int x = 0; x < w; ++x) {
      e.luma.at(y, x) = 0.5F * (s.at(y, x) + s.at(y + 1, x));
      cb.at(y, x) = 0.5F * (s.at(y, x) - s.at(y + 1, x));
    }
  }

  // Quadrature via a +/-1 sample x-shift (90 deg at 4fsc), WRAPPING around
  // each line (matches the reference's np.roll(..., axis=1) exactly — a
  // deliberate choice here, unlike the Neumann-boundary gradients used
  // elsewhere in the engine).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < ho; ++y) {
    for (int x = 0; x < w; ++x) {
      const int x_minus = (x - 1 + w) % w;
      const int x_plus = (x + 1) % w;
      const float q = 0.5F * (cb.at(y, x_minus) - cb.at(y, x_plus));
      const Complex cb_q(cb.at(y, x), q);
      e.chroma.at(y, x) = cb_q * std::conj(carrier.at(y, x));
    }
  }
  return e;
}

Plane ComplexCoherence(const ComplexPlane& z1, const ComplexPlane& z2, int r) {
  const int h = z1.height();
  const int w = z1.width();
  Plane cross_re(h, w), cross_im(h, w), mag1sq(h, w), mag2sq(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < static_cast<long>(z1.size()); ++i) {
    const Complex cross = z1[i] * std::conj(z2[i]);
    cross_re[i] = cross.real();
    cross_im[i] = cross.imag();
    mag1sq[i] = std::norm(z1[i]);
    mag2sq[i] = std::norm(z2[i]);
  }
  const Plane blurred_re = BoxBlur(cross_re, r);
  const Plane blurred_im = BoxBlur(cross_im, r);
  const Plane blurred_1 = BoxBlur(mag1sq, r);
  const Plane blurred_2 = BoxBlur(mag2sq, r);

  Plane out(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < static_cast<long>(out.size()); ++i) {
    const float num = std::sqrt(blurred_re[i] * blurred_re[i] +
                                blurred_im[i] * blurred_im[i]);
    const float den = std::sqrt(std::max(0.0F, blurred_1[i] * blurred_2[i])) + 1e-6F;
    out[i] = std::clamp(num / den, 0.0F, 1.0F);
  }
  return out;
}

ComplexPlane WarpByTilesComplex(const ComplexPlane& a, const Plane& mdy,
                                const Plane& mdx, int tile) {
  const int h = a.height();
  const int w = a.width();
  const PerPixelMotion v = VectorsPerPixel(mdy, mdx, tile, h, w);
  ComplexPlane out(h, w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int sy = std::clamp(y - static_cast<int>(std::lround(v.dy.at(y, x))), 0, h - 1);
      const int sx = std::clamp(x - static_cast<int>(std::lround(v.dx.at(y, x))), 0, w - 1);
      out.at(y, x) = a.at(sy, sx);
    }
  }
  return out;
}

Plane UpsampleConfidence(const Plane& conf, int tile, int out_h, int out_w) {
  const int th = conf.height();
  const int tw = conf.width();
  Plane px(out_h, out_w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < out_h; ++y) {
    const int ty = std::min(y / tile, th - 1);
    for (int x = 0; x < out_w; ++x) {
      const int tx = std::min(x / tile, tw - 1);
      const float c = conf.at(ty, tx);
      px.at(y, x) = c * c;  // squared: crush weak matches harder
    }
  }
  return BoxBlur(std::move(px), 8);
}

McWarpResult McWarp(const Plane& y_from, const Plane& y_to,
                    const std::vector<Plane>& arrays, int tile, int search) {
  const MotionField mf = EstimateMotion(y_from, y_to, tile, search);
  McWarpResult r;
  r.warped.reserve(arrays.size());
  for (const Plane& a : arrays) r.warped.push_back(WarpByTiles(a, mf.dy, mf.dx, tile));
  r.confidence = UpsampleConfidence(mf.confidence, tile, y_from.height(), y_from.width());
  return r;
}

MotionCompensatedResult MotionCompensateEnvelope(
    const NeighborRawState& neighbor, const Plane& y_cur,
    const ComplexPlane& carrier_cur, int parity_cur, int parity_nb, int tile,
    int search, const MotionField* motion) {
  MotionField local_motion;
  const MotionField* mf = motion;
  if (!mf) {
    local_motion = EstimateMotion(neighbor.luma, y_cur, tile, search);
    mf = &local_motion;
  }

  const Envelope env = EnvelopeOf(neighbor.composite, neighbor.carrier);
  const float row_off =
      (static_cast<float>(parity_cur) - static_cast<float>(parity_nb) - 1.0F) / 2.0F;
  const int out_h = y_cur.height();
  const int out_w = y_cur.width();

  const Plane yw = WarpBilinearTiles(env.luma, mf->dy, mf->dx, tile, row_off, out_h, out_w);

  // WarpBilinearTiles only takes real planes, so warp chi_b's real/imag
  // parts separately (matches the reference's Cw = warp(.real) + 1j *
  // warp(.imag) exactly).
  Plane env_chi_re(env.chroma.height(), env.chroma.width());
  Plane env_chi_im(env.chroma.height(), env.chroma.width());
  for (size_t i = 0; i < env.chroma.size(); ++i) {
    env_chi_re[i] = env.chroma[i].real();
    env_chi_im[i] = env.chroma[i].imag();
  }
  const Plane cw_re = WarpBilinearTiles(env_chi_re, mf->dy, mf->dx, tile, row_off, out_h, out_w);
  const Plane cw_im = WarpBilinearTiles(env_chi_im, mf->dy, mf->dx, tile, row_off, out_h, out_w);

  MotionCompensatedResult result;
  result.composite = Plane(out_h, out_w);
  result.carrier = ComplexPlane(out_h, out_w);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (long i = 0; i < static_cast<long>(result.composite.size()); ++i) {
    const Complex cw(cw_re[i], cw_im[i]);
    const Complex c_w = -carrier_cur[i];
    result.carrier[i] = c_w;
    result.composite[i] = yw[i] + (cw * c_w).real();
  }
  // Matches the reference: confidence upsampled at the NEIGHBOUR's shape
  // (Y_nb.shape), not the current frame's — usually identical in practice.
  result.confidence =
      UpsampleConfidence(mf->confidence, tile, neighbor.luma.height(), neighbor.luma.width());
  return result;
}

MotionCompensatedResult MotionCompensatePrev(
    const NeighborRawState& prev, const Plane& y_cur_init, int tile,
    int search, const MotionField* motion) {
  MotionField local_motion;
  const MotionField* mf = motion;
  if (!mf) {
    local_motion = EstimateMotion(prev.luma, y_cur_init, tile, search);
    mf = &local_motion;
  }
  MotionCompensatedResult result;
  result.composite = WarpByTiles(prev.composite, mf->dy, mf->dx, tile);
  result.carrier = WarpByTilesComplex(prev.carrier, mf->dy, mf->dx, tile);
  result.confidence =
      UpsampleConfidence(mf->confidence, tile, prev.luma.height(), prev.luma.width());
  return result;
}

}  // namespace hvd
