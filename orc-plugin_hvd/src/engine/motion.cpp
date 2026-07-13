// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/motion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace hvd {

namespace {

// Box-blur passes via prefix sums (integral images): O(n) independent of
// radius, per THEORY 9f. Semantics are IDENTICAL to the naive zero-padded
// 'same' convolution with CONSTANT 1/(2r+1) normalisation per axis —
// prefix-sum differences over clipped windows are exactly a zero-padded
// windowed sum, and the normalisation stays constant (edges attenuated,
// not renormalised). Double accumulators keep the running sums exact for
// float inputs (the reference used float64 cumsums for the same reason).
void BoxBlur1DRows(Plane* a, int r) {
  const int h = a->height();
  const int w = a->width();
  const float norm = 1.0F / static_cast<float>(2 * r + 1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int x = 0; x < w; ++x) {
    std::vector<double> c(static_cast<size_t>(h) + 1, 0.0);
    for (int y = 0; y < h; ++y) c[y + 1] = c[y] + a->at(y, x);
    for (int y = 0; y < h; ++y) {
      const int y0 = std::max(y - r, 0);
      const int y1 = std::min(y + r + 1, h);
      a->at(y, x) = static_cast<float>((c[y1] - c[y0]) * norm);
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
    std::vector<double> c(static_cast<size_t>(w) + 1, 0.0);
    for (int x = 0; x < w; ++x) c[x + 1] = c[x] + a->at(y, x);
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(x - r, 0);
      const int x1 = std::min(x + r + 1, w);
      a->at(y, x) = static_cast<float>((c[x1] - c[x0]) * norm);
    }
  }
}

// Non-overlapping f x f block average, cropping any remainder rows/cols —
// matches reference/hvd/decoder.py's _decimate.
Plane Decimate(const Plane& a, int f) {
  const int h = a.height();
  const int w = a.width();
  const int hh = h / f;
  const int ww = w / f;
  Plane out(hh, ww, 0.0F);
  const float norm = 1.0F / static_cast<float>(f * f);
  for (int y = 0; y < hh; ++y) {
    for (int x = 0; x < ww; ++x) {
      float acc = 0.0F;
      for (int dy = 0; dy < f; ++dy) {
        for (int dx = 0; dx < f; ++dx) {
          acc += a.at(y * f + dy, x * f + dx);
        }
      }
      out.at(y, x) = acc * norm;
    }
  }
  return out;
}

float Median(std::vector<float> v) {
  if (v.empty()) return 0.0F;
  const size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  float m = v[k];
  if (v.size() % 2 == 0) {
    // numpy.median averages the two middle elements for even-sized input.
    std::nth_element(v.begin(), v.begin() + k - 1, v.begin() + k);
    m = 0.5F * (m + v[k - 1]);
  }
  return m;
}

// Result of one block-matching pass — mirrors _bm_pass's (mdy, mdx, best,
// cost0) tuple. cost0 is only meaningful when has_cost0 is true (absolute
// mode with (0,0) among the candidates).
struct BmResult {
  Plane mdy, mdx, best, cost0;
  bool has_cost0 = false;
};

// One block-matching pass over candidate shifts (dys x dxs). If
// base_dy/base_dx are given, candidates are relative to them (per-tile) —
// pyramid/half-pel refinement; otherwise candidates are absolute shifts.
// A/B must be the same shape. `tile` is the block size in pixels.
BmResult BmPass(const Plane& A, const Plane& B, int tile,
                const std::vector<int>& dys, const std::vector<int>& dxs,
                const Plane* base_dy, const Plane* base_dx) {
  const int h = A.height();
  const int w = A.width();
  const int th = (h + tile - 1) / tile;
  const int tw = (w + tile - 1) / tile;

  int max_s = 0;
  for (int d : dys) max_s = std::max(max_s, std::abs(d));
  for (int d : dxs) max_s = std::max(max_s, std::abs(d));
  if (base_dy) {
    float base_max = 0.0F;
    for (size_t i = 0; i < base_dy->size(); ++i)
      base_max = std::max(base_max, std::fabs((*base_dy)[i]));
    for (size_t i = 0; i < base_dx->size(); ++i)
      base_max = std::max(base_max, std::fabs((*base_dx)[i]));
    max_s += static_cast<int>(base_max);
  }
  const int pad = max_s + 1;

  // Edge-padded reference (clamp-to-edge, matching numpy's pad mode="edge").
  Plane Bp(h + 2 * pad, w + 2 * pad);
  for (int y = 0; y < Bp.height(); ++y) {
    const int sy = std::clamp(y - pad, 0, h - 1);
    for (int x = 0; x < Bp.width(); ++x) {
      const int sx = std::clamp(x - pad, 0, w - 1);
      Bp.at(y, x) = B.at(sy, sx);
    }
  }

  BmResult r;
  r.mdy = Plane(th, tw, 0.0F);
  r.mdx = Plane(th, tw, 0.0F);
  r.best = Plane(th, tw, std::numeric_limits<float>::infinity());
  const bool want_cost0 = (base_dy == nullptr);
  if (want_cost0) r.cost0 = Plane(th, tw, 0.0F);

  Plane se(th, tw);

  for (int dy : dys) {
    for (int dx : dxs) {
      // Parallel over TILE ROWS (each ty owns rows [ty*tile, (ty+1)*tile) of
      // the image AND row ty of `se`, so there are no write conflicts) —
      // THEORY 9f: "tiles and fields parallelise trivially". This loop is
      // the whole cost of motion estimation (~130 SSD passes over the full
      // frame per pyramid) and used to run single-threaded.
      if (!base_dy) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ty = 0; ty < th; ++ty) {
          const int y_end = std::min((ty + 1) * tile, h);
          for (int tx = 0; tx < tw; ++tx) {
            const int x_end = std::min((tx + 1) * tile, w);
            float acc = 0.0F;
            for (int y = ty * tile; y < y_end; ++y) {
              const float* arow = &A.at(y, 0);
              const float* brow = &Bp.at(y + pad - dy, pad - dx);
              for (int x = tx * tile; x < x_end; ++x) {
                const float diff = arow[x] - brow[x];
                acc += diff * diff;
              }
            }
            se.at(ty, tx) = acc;
          }
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ty = 0; ty < th; ++ty) {
          const int y_end = std::min((ty + 1) * tile, h);
          for (int tx = 0; tx < tw; ++tx) {
            // The base vector is constant per TILE — hoist its rounding and
            // the source offsets out of the per-pixel loop (they used to be
            // recomputed 32x32 times per tile: two libm lrounds per pixel
            // dominated this, the hottest loop in motion estimation).
            const int sy = static_cast<int>(std::lround(base_dy->at(ty, tx))) + dy;
            const int sx = static_cast<int>(std::lround(base_dx->at(ty, tx))) + dx;
            const int x_end = std::min((tx + 1) * tile, w);
            float acc = 0.0F;
            for (int y = ty * tile; y < y_end; ++y) {
              const float* arow = &A.at(y, 0);
              const float* brow = &Bp.at(y + pad - sy, pad - sx);
              for (int x = tx * tile; x < x_end; ++x) {
                const float diff = arow[x] - brow[x];
                acc += diff * diff;
              }
            }
            se.at(ty, tx) = acc;
          }
        }
      }

      if (want_cost0 && dy == 0 && dx == 0) {
        for (size_t i = 0; i < se.size(); ++i) r.cost0[i] = se[i];
      }

      const float bias = 1.0F + 0.02F * static_cast<float>(std::abs(dy) + std::abs(dx));
      for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
          const float cost = se.at(ty, tx) * bias;
          if (cost < r.best.at(ty, tx)) {
            r.best.at(ty, tx) = cost;
            if (!base_dy) {
              r.mdy.at(ty, tx) = static_cast<float>(dy);
              r.mdx.at(ty, tx) = static_cast<float>(dx);
            } else {
              r.mdy.at(ty, tx) = base_dy->at(ty, tx) + static_cast<float>(dy);
              r.mdx.at(ty, tx) = base_dx->at(ty, tx) + static_cast<float>(dx);
            }
          }
        }
      }
    }
  }
  r.has_cost0 = want_cost0;
  return r;
}

std::vector<int> Range(int lo_inclusive, int hi_inclusive) {
  std::vector<int> v;
  v.reserve(static_cast<size_t>(hi_inclusive - lo_inclusive + 1));
  for (int i = lo_inclusive; i <= hi_inclusive; ++i) v.push_back(i);
  return v;
}

Plane AddScalar(const Plane& a, float s) {
  Plane out(a.height(), a.width());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + s;
  return out;
}

// Cost of one specific per-tile absolute shift (base + offset), i.e. a
// single-candidate relative-mode BmPass — used for the parabolic sub-pixel
// fit (_cost_at in the reference).
Plane CostAt(const Plane& A, const Plane& B, int tile, const Plane& mdy,
            const Plane& mdx, int dy_off, int dx_off) {
  const Plane by = AddScalar(mdy, static_cast<float>(dy_off));
  const Plane bx = AddScalar(mdx, static_cast<float>(dx_off));
  const BmResult r = BmPass(A, B, tile, {0}, {0}, &by, &bx);
  return r.best;
}

// Confidence layers shared by the full and the predicted+verified ME (port
// of decoder._motion_conf, factored out of estimate_motion when
// verify_motion landed): tile-energy explained, median-calibrated outlier
// rejection, global pair validity (scene-cut detector).
Plane MotionConfidence(const Plane& A, const Plane& best, int tile) {
  const int h = A.height();
  const int w = A.width();
  const int th = best.height();
  const int tw = best.width();

  Plane tmean(th, tw, 0.0F), tvar(th, tw, 0.0F);
  {
    Plane sum(th, tw, 0.0F), sum2(th, tw, 0.0F);
    for (int y = 0; y < h; ++y) {
      const int ty = std::min(y / tile, th - 1);
      for (int x = 0; x < w; ++x) {
        const int tx = std::min(x / tile, tw - 1);
        const float v = A.at(y, x);
        sum.at(ty, tx) += v;
        sum2.at(ty, tx) += v * v;
      }
    }
    const float norm = 1.0F / static_cast<float>(tile * tile);
    for (int i = 0; i < th * tw; ++i) {
      tmean[i] = sum[i] * norm;
      tvar[i] = sum2[i] * norm - tmean[i] * tmean[i];
    }
  }
  Plane conf(th, tw);
  for (int i = 0; i < th * tw; ++i) {
    const float energy = (tvar[i] + 1.0F) * static_cast<float>(tile * tile);
    conf[i] = std::clamp(1.0F - best[i] / energy, 0.0F, 1.0F);
  }

  std::vector<float> ratios(static_cast<size_t>(th) * tw);
  for (int i = 0; i < th * tw; ++i)
    ratios[i] = best[i] / static_cast<float>(tile * tile);
  const float med = Median(ratios);
  for (int i = 0; i < th * tw; ++i) {
    const float excess = std::max(0.0F, ratios[i] - med);
    conf[i] *= (med + 1.0F) / (med + 1.0F + excess);
    conf[i] *= 6.0F / (6.0F + med);
  }
  return conf;
}

}  // namespace

Plane BoxBlur(Plane a, int r) {
  BoxBlur1DRows(&a, r);
  BoxBlur1DCols(&a, r);
  return a;
}

MotionField EstimateMotion(const Plane& y_ref, const Plane& y_cur, int tile,
                            int search) {
  const Plane A = BoxBlur(y_cur, 2);
  const Plane B = BoxBlur(y_ref, 2);
  const int h = A.height();
  const int w = A.width();
  const int th = (h + tile - 1) / tile;
  const int tw = (w + tile - 1) / tile;

  // --- Level 0: 4x-decimated exhaustive coarse search ---------------------
  constexpr int kDecimation = 4;
  const int cs = std::max(1, (search + kDecimation - 1) / kDecimation);
  const Plane Ad = Decimate(A, kDecimation);
  const Plane Bd = Decimate(B, kDecimation);
  const int coarse_tile = std::max(4, tile / kDecimation);
  const BmResult coarse =
      BmPass(Ad, Bd, coarse_tile, Range(-cs, cs), Range(-cs, cs), nullptr, nullptr);

  // Upscale the coarse tile-grid vectors onto the full-res tile grid
  // (nearest-neighbour), then scale pixel units back up by kDecimation.
  const int hh = coarse.mdy.height();
  const int ww = coarse.mdy.width();
  Plane base_dy(th, tw), base_dx(th, tw);
  for (int ty = 0; ty < th; ++ty) {
    const int gy = std::min((ty * hh) / th, hh - 1);
    for (int tx = 0; tx < tw; ++tx) {
      const int gx = std::min((tx * ww) / tw, ww - 1);
      base_dy.at(ty, tx) = coarse.mdy.at(gy, gx) * static_cast<float>(kDecimation);
      base_dx.at(ty, tx) = coarse.mdx.at(gy, gx) * static_cast<float>(kDecimation);
    }
  }

  // --- Level 1: full-res +/-3 px refinement around the coarse vector ------
  const BmResult refined =
      BmPass(A, B, tile, Range(-3, 3), Range(-3, 3), &base_dy, &base_dx);
  Plane mdy = refined.mdy;
  Plane mdx = refined.mdx;
  const Plane& best_after_refine = refined.best;

  // --- Half-pel parabolic sub-pixel fit around the refined integer vector -
  const Plane cy_m = CostAt(A, B, tile, mdy, mdx, -1, 0);
  const Plane cy_p = CostAt(A, B, tile, mdy, mdx, 1, 0);
  const Plane cx_m = CostAt(A, B, tile, mdy, mdx, 0, -1);
  const Plane cx_p = CostAt(A, B, tile, mdy, mdx, 0, 1);

  auto parabolic = [](float cm, float c0, float cp) {
    const float den = cm - 2.0F * c0 + cp;
    float sub = 0.0F;
    if (den > 1e-9F) sub = 0.5F * (cm - cp) / std::max(den, 1e-9F);
    return std::clamp(sub, -0.5F, 0.5F);
  };
  Plane sdy(th, tw), sdx(th, tw);
  for (int i = 0; i < th * tw; ++i) {
    sdy[i] = parabolic(cy_m[i], best_after_refine[i], cy_p[i]);
    sdx[i] = parabolic(cx_m[i], best_after_refine[i], cx_p[i]);
  }

  // --- Explicit zero-motion evaluation + "prefer zero" margin rule --------
  const BmResult zero = BmPass(A, B, tile, {0}, {0}, nullptr, nullptr);
  const Plane& best0 = zero.best;
  const Plane& cost0 = zero.cost0;

  Plane best = best_after_refine;  // mutable copy: the margin rule updates it
  for (int i = 0; i < th * tw; ++i) {
    if (best0[i] < best[i]) {
      mdy[i] = 0.0F;
      mdx[i] = 0.0F;
      best[i] = best0[i];
    }
    if (best[i] > 0.85F * cost0[i]) {
      mdy[i] = 0.0F;
      mdx[i] = 0.0F;
      best[i] = cost0[i];
      sdy[i] = 0.0F;  // sub-pixel is meaningless on a margin-forced vector
      sdx[i] = 0.0F;
    }
  }

  // --- Per-tile confidence: local contrast energy + outlier calibration ---
  Plane conf = MotionConfidence(A, best, tile);

  MotionField out;
  out.tile = tile;
  out.dy = Plane(th, tw);
  out.dx = Plane(th, tw);
  for (int i = 0; i < th * tw; ++i) {
    out.dy[i] = mdy[i] + sdy[i];
    out.dx[i] = mdx[i] + sdx[i];
  }
  out.confidence = std::move(conf);
  return out;
}

MotionField VerifyMotion(const Plane& y_ref, const Plane& y_cur,
                         const Plane& pdy, const Plane& pdx, int tile) {
  const Plane A = BoxBlur(y_cur, 2);
  const Plane B = BoxBlur(y_ref, 2);

  // Rounded integer prediction as the (single) relative-mode candidate.
  Plane pdyi(pdy.height(), pdy.width());
  Plane pdxi(pdx.height(), pdx.width());
  for (size_t i = 0; i < pdyi.size(); ++i) {
    pdyi[i] = static_cast<float>(std::lround(pdy[i]));
    pdxi[i] = static_cast<float>(std::lround(pdx[i]));
  }

  // Two SSD evaluations per tile: the predicted vector and the zero vector.
  const BmResult pred = BmPass(A, B, tile, {0}, {0}, &pdyi, &pdxi);
  const BmResult zero = BmPass(A, B, tile, {0}, {0}, nullptr, nullptr);

  const int th = pred.best.height();
  const int tw = pred.best.width();
  MotionField out;
  out.tile = tile;
  out.dy = Plane(th, tw);
  out.dx = Plane(th, tw);
  Plane best(th, tw);
  for (int i = 0; i < th * tw; ++i) {
    if (zero.best[i] <= pred.best[i]) {
      out.dy[i] = 0.0F;
      out.dx[i] = 0.0F;
      best[i] = zero.best[i];
    } else {
      // Keep the FULL float prediction where it won — its sub-pixel part
      // came from the trajectory fit and is better information than the
      // rounded value the SSD was evaluated at.
      out.dy[i] = pdy[i];
      out.dx[i] = pdx[i];
      best[i] = pred.best[i];
    }
  }
  out.confidence = MotionConfidence(A, best, tile);
  return out;
}

void FitTrajectory(const std::vector<OffsetMotion>& measurements,
                   Plane* vy, Plane* vx) {
  if (measurements.empty()) return;
  const int th = measurements.front().field.dy.height();
  const int tw = measurements.front().field.dy.width();
  *vy = Plane(th, tw, 0.0F);
  *vx = Plane(th, tw, 0.0F);

  // Per-tile masked median over <= 6 samples (numpy.nanmedian semantics:
  // average of the two middles for even counts; all-masked tiles fall back
  // to 0, matching the reference's nan_to_num). The plain median over so
  // few samples is robust enough — a weighted median would be heavier for
  // no measured benefit (reference's own comment).
  std::vector<float> sy, sx;
  sy.reserve(measurements.size());
  sx.reserve(measurements.size());
  for (int i = 0; i < th * tw; ++i) {
    sy.clear();
    sx.clear();
    for (const OffsetMotion& m : measurements) {
      if (m.offset == 0) continue;
      if (m.field.confidence[i] <= 0.15F) continue;
      const float o = static_cast<float>(m.offset);
      sy.push_back((m.field.dy[i] - m.parity_shift) / o);
      sx.push_back(m.field.dx[i] / o);
    }
    if (sy.empty()) continue;
    (*vy)[i] = Median(sy);
    (*vx)[i] = Median(sx);
  }
}

void TrajectorySnap(std::vector<OffsetMotion>* measurements) {
  if (!measurements || measurements->size() < 2) return;
  Plane vym, vxm;
  FitTrajectory(*measurements, &vym, &vxm);
  const int th = vym.height();
  const int tw = vym.width();

  // Consensus requirement: the snap only applies where >= 3 offsets agree
  // with the fitted trajectory within 1.5 px — structured motion qualifies,
  // noise-dominated matching leaves the pairwise vectors untouched.
  constexpr float kAgreeTol = 1.5F;
  Plane agree(th, tw, 0.0F);
  for (const OffsetMotion& m : *measurements) {
    const float o = static_cast<float>(m.offset);
    for (int i = 0; i < th * tw; ++i) {
      const float py = o * vym[i] + m.parity_shift;
      const float px = o * vxm[i];
      const float dev =
          std::fabs(m.field.dy[i] - py) + std::fabs(m.field.dx[i] - px);
      if (dev <= kAgreeTol) agree[i] += 1.0F;
    }
  }

  for (OffsetMotion& m : *measurements) {
    const float o = static_cast<float>(m.offset);
    for (int i = 0; i < th * tw; ++i) {
      if (agree[i] < 3.0F) continue;
      const float py = o * vym[i] + m.parity_shift;
      const float px = o * vxm[i];
      const float dev =
          std::fabs(m.field.dy[i] - py) + std::fabs(m.field.dx[i] - px);
      if (dev <= kAgreeTol) {
        m.field.dy[i] = py;
        m.field.dx[i] = px;
      }
    }
  }
}

}  // namespace hvd
