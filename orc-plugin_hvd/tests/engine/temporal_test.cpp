// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests for temporal.h's warp/envelope/coherence primitives.

#include <algorithm>
#include <cmath>

#include "check.h"
#include "engine/temporal.h"

namespace {

hvd::Plane MakeTexture(int h, int w) {
  hvd::Plane p(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float fy = static_cast<float>(y);
      const float fx = static_cast<float>(x);
      p.at(y, x) = 40.0F * std::sin(fy * 0.31F) * std::cos(fx * 0.27F) +
                   15.0F * std::sin(fx * 0.9F + fy * 0.5F);
    }
  }
  return p;
}

hvd::ComplexPlane MakeCarrier(int h, int w) {
  hvd::ComplexPlane c(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // theta advances by pi per line (NTSC line-to-line burst flip) plus a
      // per-sample quadrature step, same convention as engine.cpp's
      // MakeCarrier — doesn't need to be identical, just a valid unit-
      // modulus carrier for the invariant under test.
      const float theta = static_cast<float>(y) * 3.14159265F +
                          static_cast<float>(x) * 1.5707963F;
      c.at(y, x) = hvd::Complex(std::cos(theta), std::sin(theta));
    }
  }
  return c;
}

}  // namespace

void RunTests() {
  constexpr int kH = 64, kW = 64;

  // --- EnvelopeOf: Re(chi_b * carrier) must equal Cb EXACTLY (up to float
  // rounding), since |carrier| == 1 by construction — this is a pure
  // algebraic identity of the envelope construction, not an approximation,
  // regardless of what q (the quadrature term) is. ------------------------
  {
    const hvd::Plane s = MakeTexture(kH, kW);
    const hvd::ComplexPlane carrier = MakeCarrier(kH, kW);
    const hvd::Envelope env = hvd::EnvelopeOf(s, carrier);
    CHECK(env.luma.height() == kH - 1);
    CHECK(env.chroma.height() == kH - 1);

    for (int y = 0; y < kH - 1; ++y) {
      for (int x = 0; x < kW; ++x) {
        const float cb = 0.5F * (s.at(y, x) - s.at(y + 1, x));
        const float reconstructed =
            (env.chroma.at(y, x) * carrier.at(y, x)).real();
        CHECK_NEAR(reconstructed, cb, 1e-3);
      }
    }
  }

  // --- WarpByTiles: with a uniform per-tile shift, the warped output at
  // an interior pixel must equal the source sampled at (y - dy, x - dx) —
  // this is the operator's own definition, so this checks the tile
  // interpolation/rounding/lookup machinery does what it claims to. -------
  {
    const hvd::Plane cur = MakeTexture(kH, kW);
    constexpr int kTile = 16;
    const int th = (kH + kTile - 1) / kTile;
    const int tw = (kW + kTile - 1) / kTile;
    constexpr int kShiftY = 2, kShiftX = -3;
    const hvd::Plane mdy(th, tw, static_cast<float>(kShiftY));
    const hvd::Plane mdx(th, tw, static_cast<float>(kShiftX));

    const hvd::Plane warped = hvd::WarpByTiles(cur, mdy, mdx, kTile);
    for (int y = 4; y < kH - 4; ++y) {
      for (int x = 4; x < kW - 4; ++x) {
        const float expected = cur.at(y - kShiftY, x - kShiftX);
        CHECK_NEAR(warped.at(y, x), expected, 1e-3);
      }
    }
  }

  // --- ComplexCoherence: identical fields must give coherence ~1 (up to
  // the box-blur radius's own numerical fuzz); a field decorrelated by a
  // random-ish phase rotation must give a visibly lower coherence. -------
  {
    hvd::ComplexPlane z1(kH, kW);
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        z1.at(y, x) = hvd::Complex(1.0F + 0.3F * std::sin(y * 0.2F),
                                   0.4F * std::cos(x * 0.3F));
      }
    }
    const hvd::Plane self_coherence = hvd::ComplexCoherence(z1, z1, 6);
    for (int y = 8; y < kH - 8; ++y) {
      for (int x = 8; x < kW - 8; ++x) {
        CHECK(self_coherence.at(y, x) > 0.99F);
      }
    }

    hvd::ComplexPlane z2(kH, kW);
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        // Rotate each pixel by an unrelated, rapidly-varying phase: same
        // magnitude as z1 (so a magnitude-only comparison would call this
        // "similar"), but decorrelated phase.
        const float theta = static_cast<float>((y * 37 + x * 53) % 17);
        z2.at(y, x) = z1.at(y, x) * hvd::Complex(std::cos(theta), std::sin(theta));
      }
    }
    const hvd::Plane cross_coherence = hvd::ComplexCoherence(z1, z2, 6);
    float mean = 0.0F;
    for (size_t i = 0; i < cross_coherence.size(); ++i) mean += cross_coherence[i];
    mean /= static_cast<float>(cross_coherence.size());
    CHECK(mean < 0.5F);
  }
  // --- McWarp: estimate motion from ref toward a shifted "cur", then warp
  // ref itself through that motion field — the result should land back on
  // cur at interior pixels (exercises EstimateMotion + WarpByTiles
  // together, the actual composition used by the reference). -------------
  {
    const hvd::Plane ref = MakeTexture(kH, kW);
    constexpr int kShiftY = 3, kShiftX = -2;
    hvd::Plane cur(kH, kW);
    for (int y = 0; y < kH; ++y) {
      const int sy = std::clamp(y - kShiftY, 0, kH - 1);
      for (int x = 0; x < kW; ++x) {
        const int sx = std::clamp(x - kShiftX, 0, kW - 1);
        cur.at(y, x) = ref.at(sy, sx);
      }
    }
    const hvd::McWarpResult mc = hvd::McWarp(ref, cur, {ref}, 32, 8);
    CHECK(mc.warped.size() == 1);
    for (int y = 8; y < kH - 8; ++y) {
      for (int x = 8; x < kW - 8; ++x) {
        CHECK_NEAR(mc.warped[0].at(y, x), cur.at(y, x), 2.0);
      }
    }
  }

  // --- MotionCompensatePrev, with an EXPLICIT motion field (isolates the
  // warp wiring from the estimator's own behaviour): result.composite must
  // equal WarpByTiles(prev.composite, ...) exactly, and result.carrier the
  // complex equivalent via WarpByTilesComplex. --------------------------
  {
    const hvd::Plane composite = MakeTexture(kH, kW);
    const hvd::ComplexPlane carrier = MakeCarrier(kH, kW);
    hvd::NeighborRawState prev;
    prev.luma = MakeTexture(kH, kW);  // arbitrary; unused since motion is given
    prev.composite = composite;
    prev.carrier = carrier;

    constexpr int kTile = 16;
    const int th = (kH + kTile - 1) / kTile;
    const int tw = (kW + kTile - 1) / kTile;
    hvd::MotionField forced;
    forced.tile = kTile;
    forced.dy = hvd::Plane(th, tw, 1.0F);
    forced.dx = hvd::Plane(th, tw, 2.0F);
    forced.confidence = hvd::Plane(th, tw, 0.8F);

    const hvd::MotionCompensatedResult r =
        hvd::MotionCompensatePrev(prev, prev.luma, kTile, 8, &forced);

    const hvd::Plane expected_composite =
        hvd::WarpByTiles(composite, forced.dy, forced.dx, kTile);
    const hvd::ComplexPlane expected_carrier =
        hvd::WarpByTilesComplex(carrier, forced.dy, forced.dx, kTile);
    for (int y = 0; y < kH; ++y) {
      for (int x = 0; x < kW; ++x) {
        CHECK_NEAR(r.composite.at(y, x), expected_composite.at(y, x), 1e-4);
        CHECK_NEAR(r.carrier.at(y, x).real(), expected_carrier.at(y, x).real(), 1e-4);
        CHECK_NEAR(r.carrier.at(y, x).imag(), expected_carrier.at(y, x).imag(), 1e-4);
      }
    }
    CHECK(r.confidence.height() == kH);
    CHECK(r.confidence.width() == kW);
  }

  // --- MotionCompensateEnvelope: with an explicit zero motion field and
  // same-parity neighbour, just check the plumbing produces finite,
  // correctly-shaped output — deriving a tight closed-form check here
  // would need re-deriving the half-line resampling algebra, which isn't
  // worth it for a sanity test at this stage. -----------------------------
  {
    const hvd::Plane composite = MakeTexture(kH, kW);
    const hvd::ComplexPlane carrier = MakeCarrier(kH, kW);
    hvd::NeighborRawState nb;
    nb.luma = MakeTexture(kH, kW);
    nb.composite = composite;
    nb.carrier = carrier;

    constexpr int kTile = 16;
    const int th = (kH + kTile - 1) / kTile;
    const int tw = (kW + kTile - 1) / kTile;
    hvd::MotionField zero_motion;
    zero_motion.tile = kTile;
    zero_motion.dy = hvd::Plane(th, tw, 0.0F);
    zero_motion.dx = hvd::Plane(th, tw, 0.0F);
    zero_motion.confidence = hvd::Plane(th, tw, 0.5F);

    const hvd::Plane y_cur = MakeTexture(kH, kW);
    const hvd::MotionCompensatedResult r = hvd::MotionCompensateEnvelope(
        nb, y_cur, carrier, /*parity_cur=*/0, /*parity_nb=*/0, kTile, 8,
        &zero_motion);

    CHECK(r.composite.height() == kH);
    CHECK(r.composite.width() == kW);
    for (size_t i = 0; i < r.composite.size(); ++i) {
      CHECK(std::isfinite(r.composite[i]));
      CHECK(std::isfinite(r.carrier[i].real()));
      CHECK(std::isfinite(r.carrier[i].imag()));
    }
  }
}

TEST_MAIN()
