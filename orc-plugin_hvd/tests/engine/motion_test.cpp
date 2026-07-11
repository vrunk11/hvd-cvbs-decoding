// SPDX-License-Identifier: GPL-3.0-or-later
//
// Synthetic shift-recovery test for EstimateMotion(): build a textured
// image, shift it by a known integer (dy, dx), and check the tiled block
// matcher recovers that shift on interior tiles (border tiles can be less
// reliable since the "shifted-in" content there isn't real texture, just
// clamped edge extension).

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "check.h"
#include "engine/motion.h"

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

// cur(y, x) = ref(y - sy, x - sx), clamped at the border (matches the
// clamp-to-edge convention EstimateMotion's own internal padding uses, so
// border tiles aren't penalised for a boundary-handling mismatch that has
// nothing to do with the estimator itself).
hvd::Plane ShiftClamped(const hvd::Plane& ref, int sy, int sx) {
  const int h = ref.height();
  const int w = ref.width();
  hvd::Plane out(h, w);
  for (int y = 0; y < h; ++y) {
    const int sy_clamped = std::clamp(y - sy, 0, h - 1);
    for (int x = 0; x < w; ++x) {
      const int sx_clamped = std::clamp(x - sx, 0, w - 1);
      out.at(y, x) = ref.at(sy_clamped, sx_clamped);
    }
  }
  return out;
}

}  // namespace

void RunTests() {
  constexpr int kH = 128, kW = 128, kTile = 32;
  constexpr int kShiftY = 3, kShiftX = -2;

  const hvd::Plane ref = MakeTexture(kH, kW);
  const hvd::Plane cur = ShiftClamped(ref, kShiftY, kShiftX);

  const hvd::MotionField mf = hvd::EstimateMotion(ref, cur, kTile, /*search=*/8);

  const int th = mf.dy.height();
  const int tw = mf.dy.width();
  CHECK(th == (kH + kTile - 1) / kTile);
  CHECK(tw == (kW + kTile - 1) / kTile);

  // Interior tiles only (a 1-tile margin from every edge): with a 3x2-pixel
  // shift on a 32px tile and clamped borders, edge tiles can be slightly
  // biased by the clamp region, but interior ones see genuine shifted
  // texture throughout and should recover the shift tightly.
  bool any_checked = false;
  for (int ty = 1; ty < th - 1; ++ty) {
    for (int tx = 1; tx < tw - 1; ++tx) {
      any_checked = true;
      CHECK_NEAR(mf.dy.at(ty, tx), static_cast<double>(kShiftY), 0.6);
      CHECK_NEAR(mf.dx.at(ty, tx), static_cast<double>(kShiftX), 0.6);
      // A real, unambiguous shift on textured content should not be
      // discarded by the "prefer zero motion" margin rule.
      CHECK(mf.confidence.at(ty, tx) > 0.2F);
    }
  }
  CHECK(any_checked);  // sanity: the test grid is big enough to have any
}

TEST_MAIN()
