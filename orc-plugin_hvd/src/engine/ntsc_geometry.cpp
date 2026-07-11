// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/ntsc_geometry.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hvd {

namespace {
constexpr float kHalfPi = 1.57079632679489661923F;  // pi / 2

// 25th percentile of a copy of `v` (partial sort, O(n)). `v` is consumed.
float Percentile25(std::vector<float>* v) {
  if (v->empty()) return 0.0F;
  const size_t k = v->size() / 4;  // floor(0.25 * n), matches numpy 'lower'
  std::nth_element(v->begin(), v->begin() + k, v->end());
  return (*v)[k];
}

float Median(std::vector<float>* v) {
  if (v->empty()) return 0.0F;
  const size_t k = v->size() / 2;
  std::nth_element(v->begin(), v->begin() + k, v->end());
  return (*v)[k];
}
}  // namespace

ComplexPlane MakeCarrier(const std::vector<float>& theta,
                         const FieldGeometry& g) {
  const int h = static_cast<int>(theta.size());
  const int w = g.active_width();
  ComplexPlane carrier(h, w);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Absolute sample index of this active column, so the 90-deg/sample
      // carrier is referenced to the true line start (not the crop origin).
      const int sample = g.active_video_start + x;
      const float phase = theta[y] + kHalfPi * static_cast<float>(sample);
      carrier.at(y, x) = std::polar(1.0F, phase);
    }
  }
  return carrier;
}

float EstimateNoiseIre(const Plane& s) {
  const int h = s.height();
  const int w = s.width();
  if (w <= 8 || h <= 0) return 0.0F;

  // d[y, x] = S[x] - 2 S[x+4] + S[x+8]  over x in [0, w-8).
  std::vector<float> d;
  d.reserve(static_cast<size_t>(h) * (w - 8));
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x + 8 < w; ++x) {
      d.push_back(s.at(y, x) - 2.0F * s.at(y, x + 4) + s.at(y, x + 8));
    }
  }
  if (d.empty()) return 0.0F;

  std::vector<float> d_med = d;
  const float med = Median(&d_med);
  for (float& v : d) v = std::fabs(v - med);
  const float q = Percentile25(&d);

  // For Gaussian d, P25(|d|) = 0.3186 * sigma_d, and sigma_d^2 = 6 * s^2.
  return q / 0.3186F / std::sqrt(6.0F);
}

}  // namespace hvd
