// SPDX-License-Identifier: GPL-3.0-or-later

#include "engine/lockin.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace hvd {

namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr float kHalfPi = 1.57079632679489661923F;

float MedianOf(std::vector<float> v) {
  if (v.empty()) return 0.0F;
  const size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Wrap an angle to (-pi, pi], matching numpy's angle(exp(i*x)).
float WrapPi(float a) {
  return std::atan2(std::sin(a), std::cos(a));
}

// Complex lock-in output per line over the burst window:
//   z[line] = mean_x[ (seg - mean(seg)) * exp(-i (pi/2) x) ]
// with x the ABSOLUTE sample index (so the local oscillator matches the carrier
// referenced to line start, exactly as MakeCarrier does). Returns z and |z|.
void BurstLockin(const Plane& field_ire, const FieldGeometry& g,
                 std::vector<Complex>* z_out, std::vector<float>* amp_out) {
  const int h = field_ire.height();
  const int x0 = g.colour_burst_start;
  const int x1 = g.colour_burst_end;
  const int n = std::max(0, x1 - x0);

  z_out->assign(h, Complex{0.0F, 0.0F});
  amp_out->assign(h, 0.0F);
  if (n == 0) return;

  for (int y = 0; y < h; ++y) {
    // DC pedestal removal: subtract the burst-window mean.
    float mean = 0.0F;
    for (int x = x0; x < x1; ++x) mean += field_ire.at(y, x);
    mean /= static_cast<float>(n);

    Complex acc{0.0F, 0.0F};
    for (int x = x0; x < x1; ++x) {
      const Complex ref = std::polar(1.0F, -kHalfPi * static_cast<float>(x));
      acc += (field_ire.at(y, x) - mean) * ref;
    }
    acc /= static_cast<float>(n);
    (*z_out)[y] = acc;
    (*amp_out)[y] = std::abs(acc);
  }
}
}  // namespace

std::vector<float> TridiagSmooth(const std::vector<float>& d,
                                 const std::vector<float>& a, float lam) {
  const int n = static_cast<int>(d.size());
  std::vector<float> x(n, 0.0F);
  if (n == 0) return x;
  if (n == 1) {
    x[0] = a[0] > 0.0F ? d[0] : 0.0F;
    return x;
  }

  // System matrix: diag(a) + lam * L, L the 1-D graph Laplacian (Neumann ends).
  std::vector<float> diag(n);
  for (int i = 0; i < n; ++i) diag[i] = a[i] + 2.0F * lam;
  diag[0] -= lam;
  diag[n - 1] -= lam;
  const float off = -lam;  // constant off-diagonal

  // rhs = a * d
  std::vector<float> rhs(n);
  for (int i = 0; i < n; ++i) rhs[i] = a[i] * d[i];

  // Thomas algorithm (forward elimination, back substitution).
  std::vector<float> c(n - 1);
  std::vector<float> dd(n);
  c[0] = off / diag[0];
  dd[0] = rhs[0] / diag[0];
  for (int i = 1; i < n; ++i) {
    const float m = diag[i] - off * c[i - 1];
    if (i < n - 1) c[i] = off / m;
    dd[i] = (rhs[i] - off * dd[i - 1]) / m;
  }
  x[n - 1] = dd[n - 1];
  for (int i = n - 2; i >= 0; --i) x[i] = dd[i] - c[i] * x[i + 1];
  return x;
}

std::vector<float> BurstLockinPhase(const Plane& field_ire,
                                    const FieldGeometry& g) {
  const int h = field_ire.height();
  std::vector<Complex> z;
  std::vector<float> amp;
  BurstLockin(field_ire, g, &z, &amp);

  // burst = A sin(phi) = Re[(-iA) e^{i phi}] => lock-in z = (-iA/2) e^{i theta}
  // => theta = angle(z) + pi/2.
  std::vector<float> theta(h, 0.0F);
  for (int y = 0; y < h; ++y) theta[y] = std::arg(z[y]) + kHalfPi;

  // "good" lines: burst amplitude above 20 % of the field maximum.
  const float amp_max = amp.empty() ? 0.0F : *std::max_element(amp.begin(),
                                                               amp.end());
  const float good_thresh = amp_max > 0.0F ? amp_max * 0.2F : 1.0F;
  std::vector<char> good(h, 0);
  int first_good = -1;
  for (int y = 0; y < h; ++y) {
    good[y] = amp[y] > good_thresh ? 1 : 0;
    if (good[y] && first_good < 0) first_good = y;
  }
  if (first_good < 0) return theta;  // no burst anywhere: keep raw angles

  // Model phase: pi per line, anchored at the first good line.
  std::vector<float> model(h);
  for (int y = 0; y < h; ++y) {
    model[y] = theta[first_good] + kPi * static_cast<float>(y - first_good);
  }

  // Deviation d = wrap(theta - model) about the model.
  std::vector<float> d(h);
  for (int y = 0; y < h; ++y) d[y] = WrapPi(theta[y] - model[y]);

  // Amplitude-derived weights, normalised by the median good amplitude and
  // clipped to [0, 2]; zero on lines with no usable burst.
  std::vector<float> good_amp;
  for (int y = 0; y < h; ++y)
    if (good[y]) good_amp.push_back(amp[y]);
  const float med_amp = MedianOf(good_amp) + 1e-9F;

  std::vector<float> a(h, 0.0F);
  for (int y = 0; y < h; ++y) {
    if (good[y]) a[y] = std::clamp(amp[y] / med_amp, 0.0F, 2.0F);
  }

  const float lam = 25.0F;
  std::vector<float> x = TridiagSmooth(d, a, lam);

  // One IRLS outlier rejection pass (Huber-like, scale ~0.15 rad).
  std::vector<float> a2(h, 0.0F);
  for (int y = 0; y < h; ++y) {
    const float r = std::fabs(d[y] - x[y]);
    a2[y] = a[y] * 0.15F / std::max(r, 0.15F);
  }
  x = TridiagSmooth(d, a2, lam);

  for (int y = 0; y < h; ++y) theta[y] = model[y] + x[y];
  return theta;
}

float BurstAmplitudeIre(const Plane& field_ire, const FieldGeometry& g) {
  std::vector<Complex> z;
  std::vector<float> amp;
  BurstLockin(field_ire, g, &z, &amp);
  // amp here is |z| = A/2 for burst A sin(phi); the reported amplitude is 2|z|.
  std::vector<float> pos;
  for (float v : amp)
    if (v > 0.0F) pos.push_back(2.0F * v);
  if (pos.empty()) return 20.0F;

  const float med_pos = MedianOf(pos);
  std::vector<float> good;
  for (float v : pos)
    if (v > 0.25F * med_pos) good.push_back(v);
  return good.empty() ? 20.0F : MedianOf(good);
}

}  // namespace hvd
