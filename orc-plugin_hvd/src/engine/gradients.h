// SPDX-License-Identifier: GPL-3.0-or-later
//
// gradients.h — forward-difference operators with Neumann boundaries and their
// EXACT adjoints. Ports `_dx`, `_dy`, `_dxT`, `_dyT` from the reference.
//
// Why Neumann (not periodic): a periodic roll couples opposite image borders
// through the priors — a subtle but real artefact source flagged by the
// reference's pre-port audit. The last column/row forward difference is defined
// as 0 instead of wrapping to column/row 0.
//
// Why the adjoints must be exact: the variational solver uses conjugate
// gradient, whose search direction is only a true gradient if the operator
// applied in the objective and the operator applied in its transpose form a
// genuine adjoint pair, i.e. <Dx a, b> == <a, DxT b> for all a, b. The
// `gradients_test.cpp` unit test checks exactly this identity. The templates
// work for both `float` (luma) and `Complex` (chroma) planes.

#ifndef ORC_PLUGIN_HVD_ENGINE_GRADIENTS_H_
#define ORC_PLUGIN_HVD_ENGINE_GRADIENTS_H_

#include "engine/plane.h"

namespace hvd {

// d[:, :-1] = a[:, 1:] - a[:, :-1];  d[:, -1] = 0
template <typename T>
BasicPlane<T> Dx(const BasicPlane<T>& a) {
  BasicPlane<T> d(a.height(), a.width());
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x + 1 < a.width(); ++x) {
      d.at(y, x) = a.at(y, x + 1) - a.at(y, x);
    }
    if (a.width() > 0) d.at(y, a.width() - 1) = T{};
  }
  return d;
}

// d[:-1] = a[1:] - a[:-1];  d[-1] = 0
template <typename T>
BasicPlane<T> Dy(const BasicPlane<T>& a) {
  BasicPlane<T> d(a.height(), a.width());
  for (int y = 0; y + 1 < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      d.at(y, x) = a.at(y + 1, x) - a.at(y, x);
    }
  }
  // Last row already zero-initialised.
  return d;
}

// Adjoint of Dx (negative divergence with matching boundary):
//   d[:, 0]    = -a[:, 0]
//   d[:, 1:-1] =  a[:, :-2] - a[:, 1:-1]
//   d[:, -1]   =  a[:, -2]
template <typename T>
BasicPlane<T> DxT(const BasicPlane<T>& a) {
  BasicPlane<T> d(a.height(), a.width());
  const int w = a.width();
  for (int y = 0; y < a.height(); ++y) {
    // For w < 2, Dx is the zero operator, so its adjoint is zero too
    // (d is already zero-initialised).
    if (w < 2) continue;
    d.at(y, 0) = -a.at(y, 0);
    for (int x = 1; x < w - 1; ++x) d.at(y, x) = a.at(y, x - 1) - a.at(y, x);
    d.at(y, w - 1) = a.at(y, w - 2);
  }
  return d;
}

// Adjoint of Dy.
template <typename T>
BasicPlane<T> DyT(const BasicPlane<T>& a) {
  BasicPlane<T> d(a.height(), a.width());
  const int h = a.height();
  const int w = a.width();
  // For h < 2, Dy is the zero operator, so its adjoint is zero too.
  if (h < 2) return d;
  for (int x = 0; x < w; ++x) {
    d.at(0, x) = -a.at(0, x);
    for (int y = 1; y < h - 1; ++y) d.at(y, x) = a.at(y - 1, x) - a.at(y, x);
    d.at(h - 1, x) = a.at(h - 2, x);
  }
  return d;
}

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_GRADIENTS_H_
