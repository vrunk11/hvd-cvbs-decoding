// SPDX-License-Identifier: GPL-3.0-or-later
//
// fft2d.h — thin 2-D complex FFT used by the holographic init.
//
// Backed by FFTW3 (single precision). FFTW is declared as a dependency of the
// plugin's own CMake target — the decode-orc SDK explicitly allows plugins to
// bring their own third-party libraries (FFmpeg, FFTW, ...); they are NOT
// inherited from the host. See docs/PORTING.md.
//
// The reference uses numpy's fft2/ifft2 with the standard normalisation
// (forward unnormalised, inverse divided by N). This wrapper matches that so
// the demodulate -> crop -> inverse pipeline in holographic_init is a faithful
// translation. Plans are cached per (height, width) and reused across frames.

#ifndef ORC_PLUGIN_HVD_ENGINE_FFT2D_H_
#define ORC_PLUGIN_HVD_ENGINE_FFT2D_H_

#include "engine/plane.h"

namespace hvd {

class Fft2d {
 public:
  Fft2d();
  ~Fft2d();

  Fft2d(const Fft2d&) = delete;
  Fft2d& operator=(const Fft2d&) = delete;

  // In-place semantics via return value. `Forward` is unnormalised;
  // `Inverse` divides by (height * width), matching numpy.
  ComplexPlane Forward(const ComplexPlane& in);
  ComplexPlane Inverse(const ComplexPlane& in);

 private:
  ComplexPlane Run(const ComplexPlane& in, int sign, bool normalise);

  struct Impl;
  Impl* impl_;  // owns cached FFTW plans; freed in the destructor
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_FFT2D_H_
