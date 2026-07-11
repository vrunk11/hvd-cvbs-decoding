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

  // Configure how many threads FFTW itself should use internally for each
  // transform executed through this Fft2d instance. Defaults to
  // std::thread::hardware_concurrency() at construction (use every core for
  // the one frame usually being decoded — the preview/normal path).
  //
  // IMPORTANT for parallel export (hvd_chroma_decoder_stage.cpp): FFTW's
  // threading is a global PLANNER setting (fftwf_plan_with_nthreads),
  // applied at plan-CREATION time, not per-execution — so this only
  // affects plans created AFTER this call, and every Fft2d sharing the
  // process (i.e. every export worker thread's own Fft2d) affects the same
  // global FFTW planner state while planning. That's already serialised by
  // the same planning mutex that guards fftwf_plan_dft_2d (see fft2d.cpp),
  // so it's safe, but each export worker MUST call SetThreadCount(1) before
  // its first Forward()/Inverse() — otherwise every one of the N worker
  // threads would ALSO fan its own FFTs out across every core, the same
  // oversubscription problem LimitOpenMpThreadsPerWorker() exists to avoid,
  // just for FFTW instead of OpenMP.
  //
  // No-op (transforms always run on 1 thread, silently) if FFTW wasn't
  // built with threading support — see CMakeLists.txt's FFTW3F_THREADS
  // detection.
  void SetThreadCount(int n);

 private:
  ComplexPlane Run(const ComplexPlane& in, int sign, bool normalise);

  struct Impl;
  Impl* impl_;  // owns cached FFTW plans; freed in the destructor
};

}  // namespace hvd

#endif  // ORC_PLUGIN_HVD_ENGINE_FFT2D_H_
