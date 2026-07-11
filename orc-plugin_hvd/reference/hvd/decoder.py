"""
hvd.decoder — Holographic-Variational NTSC decoder core.

The idea, in one paragraph
--------------------------
A time-base-corrected NTSC field is, mathematically, an *off-axis
hologram*: S(x,y) = Y(x,y) + Re[ chi(x,y) * exp(i*phi(x,y)) ], where Y
is the low-frequency "object" term, chi = V - iU is a complex field
(the chrominance phasor), and phi is a spatial carrier whose fringes
run diagonally across the frame (90 deg/sample horizontally, 180
deg/line vertically — the 227.5 cycles/line geometry of NTSC). Dot
crawl and rainbowing are precisely the *twin image* and *zero order
leakage* artefacts of holography. So instead of comb-filtering, we:

  1. recover the carrier phase per line with a lock-in amplifier on
     the colour burst (physics instrumentation);
  2. reconstruct chi with the standard digital-holography move —
     demodulate by exp(-i*phi), low-pass in 2D (this alone already
     equals/beats a 2D comb);
  3. refine (Y, chi) *jointly* as a regularized inverse problem:

        argmin  || S - Y - Re[chi e^{i phi}] ||^2
        Y, chi     + lY * ||grad Y||^2  +  lC * ||grad chi||^2

     solved by conjugate gradient. No hard Y/C split ever happens:
     the separation *emerges* from the optimisation, and the classic
     artefact trade-offs become two scalar knobs (lY, lC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

from .tbc import TbcSource, VideoParameters

# ----------------------------------------------------------------- colour

YUV_TO_RGB = np.array(
    [[1.0, 0.0, 1.13983],
     [1.0, -0.39465, -0.58060],
     [1.0, 2.03211, 0.0]]
)

IRE_BLACK = 7.5
IRE_WHITE = 100.0


@dataclass
class DecoderConfig:
    psi_init: bool = False         # phase-shifting-interferometry init:
                                   # closed-form per-pixel weighted LS over
                                   # the >=3 carrier phases the neighbor
                                   # fields provide (optics N-step PSI);
                                   # temporally exact where static, blended
                                   # by total confidence
    symmetry_variant: bool = False # Transform-style spectral-symmetry
                                   # certified-chroma init variant
    lambda_c: float = 1.0      # chroma-smoothness vs luma-plausibility arbitration
    chroma_aniso: float = 0.5  # mu_h = aniso * lambda_c (chroma is broader
                               # horizontally). NOTE: tuned pre-field-pipeline
                               # when 'vertical' meant frame lines; field
                               # lines are 2x coarser — re-swept in the
                               # pre-port audit.
    charbonnier_eps: float = 0.5   # edge-preservation scale (IRE) for the luma prior
    chroma_eps: float = 1.0        # edge-preservation scale (IRE) for the chroma prior
    structure_coupling: float = 0.25  # parallel-level-sets Y->chi edge coupling
    frame_decode: bool = True      # weave fields, decode at frame geometry
    temporal_strength: float = 0.0   # 3D mode: weight of robust temporal priors
    bidirectional: bool = True       # 3D uses both n-1 and n+1 neighbors
    mc_search: int = 16              # block-matching search radius (px)
    mc_tile: int = 32                # block-matching tile size (px)
    nr_anchor: float = 1.0           # strength of decode->NR->re-encode anchor
    nr_eps: float = 0.0              # blend/re-encode robustness (IRE); 0 = auto
    nr_radius: int = 2               # temporal NR radius (+/- fields)
    ntsc_j: bool = False             # NTSC-J: no 7.5 IRE setup (black at
                                     # 0 IRE) — Japanese LaserDiscs. Wrong
                                     # setup = wrong black level/contrast.
    acc: bool = True                 # Automatic Color Control: calibrate
                                     # saturation from the MEASURED burst
                                     # amplitude (nominal 20 IRE), like
                                     # every TV since the 1960s. Corrects
                                     # level drift of the source chain.
    extended_temporal: bool = True   # also use fields f±3 (carrier offset
                                     # 2.25 cycles -> 90 deg, |dc|=sqrt(2):
                                     # two more non-degenerate equations;
                                     # f±4 would be carrier-in-phase, dc=0,
                                     # inert by geometry)
    drizzle: bool = False            # astronomy-style vertical 2x super-
                                     # resolution output (see drizzle_frame).
                                     # Horizontal SR would be pointless: at
                                     # 4fsc the ~4.2 MHz luma is already
                                     # oversampled 1.7x, no aliased detail
                                     # exists to recover. Vertically, 480
                                     # unfiltered scan lines DO alias — that
                                     # is where stacking recovers detail.
    coherence_gate: float = 0.6      # InSAR-style complex coherence gating
                                     # of temporal equations (0 = off);
                                     # value = weight of the coherence term
    chunk_frames: int = 6            # 3D streaming window (frames)
    chunk_overlap: int = 2           # temporal context overlap (frames)
    output_fidelity: bool = True     # DEFAULT purist: re-impose Y = S - Re[chi c]
                                     # on output. The NR/anchor machinery then
                                     # only *guides the Y/C separation*; the
                                     # final luma is the raw composite minus
                                     # reconstructed chroma — zero temporal
                                     # smoothing in the deliverable.
    passes: int = 1                  # fixed-point refinement passes over segment
    temporal_eps: float = 0.0        # motion gate scale (IRE); 0 = auto
                                     # (self-calibrated to ~7x measured noise)
                                     # => static => strong 3D constraint;
                                     # large diff => motion => falls back to 2D
    cg_iterations: int = 60    # 0 => pure holographic reconstruction
    init_lpf_h_mhz: float = 1.3     # horizontal chroma bandwidth for init
    init_lpf_v_cph: float = 60.0    # vertical bandwidth (cycles/active-picture-height)
    chroma_gain: float = 1.0
    monochrome: bool = False


# ------------------------------------------------------- phase / lock-in

def estimate_noise_ire(S: np.ndarray) -> float:
    """Robust per-field noise estimate (IRE), from the stride-4
    horizontal second difference: at 4fsc the carrier completes 360
    deg over 4 samples, so S[x] - 2 S[x+4] + S[x+8] cancels chroma AND
    smooth luma exactly, leaving noise (variance 6 sigma^2). MAD makes
    it immune to sparse luma detail."""
    d = S[:, :-8] - 2.0 * S[:, 4:-4] + S[:, 8:]
    # 25th percentile of |d|: even more outlier-proof than MAD, since
    # chroma/luma edges (incomplete cancellation) can be locally dense.
    # For Gaussian d, P25(|d|) = 0.3186 sigma_d, and sigma_d^2 = 6 s^2.
    q = np.percentile(np.abs(d - np.median(d)), 25)
    return float(q / 0.3186 / np.sqrt(6.0))


def burst_lockin_phase(field_ire: np.ndarray, p: VideoParameters) -> np.ndarray:
    """Per-line subcarrier phase offset via lock-in detection on the burst.

    Returns theta[line] such that phi(line, x) = theta[line] + (pi/2)*x.
    Lines with no detectable burst inherit the model phase (pi per line).
    """
    h, _ = field_ire.shape
    x = np.arange(p.colour_burst_start, p.colour_burst_end)
    ref = np.exp(-1j * (np.pi / 2.0) * x)           # local oscillator
    seg = field_ire[:, p.colour_burst_start:p.colour_burst_end]
    seg = seg - seg.mean(axis=1, keepdims=True)      # kill DC pedestal
    z = (seg * ref[None, :]).mean(axis=1)            # complex lock-in output
    amp = np.abs(z)

    # burst = A*sin(phi) = Re[(-iA) e^{i phi}] => lock-in z = (-iA/2) e^{i theta}
    # => theta = angle(z) + pi/2
    theta = np.angle(z) + np.pi / 2.0

    # Robust smoothing of the phase trajectory across lines
    # (Kalman/RTS idea from communications): after time-base
    # correction the true subcarrier phase varies *slowly* line to
    # line around the pi-per-line model, while burst measurements on
    # damaged lines (dropouts, head noise) are wild outliers. We
    # therefore solve, on the deviation d = theta - model,
    #     min_x  sum a_l (x_l - d_l)^2 + lam sum (x_{l+1} - x_l)^2
    # (a tridiagonal system, Thomas solve) with amplitude-derived
    # weights a_l, then one IRLS re-weighting to reject outliers.
    # Every downstream equation depends on phi, so hardening it here
    # hardens the whole decoder on real, damaged media.
    good = amp > (amp.max() * 0.2 if amp.max() > 0 else 1.0)
    if good.any():
        idx = np.where(good)[0]
        ref_line = idx[0]
        model = theta[ref_line] + np.pi * (np.arange(h) - ref_line)
        d = np.angle(np.exp(1j * (theta - model)))
        import os as _os
        if _os.environ.get("HVD_NO_PHASE_SMOOTH"):
            theta_u = model + d
            return np.where(good, theta_u, model)
        a = np.where(good, np.clip(amp / (np.median(amp[good]) + 1e-9),
                                   0.0, 2.0), 0.0)
        lam = 25.0
        x = _tridiag_smooth(d, a, lam)
        # IRLS outlier rejection (Huber-like, scale ~0.15 rad)
        r = np.abs(d - x)
        a2 = a * 0.15 / np.maximum(r, 0.15)
        x = _tridiag_smooth(d, a2, lam)
        theta = model + x
    return theta


def _tridiag_smooth(d, a, lam):
    """Solve (diag(a) + lam*L) x = a*d with L the 1-D graph Laplacian,
    via the Thomas algorithm. O(n), no dependencies."""
    n = len(d)
    diag = a + 2.0 * lam
    diag[0] -= lam
    diag[-1] -= lam
    off = np.full(n - 1, -lam)
    rhs = a * d
    # forward sweep
    c = np.empty(n - 1)
    dd = np.empty(n)
    c[0] = off[0] / diag[0]
    dd[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - off[i - 1] * c[i - 1]
        if i < n - 1:
            c[i] = off[i] / m
        dd[i] = (rhs[i] - off[i - 1] * dd[i - 1]) / m
    x = np.empty(n)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - c[i] * x[i + 1]
    return x


def burst_amplitude_ire(field_ire: np.ndarray, p: VideoParameters) -> float:
    """Measured colour-burst amplitude (IRE), for Automatic Color
    Control. Lock-in output magnitude |z| = A/2 for burst A*sin(phi);
    median over burst-bearing lines rejects damaged ones. Every
    analogue TV normalises chroma gain by this — a source chain whose
    levels drifted otherwise decodes over/under-saturated."""
    x = np.arange(p.colour_burst_start, p.colour_burst_end)
    seg = field_ire[:, p.colour_burst_start:p.colour_burst_end]
    seg = seg - seg.mean(axis=1, keepdims=True)
    z = (seg * np.exp(-1j * (np.pi / 2.0) * x)[None, :]).mean(axis=1)
    amp = 2.0 * np.abs(z)
    good = amp > 0.25 * np.median(amp[amp > 0]) if (amp > 0).any() else amp > -1
    return float(np.median(amp[good])) if good.any() else 20.0


def phase_map(theta: np.ndarray, width: int) -> np.ndarray:
    x = np.arange(width)
    return theta[:, None] + (np.pi / 2.0) * x[None, :]


# ------------------------------------------- holographic reconstruction

def _gaussian_lpf_kernel_fft(shape, p: VideoParameters, cfg: DecoderConfig):
    """Frequency response of the 2D Gaussian low-pass used to isolate the
    baseband chroma after carrier demodulation (the 'hologram crop')."""
    h, w = shape
    fx = fftfreq(w, d=1.0 / p.sample_rate)                 # Hz
    fy = fftfreq(h, d=1.0)                                  # cycles/line
    cutoff_x = cfg.init_lpf_h_mhz * 1e6
    cutoff_y = cfg.init_lpf_v_cph / (2.0 * h)               # cycles/line units
    gx = np.exp(-0.5 * (fx / cutoff_x) ** 2)
    gy = np.exp(-0.5 * (fy / cutoff_y) ** 2)
    return gy[:, None] * gx[None, :]


def holographic_init(S: np.ndarray, phi: np.ndarray,
                     p: VideoParameters, cfg: DecoderConfig):
    """Digital-holography reconstruction with Dubois-style adaptive
    sideband selection (borrowed from frequency-domain CFA
    demosaicing): TWO complementary anisotropic crops of the chroma
    sideband are computed — narrow-in-x/wide-in-y and the transpose —
    and blended per pixel according to which one leaves the locally
    smoothest residual luma (least cross-contamination). Same
    arbitration criterion as the refinement, applied at t=0: a better
    linearisation point for the IRLS."""
    carrier = np.exp(1j * phi)
    demod = S * np.conj(carrier)              # sideband -> DC, Y -> +/-fsc
    D = fft2(demod)

    variants = []
    for hx, vy in ((0.8, 120.0), (1.8, 30.0)):
        import dataclasses as _dc
        c2 = _dc.replace(cfg, init_lpf_h_mhz=hx, init_lpf_v_cph=vy)
        G = _gaussian_lpf_kernel_fft(S.shape, p, c2)
        chi_v = 2.0 * ifft2(D * G)
        Yv = S - np.real(chi_v * carrier)
        E = _box_blur(np.abs(_dx(Yv)) + np.abs(_dy(Yv)), r=3)
        variants.append((chi_v, 1.0 / (E + 0.5)))

    # --- Third variant: "Transform NTSC, repaired" -------------------
    # Easterbrook's Transform PAL extracts chroma by enforcing spectral
    # symmetry about the carrier — and was NEVER ported to NTSC because
    # I and Q share one carrier, so quadrature interference makes the
    # sidebands asymmetric ((kQ+kI) != (kQ-kI)), defeating the test as
    # a *detector*. But the test survives as a CERTIFIER: in the
    # demodulated domain, min(|Z(+k)|, |Z(-k)|) per point-reflected bin
    # pair is a lower bound of chroma that luma almost never fakes.
    # The asymmetric remainder — ambiguous between luma and genuinely
    # complex chroma — is precisely what the variational arbitration
    # exists to resolve. Division of labour: symmetry certifies, the
    # solver adjudicates the rest. (BBC RD Transform decoder,
    # jim-easterbrook.me.uk/pal, "The NTSC problem".)
    if cfg.symmetry_variant:
        mag = np.abs(D)
        mag_r = np.roll(np.roll(mag[::-1, ::-1], 1, axis=0), 1, axis=1)
        sym = np.minimum(mag, mag_r) / (mag + 1e-6)
        c3 = _dc.replace(cfg, init_lpf_h_mhz=1.3, init_lpf_v_cph=60.0)
        G3 = _gaussian_lpf_kernel_fft(S.shape, p, c3)
        chi_s = 2.0 * ifft2(D * sym * G3)
        Ys_ = S - np.real(chi_s * carrier)
        Es = _box_blur(np.abs(_dx(Ys_)) + np.abs(_dy(Ys_)), r=3)
        variants.append((chi_s, 1.0 / (Es + 0.5)))

    wsum = sum(w for (_, w) in variants)
    chi = sum(cv * w for (cv, w) in variants) / wsum
    Y = S - np.real(chi * carrier)
    return Y, chi


# ------------------------------------------------- variational refinement

def _laplacian(a: np.ndarray) -> np.ndarray:
    out = -4.0 * a
    out += np.roll(a, 1, 0) + np.roll(a, -1, 0)
    out += np.roll(a, 1, 1) + np.roll(a, -1, 1)
    return out


# --------------------------------------------- motion compensation (3D)

def _box_blur(a, r=2):
    k = np.ones(2 * r + 1) / (2 * r + 1)
    a = np.apply_along_axis(np.convolve, 0, a, k, 'same')
    return np.apply_along_axis(np.convolve, 1, a, k, 'same')


def _decimate(a, f):
    h, w = a.shape
    hh, ww = h // f * f, w // f * f
    return a[:hh, :ww].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _bm_pass(A, B, tile, dys, dxs, base_dy=None, base_dx=None):
    """One block-matching pass over the given shift candidates.
    If base_dy/base_dx (per-tile) are given, candidates are relative
    to them (pyramid refinement)."""
    h, w = A.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile
    ph, pw = th * tile, tw * tile
    max_s = max(max(abs(d) for d in dys), max(abs(d) for d in dxs))
    if base_dy is not None:
        max_s += int(max(np.abs(base_dy).max(), np.abs(base_dx).max()))
    Bp = np.pad(B, max_s + 1, mode="edge")

    def tilesum(D):
        Dp = np.zeros((ph, pw))
        Dp[:h, :w] = D
        return Dp.reshape(th, tile, tw, tile).sum(axis=(1, 3))

    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // tile, th - 1)
    tx = np.minimum(xx // tile, tw - 1)

    best = np.full((th, tw), np.inf)
    mdy = np.zeros((th, tw), np.int32)
    mdx = np.zeros((th, tw), np.int32)
    cost0 = None
    for dy in dys:
        for dx in dxs:
            if base_dy is None:
                Bs = Bp[max_s + 1 - dy:max_s + 1 - dy + h,
                        max_s + 1 - dx:max_s + 1 - dx + w]
                tot_dy, tot_dx = dy, dx
            else:
                sy = base_dy[ty, tx] + dy
                sx = base_dx[ty, tx] + dx
                Bs = Bp[(max_s + 1) + yy - sy, (max_s + 1) + xx - sx]
                tot_dy, tot_dx = None, None
            se = tilesum((A - Bs) ** 2)
            if base_dy is None and dy == 0 and dx == 0:
                cost0 = se
            bias = 1.0 + 0.02 * (abs(dy) + abs(dx))
            cost = se * bias
            upd = cost < best
            best[upd] = cost[upd]
            if base_dy is None:
                mdy[upd] = tot_dy
                mdx[upd] = tot_dx
            else:
                mdy[upd] = (base_dy + dy)[upd]
                mdx[upd] = (base_dx + dx)[upd]
    return mdy, mdx, best, cost0


def estimate_motion(Y_ref, Y_cur, tile=32, search=8):
    """Coarse-to-fine tiled block matching (integer-pel).

    Level 0: 4x-decimated exhaustive search over the full radius.
    Level 1: full-resolution refinement of +/-3 px around the coarse
    vector, plus an explicit zero-motion candidate set for the margin
    rule. ~50x cheaper than exhaustive full-resolution search while
    keeping the same behaviour on the benchmarks.

    Confidence layers (unchanged): tile-energy, median-calibrated
    outlier rejection, and global pair validity (scene-cut detector).
    """
    A = _box_blur(Y_cur)
    B = _box_blur(Y_ref)
    h, w = A.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile

    f = 4
    cs = max(1, int(np.ceil(search / f)))
    Ad, Bd = _decimate(A, f), _decimate(B, f)
    cdy, cdx, _, _ = _bm_pass(Ad, Bd, max(4, tile // f),
                              range(-cs, cs + 1), range(-cs, cs + 1))
    # upscale coarse vectors to full-res tile grid
    hh, ww = cdy.shape
    gy = (np.arange(th) * hh // th).clip(0, hh - 1)
    gx = (np.arange(tw) * ww // tw).clip(0, ww - 1)
    base_dy = (cdy[gy][:, gx] * f).astype(np.int32)
    base_dx = (cdx[gy][:, gx] * f).astype(np.int32)

    mdy, mdx, best, _ = _bm_pass(A, B, tile, range(-3, 4), range(-3, 4),
                                 base_dy=base_dy, base_dx=base_dx)
    # half-pel refinement: parabolic fit on the SSD around the winner
    def _cost_at(dy_off, dx_off):
        m1, m2, c, _ = _bm_pass(A, B, tile, [0], [0],
                                base_dy=mdy + dy_off, base_dx=mdx + dx_off)
        return c
    cy_m, cy_p = _cost_at(-1, 0), _cost_at(1, 0)
    cx_m, cx_p = _cost_at(0, -1), _cost_at(0, 1)
    def _para(cm, c0, cp):
        den = cm - 2 * c0 + cp
        sub = np.where(den > 1e-9, 0.5 * (cm - cp) / np.maximum(den, 1e-9), 0.0)
        return np.clip(sub, -0.5, 0.5)
    sdy = _para(cy_m, best, cy_p)
    sdx = _para(cx_m, best, cx_p)
    # explicit zero-motion evaluation for the margin rule
    zdy, zdx, best0, cost0 = _bm_pass(A, B, tile, [0], [0])

    take0 = best0 < best
    mdy[take0] = 0
    mdx[take0] = 0
    best[take0] = best0[take0]

    keep_zero = best > 0.85 * cost0
    mdy[keep_zero] = 0
    mdx[keep_zero] = 0
    best[keep_zero] = cost0[keep_zero]
    # sub-pixel is meaningless on vectors the margin rule just forced
    sdy[keep_zero] = 0.0
    sdx[keep_zero] = 0.0

    def tilesum(D):
        ph, pw = th * tile, tw * tile
        Dp = np.zeros((ph, pw))
        Dp[:h, :w] = D
        return Dp.reshape(th, tile, tw, tile).sum(axis=(1, 3))

    tmean = tilesum(A) / (tile * tile)
    tvar = tilesum(A ** 2) / (tile * tile) - tmean ** 2
    energy = (tvar + 1.0) * (tile * tile)
    conf = np.clip(1.0 - best / energy, 0.0, 1.0)

    r = best / (tile * tile)
    med = np.median(r)
    excess = np.maximum(0.0, r - med)
    conf *= (med + 1.0) / (med + 1.0 + excess)
    conf *= 6.0 / (6.0 + med)
    return mdy + sdy, mdx + sdx, conf


def _vectors_per_pixel(mdy, mdx, tile, shape):
    """OBMC-style smooth motion field (video-coding idea): bilinearly
    interpolate the per-tile vectors between TILE CENTERS, giving one
    float vector per pixel. Removes the visible seams of piecewise-
    constant tile warps on complex motion; every consumer of the
    motion field (raw equations, anchor blend) benefits at once. At
    genuine motion boundaries the interpolated in-between vectors are
    wrong for a few pixels — the per-pixel robust gates absorb that,
    exactly as they absorb any other mismatch."""
    h, w = shape
    th, tw = mdy.shape
    # 3x3 component-wise vector median first (video-coding classic):
    # isolated spurious vectors from flat tiles would otherwise BLEED
    # into their neighbors through the interpolation, instead of being
    # confined as with nearest-tile warping. The median squashes
    # isolated outliers while genuine motion fields (>= 2x2 tiles)
    # pass through untouched.
    def _med3(V):
        if V.shape[0] < 3 or V.shape[1] < 3:
            return V
        Vp = np.pad(V, 1, mode="edge")
        stack = [Vp[1+dy:1+dy+V.shape[0], 1+dx:1+dx+V.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        return np.median(np.stack(stack), axis=0)
    # outlier-snap rather than blanket median: a vector is replaced
    # by its 3x3 median only when it disagrees with it by > 3 px, so
    # coherent motion clusters (even small objects) pass untouched
    # while isolated flat-tile garbage is squashed before it can
    # bleed through the interpolation.
    my, mx = _med3(mdy), _med3(mdx)
    bad = (np.abs(mdy - my) + np.abs(mdx - mx)) > 3.0
    mdy = np.where(bad, my, mdy)
    mdx = np.where(bad, mx, mdx)
    cy = (np.arange(th) + 0.5) * tile
    cx = (np.arange(tw) + 0.5) * tile
    yy = np.arange(h, dtype=np.float64)
    xx = np.arange(w, dtype=np.float64)
    iy = np.clip(np.searchsorted(cy, yy) - 1, 0, th - 2) if th > 1 else np.zeros(h, int)
    ix = np.clip(np.searchsorted(cx, xx) - 1, 0, tw - 2) if tw > 1 else np.zeros(w, int)
    if th > 1:
        fy = np.clip((yy - cy[iy]) / tile, 0.0, 1.0)
    else:
        fy = np.zeros(h)
    if tw > 1:
        fx = np.clip((xx - cx[ix]) / tile, 0.0, 1.0)
    else:
        fx = np.zeros(w)
    def interp(V):
        v00 = V[iy][:, ix]
        v10 = V[np.minimum(iy + 1, th - 1)][:, ix]
        v01 = V[iy][:, np.minimum(ix + 1, tw - 1)]
        v11 = V[np.minimum(iy + 1, th - 1)][:, np.minimum(ix + 1, tw - 1)]
        return (v00 * (1 - fy)[:, None] * (1 - fx)[None, :]
                + v10 * fy[:, None] * (1 - fx)[None, :]
                + v01 * (1 - fy)[:, None] * fx[None, :]
                + v11 * fy[:, None] * fx[None, :])
    return interp(mdy.astype(np.float64)), interp(mdx.astype(np.float64))


def warp_by_tiles(a, mdy, mdx, tile=32):
    """Shift each tile of `a` by its (dy, dx); nearest-tile, integer-pel
    (float vectors are rounded here; sub-pixel paths use the bilinear
    envelope warp)."""
    h, w = a.shape[:2]
    vdy, vdx = _vectors_per_pixel(np.asarray(mdy, float),
                                  np.asarray(mdx, float), tile, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    sy = np.clip(yy - np.round(vdy).astype(int), 0, h - 1)
    sx = np.clip(xx - np.round(vdx).astype(int), 0, w - 1)
    return a[sy, sx]


def mc_warp(Y_from, Y_to, arrays, tile=32, search=16):
    """Estimate motion from Y_from toward Y_to, warp every array in
    `arrays` accordingly. Returns (warped_list, per-pixel confidence)."""
    mdy, mdx, conf = estimate_motion(Y_from, Y_to, tile=tile, search=search)
    h, w = Y_from.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // tile, conf.shape[0] - 1)
    tx = np.minimum(xx // tile, conf.shape[1] - 1)
    conf_px = _box_blur(conf[ty, tx] ** 2, r=8)  # squared: crush weak matches
    return [warp_by_tiles(a, mdy, mdx, tile) for a in arrays], conf_px


def envelope_of(S, phi):
    """Line-pair comb separation of a raw field into baseband envelope
    samples on the HALF-LINE grid (between lines m and m+1):
        Yb  = (S[m] + S[m+1]) / 2      (adjacent lines are 180 deg of
                                        carrier apart: chroma cancels)
        Cb  = (S[m] - S[m+1]) / 2 = Re[chi c_m]
    plus the quadrature from the +/-1 sample x-shift (90 deg at 4fsc):
        chi_b = (Cb + i (Cb(x-1) - Cb(x+1))/2) * conj(c_m)
    A fixed, local, linear transform of raw measurements — nothing is
    estimated. Being baseband, (Yb, chi_b) can be resampled at
    arbitrary sub-pixel positions and re-encoded at ANY carrier phase.
    """
    Yb = 0.5 * (S[:-1] + S[1:])
    Cb = 0.5 * (S[:-1] - S[1:])
    q = 0.5 * (np.roll(Cb, 1, axis=1) - np.roll(Cb, -1, axis=1))
    chi_b = (Cb + 1j * q) * np.exp(-1j * phi[:-1])
    return Yb.astype(np.float32), chi_b.astype(np.complex64)


def _warp_bilinear_tiles(a, dyf, dxf, tile, row_offset=0.0, out_shape=None):
    """Bilinear warp of a baseband array by per-tile float vectors,
    with a constant extra row offset (grid alignment). The output grid
    may differ from the source grid (envelope arrays have L-1 rows)."""
    ho, wo = out_shape if out_shape is not None else a.shape[:2]
    ho, wo = int(ho), int(wo)
    yy = np.arange(ho, dtype=np.float64)[:, None] * np.ones((1, wo))
    xx = np.ones((ho, 1)) * np.arange(wo, dtype=np.float64)[None, :]
    vdy, vdx = _vectors_per_pixel(np.asarray(dyf, float),
                                  np.asarray(dxf, float), tile, (ho, wo))
    sy = yy - vdy + row_offset
    sx = xx - vdx
    hs, ws = a.shape[:2]
    sy = np.clip(sy, 0, hs - 1.001)
    sx = np.clip(sx, 0, ws - 1.001)
    y0 = sy.astype(int); x0 = sx.astype(int)
    fy = (sy - y0); fx = (sx - x0)
    v = (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0 + 1, x0] * fy * (1 - fx)
         + a[y0, x0 + 1] * (1 - fy) * fx + a[y0 + 1, x0 + 1] * fy * fx)
    return v


def motion_compensate_envelope(nb_state, Y_cur, phi_cur, parity_cur,
                               parity_nb, tile=32, search=16, motion=None):
    """Envelope-domain neighbor resampling (points 3+4 unified).

    The neighbor's raw field is comb-separated into baseband
    (Yb, chi_b) on its half-line grid, motion-warped with SUB-PIXEL
    bilinear interpolation (legal in baseband, forbidden on the raw
    carrier), then RE-ENCODED at the current field's own phase + 180
    deg. Consequences:
      * the half-line parity offset between opposite fields vanishes
        by construction (the comb's half-grid IS the other parity's
        grid);
      * sub-pixel motion is honoured (parabolic half-pel vectors);
      * |dc| = 2 for every neighbor — maximal chroma leverage — where
        the raw adjacent-field pairing only reached sqrt(2).
    Grid algebra (frame lines): target field line m sits at
    2m + p_cur; neighbor half-sample i sits at 2i + p_nb + 1; matching
    measures dy in field lines, so the source half-grid row is
    m - dy + (p_cur - p_nb - 1)/2.
    """
    Y_nb, S_nb, phi_nb = nb_state
    if motion is None:
        motion = estimate_motion(Y_nb, Y_cur, tile=tile, search=search)
    mdy, mdx, conf = motion
    Yb, chib = envelope_of(S_nb, phi_nb)
    row_off = (parity_cur - parity_nb - 1) / 2.0
    out = Y_cur.shape
    Yw = _warp_bilinear_tiles(Yb, mdy, mdx, tile, row_offset=row_off,
                              out_shape=out)
    Cw = (_warp_bilinear_tiles(chib.real, mdy, mdx, tile,
                               row_offset=row_off, out_shape=out)
          + 1j * _warp_bilinear_tiles(chib.imag, mdy, mdx, tile,
                                      row_offset=row_off, out_shape=out))
    c_w = -np.exp(1j * phi_cur)
    S_w = Yw + np.real(Cw * c_w)
    h, w = Y_nb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // tile, conf.shape[0] - 1)
    tx = np.minimum(xx // tile, conf.shape[1] - 1)
    conf_px = _box_blur(conf[ty, tx] ** 2, r=8)
    return S_w.astype(np.float32), c_w, conf_px


def complex_coherence(z1, z2, r=6):
    """InSAR-style local complex coherence between two chroma fields:
        gamma = |<z1 conj(z2)>| / sqrt(<|z1|^2><|z2|^2>)
    Phase-sensitive where an SSD is not: two chroma fields can have
    similar energy yet decorrelated phase (motion residual, content
    change) — exactly the situation where a temporal chroma equation
    turns toxic. gamma ~ 1 where the phasors line up, -> small where
    they decorrelate."""
    num = np.abs(_box_blur((z1 * np.conj(z2)).real, r)
                 + 1j * _box_blur((z1 * np.conj(z2)).imag, r))
    den = np.sqrt(_box_blur(np.abs(z1) ** 2, r)
                  * _box_blur(np.abs(z2) ** 2, r)) + 1e-6
    return np.clip(num / den, 0.0, 1.0)


def motion_compensate_prev(prev, Y_cur_init, tile=32, search=16, motion=None):
    """Warp a neighbor frame's *raw measurements* toward the current
    frame: its composite S and its carrier phase map phi (per-pixel
    values copied exactly — integer-pel shifts mean no carrier-phase
    interpolation error). Matching runs on decoded luma (clean of
    carrier), the data comes from the raw composite: the current frame
    gets extra measurement equations, never recycled estimates.
    Returns (S_warped, carrier_warped, confidence)."""
    Yp, Sp, phip = prev
    if motion is None:
        motion = estimate_motion(Yp, Y_cur_init, tile=tile, search=search)
    mdy, mdx, conf = motion
    h, w = Yp.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // tile, conf.shape[0] - 1)
    tx = np.minimum(xx // tile, conf.shape[1] - 1)
    conf_px = _box_blur(conf[ty, tx] ** 2, r=8)
    S_w = warp_by_tiles(Sp, mdy, mdx, tile)
    phi_w = warp_by_tiles(phip, mdy, mdx, tile)
    return S_w, np.exp(1j * phi_w), conf_px


def synth_reference(j, Ys, chis, SP, cfg, parities=None):
    """The re-encode loop (user-suggested pass structure): build a
    denoised reference for frame j by motion-compensated temporal
    blending of the *decoded* fields, then re-encode it to composite
    for an honesty check.

    NTSC geometry does the heavy lifting: separation leakage
    (cross-colour in chi, dot-crawl in Y) anti-correlates at +/-1
    frame (180 deg carrier flip), so the blend cancels it; at +/-2
    frames the carrier is back in phase, so those neighbors average
    noise without touching the signal. The blend is robust
    (Geman-McClure per-pixel weights), so motion residuals drop out.

    Returns (Y_hat, chi_hat, conf_ref) where conf_ref is measured in
    the *composite domain*: 1 where the re-encoded synthesis explains
    the raw measurement, ->0 where it does not (the anchor never
    trusts a reference the data contradicts).
    """
    S, phi = SP[j]
    eps_b = cfg.nr_eps
    accY = Ys[j].copy()
    accC = chis[j].copy()
    accW = np.ones_like(Ys[j])
    tile = cfg.mc_tile
    hh, ww = Ys[j].shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    for k in range(j - cfg.nr_radius, j + cfg.nr_radius + 1):
        if k == j or not (0 <= k < len(Ys)):
            continue
        mdy, mdx, cf = estimate_motion(Ys[k], Ys[j], tile=tile,
                                       search=cfg.mc_search)
        ty = np.minimum(yy // tile, cf.shape[0] - 1)
        tx = np.minimum(xx // tile, cf.shape[1] - 1)
        conf = _box_blur(cf[ty, tx] ** 2, r=8)
        # decoded fields are baseband: bilinear sub-pixel warp is
        # legal, and the half-line parity offset between opposite
        # fields is compensated exactly on the sampling grid
        # (points 3+4 of the external review, in their safe home)
        if parities is not None:
            row_off = (parities[j] - parities[k]) / 2.0
        else:
            row_off = ((j % 2) - (k % 2)) / 2.0
        Yw = _warp_bilinear_tiles(Ys[k], mdy, mdx, tile,
                                  row_offset=row_off, out_shape=(hh, ww))
        Cw = (_warp_bilinear_tiles(chis[k].real, mdy, mdx, tile,
                                   row_offset=row_off, out_shape=(hh, ww))
              + 1j * _warp_bilinear_tiles(chis[k].imag, mdy, mdx, tile,
                                          row_offset=row_off,
                                          out_shape=(hh, ww)))
        d2 = (Yw - Ys[j]) ** 2 + np.abs(Cw - chis[j]) ** 2
        w = conf * eps_b ** 2 / (d2 + eps_b ** 2)
        accY += w * Yw
        accC += w * Cw
        accW += w
    Y_hat = accY / accW
    chi_hat = accC / accW

    S_hat = Y_hat + np.real(chi_hat * np.exp(1j * phi))
    r = _box_blur(np.abs(S - S_hat), r=1)
    conf_ref = cfg.nr_eps ** 2 / (r ** 2 + cfg.nr_eps ** 2)
    return Y_hat, chi_hat, conf_ref


def _dx(a):
    """Forward difference with Neumann (zero-gradient) boundary: the
    last column's difference is 0 instead of wrapping to column 0.
    Periodic rolls couple opposite image borders through the priors —
    a subtle but real artefact source flagged by the pre-port audit.
    _dxT is the exact adjoint of this operator (required for the CG
    gradients to be true gradients)."""
    d = np.empty_like(a)
    d[:, :-1] = a[:, 1:] - a[:, :-1]
    d[:, -1] = 0
    return d


def _dy(a):
    d = np.empty_like(a)
    d[:-1] = a[1:] - a[:-1]
    d[-1] = 0
    return d


def _dxT(a):
    """Adjoint of _dx (negative divergence with matching boundary)."""
    d = np.empty_like(a)
    d[:, 0] = -a[:, 0]
    d[:, 1:-1] = a[:, :-2] - a[:, 1:-1]
    d[:, -1] = a[:, -2]
    return d


def _dyT(a):
    d = np.empty_like(a)
    d[0] = -a[0]
    d[1:-1] = a[:-2] - a[1:-1]
    d[-1] = a[-2]
    return d


def variational_refine(S, phi, Y0, chi0, cfg: DecoderConfig,
                       irls_outer: int = 4, neighbors=None):
    """Refine chroma by arbitration, with luma eliminated.

    Key structural fact: the data term is *invariant* to transfers
    between Y and Re[chi c] — the holographic init already fits the
    composite exactly. So the only meaningful question is: which
    component should own each bit of signal? We enforce exact data
    fidelity by substitution, Y := S - Re[chi e^{i phi}], and solve

        argmin_chi   sum rho(grad Y(chi))
                   + mu   * sum rho_c(grad chi)
                   + nu * sum_k sum rho_t(r_t[k])          [3D mode]

    where r_t[k] is a *raw-measurement* residual against temporal
    neighbor k (past and/or future frame), motion-compensated.

    with rho, rho_c, rho_t Charbonnier penalties (edge/motion
    preserving, solved by IRLS / lagged diffusivity). Dot crawl is
    carrier-frequency ripple in Y => hugely penalised by rho => it
    migrates into chi. Twin-image/cross-colour is a 2*fsc oscillation
    in chi => hugely penalised by rho_c => it migrates back into Y.
    The two classic NTSC artefacts arbitrate each other.

    Joint structure coupling ("parallel level sets", borrowed from
    multi-modal PET/MRI reconstruction): luma and chroma edges
    co-occur in natural images, so the chroma diffusivity is opened
    wherever the residual luma sees an edge. This kills the classic
    "hanging dots" at vertical chroma transitions.

    3D mode (prev = (Y_prev, chi_prev) from the previous frame): the
    NTSC carrier flips 180 deg frame-to-frame, so for static pixels
    the temporal constraint disambiguates Y/C *exactly* like a 3D comb
    — but the robust penalty makes motion handling continuous: a large
    temporal difference saturates rho_t, its IRLS weight -> 0, and the
    pixel falls back to the spatial priors. Motion adaptivity without
    any motion detector or binary switching (same trick as robust
    penalties in variational optical flow).
    """
    eps = cfg.charbonnier_eps
    eps_c = cfg.chroma_eps
    eps_t = cfg.temporal_eps
    mu_h = cfg.lambda_c * cfg.chroma_aniso   # chroma is broader horizontally
    mu_v = cfg.lambda_c
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    carrier = np.exp(1j * phi)
    # per-neighbor: MC-warped composite, carrier difference, confidence
    nbr = [(S_w, carrier - c_w, conf) for (S_w, c_w, conf) in neighbors]
    chi = chi0.copy()

    n_inner = max(1, cfg.cg_iterations // max(1, irls_outer))

    def luma(chi):
        return S - np.real(chi * carrier)

    def temporal_residual(chi, S_w, dc):
        # r_t = S_neighbor_warped - Y - Re[chi c_neighbor]  (Y eliminated)
        #     = (S_neighbor_warped - S) + Re[chi (c - c_neighbor)]
        # For a static pixel c_neighbor = -c (180 deg frame flip,
        # measured via burst, never assumed), so r_t = 0 forces
        # Re[chi c] = (S - S_nb)/2 : exact 3D-comb separation. With
        # both past and future neighbors, failures are complementary:
        # scene cuts and occlusions break at most one side.
        return (S_w - S) + np.real(chi * dc)

    for _outer in range(irls_outer):
        Y = luma(chi)
        gxY, gyY = _dx(Y), _dy(Y)
        # normalised diffusivities: ~1 in flat areas, ->0 across edges,
        # preserving the calibrated luma/chroma balance
        wx = eps / np.sqrt(gxY ** 2 + eps ** 2)
        wy = eps / np.sqrt(gyY ** 2 + eps ** 2)

        # chroma diffusivity: own edges + joint coupling with luma edges
        cx, cy = np.abs(_dx(chi)), np.abs(_dy(chi))
        k = cfg.structure_coupling
        wcx = eps_c / np.sqrt(cx ** 2 + k * gxY ** 2 + eps_c ** 2)
        wcy = eps_c / np.sqrt(cy ** 2 + k * gyY ** 2 + eps_c ** 2)

        # temporal confidence per neighbor: Geman-McClure on the
        # *composite-domain* residual (what that neighbor's raw
        # measurement cannot explain), pre-gated by block-matching
        # confidence. Static or well-compensated pixels get extra data
        # equations (exact 3D-comb determination); motion/mismatch =>
        # wt -> 0 => graceful per-pixel fallback towards 2D.
        wts = []
        if nu > 0.0:
            for S_w, dc, conf in nbr:
                rt = temporal_residual(chi, S_w, dc)
                wts.append(conf * eps_t ** 2 / (rt ** 2 + eps_t ** 2))

        # quadratic majorant:
        #  E~ = sum wx (Dx Y)^2 + wy (Dy Y)^2
        #     + mu_h wcx |Dx chi|^2 + mu_v wcy |Dy chi|^2
        #     + nu sum_k wt_k r_tk^2
        def grad(chi):
            Yc = luma(chi)
            g_img = 2.0 * (_dxT(wx * _dx(Yc)) + _dyT(wy * _dy(Yc)))
            gC = (-g_img * np.conj(carrier)
                  + 2.0 * (mu_h * _dxT(wcx * _dx(chi))
                           + mu_v * _dyT(wcy * _dy(chi))))
            for (S_w, dc, conf), wt in zip(nbr, wts):
                gC += (2.0 * nu * wt
                       * temporal_residual(chi, S_w, dc) * np.conj(dc))
            return gC

        def curv(dC):
            dY = -np.real(dC * carrier)
            h = (np.sum(wx * _dx(dY) ** 2) + np.sum(wy * _dy(dY) ** 2)
                 + mu_h * np.sum(wcx * np.abs(_dx(dC)) ** 2)
                 + mu_v * np.sum(wcy * np.abs(_dy(dC)) ** 2))
            for (S_w, dc, conf), wt in zip(nbr, wts):
                h += nu * np.sum(wt * np.real(dC * dc) ** 2)
            return h

        g = grad(chi)
        d = -g
        gg = np.real(np.sum(np.conj(g) * g))
        for _it in range(n_inner):
            H = curv(d)
            if H <= 1e-12:
                break
            gd = np.real(np.sum(np.conj(g) * d))
            alpha = -0.5 * gd / H
            chi = chi + alpha * d
            g_new = grad(chi)
            gg_new = np.real(np.sum(np.conj(g_new) * g_new))
            beta = gg_new / max(gg, 1e-30)
            d = -g_new + beta * d
            g, gg = g_new, gg_new
            if gg < 1e-10 * S.size:
                break

    return luma(chi), chi


# ------------------------------------------------------------ field/frame

def variational_refine_joint(S, phi, Y0, chi0, cfg: DecoderConfig,
                             irls_outer: int = 4, neighbors=None,
                             anchor=None):
    """Pass-2+ solver with *relaxed* data fidelity (user-suggested
    re-encode loop). Unknowns are now BOTH Y and chi:

        E = ||S - Y - Re[chi c]||^2                       (soft data)
          + nu   sum_k wt_k (S_k - Y - Re[chi c_k])^2     (raw temporal)
          + nu_a sum w_a [ (Y - Y_hat)^2 + |chi - chi_hat|^2 ]  (anchor)
          + spatial Charbonnier priors on grad Y, grad chi

    Rationale: the exact-fidelity pass-1 formulation forces Y to
    inherit *all* composite noise (Y = S - Re[chi c]), which caps PSNR
    at the raw noise floor. Relaxing fidelity is only safe with a
    trustworthy prior — and the decode -> MC temporal NR -> re-encode
    loop manufactures exactly that: (Y_hat, chi_hat) aggregates up to
    2*nr_radius+1 raw measurements per pixel, and w_a (composite-domain
    re-encode check) disables the anchor wherever the synthesis fails
    to explain the raw data. Y may now shed noise toward Y_hat while
    the soft data term keeps the solution honest.
    """
    eps = cfg.charbonnier_eps
    eps_c = cfg.chroma_eps
    eps_t = cfg.temporal_eps
    mu_h = cfg.lambda_c * cfg.chroma_aniso
    mu_v = cfg.lambda_c
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    nu_a = cfg.nr_anchor if anchor is not None else 0.0
    if anchor is not None:
        Y_hat, chi_hat, w_a = anchor
    carrier = np.exp(1j * phi)
    nbr = [(S_w, c_w, conf) for (S_w, c_w, conf) in neighbors]

    Y = Y0.copy()
    chi = chi0.copy()
    n_inner = max(1, cfg.cg_iterations // max(1, irls_outer))

    for _outer in range(irls_outer):
        gxY, gyY = _dx(Y), _dy(Y)
        wx = eps / np.sqrt(gxY ** 2 + eps ** 2)
        wy = eps / np.sqrt(gyY ** 2 + eps ** 2)
        cx, cy = np.abs(_dx(chi)), np.abs(_dy(chi))
        k = cfg.structure_coupling
        wcx = eps_c / np.sqrt(cx ** 2 + k * gxY ** 2 + eps_c ** 2)
        wcy = eps_c / np.sqrt(cy ** 2 + k * gyY ** 2 + eps_c ** 2)
        wts = [conf * eps_t ** 2 /
               ((S_w - Y - np.real(chi * c_w)) ** 2 + eps_t ** 2)
               for (S_w, c_w, conf) in nbr] if nu > 0.0 else []

        def grad(Y, chi):
            r0 = S - Y - np.real(chi * carrier)
            gY = -2.0 * r0
            gC = -2.0 * r0 * np.conj(carrier)
            for (S_w, c_w, conf), wt in zip(nbr, wts):
                rk = S_w - Y - np.real(chi * c_w)
                gY += -2.0 * nu * wt * rk
                gC += -2.0 * nu * wt * rk * np.conj(c_w)
            if nu_a > 0.0:
                gY += 2.0 * nu_a * w_a * (Y - Y_hat)
                gC += 2.0 * nu_a * w_a * (chi - chi_hat)
            gY += 2.0 * (_dxT(wx * _dx(Y)) + _dyT(wy * _dy(Y)))
            gC += 2.0 * (mu_h * _dxT(wcx * _dx(chi))
                         + mu_v * _dyT(wcy * _dy(chi)))
            return gY, gC

        def curv(dY, dC):
            d0 = dY + np.real(dC * carrier)
            h = np.sum(d0 * d0)
            for (S_w, c_w, conf), wt in zip(nbr, wts):
                dk = dY + np.real(dC * c_w)
                h += nu * np.sum(wt * dk * dk)
            if nu_a > 0.0:
                h += nu_a * np.sum(w_a * (dY ** 2 + np.abs(dC) ** 2))
            h += np.sum(wx * _dx(dY) ** 2) + np.sum(wy * _dy(dY) ** 2)
            h += (mu_h * np.sum(wcx * np.abs(_dx(dC)) ** 2)
                  + mu_v * np.sum(wcy * np.abs(_dy(dC)) ** 2))
            return h

        gY, gC = grad(Y, chi)
        dY, dC = -gY, -gC
        gg = np.sum(gY * gY) + np.real(np.sum(np.conj(gC) * gC))
        for _it in range(n_inner):
            H = curv(dY, dC)
            if H <= 1e-12:
                break
            gd = np.sum(gY * dY) + np.real(np.sum(np.conj(gC) * dC))
            alpha = -0.5 * gd / H
            Y = Y + alpha * dY
            chi = chi + alpha * dC
            gY2, gC2 = grad(Y, chi)
            gg2 = np.sum(gY2 * gY2) + np.real(np.sum(np.conj(gC2) * gC2))
            beta = gg2 / max(gg, 1e-30)
            dY = -gY2 + beta * dY
            dC = -gC2 + beta * dC
            gY, gC, gg = gY2, gC2, gg2
            if gg < 1e-10 * S.size:
                break

    return Y, chi


def decode_field(field_raw: np.ndarray, p: VideoParameters,
                 cfg: DecoderConfig):
    """Decode one field -> (Y, U, V) over the active picture area."""
    ire = p.ire(field_raw)
    theta = burst_lockin_phase(ire, p)

    a0, a1 = p.active_video_start, p.active_video_end
    S = ire[:, a0:a1]
    phi = phase_map(theta, p.field_width)[:, a0:a1]

    Y0, chi0 = holographic_init(S, phi, p, cfg)
    if cfg.cg_iterations > 0 and not cfg.monochrome:
        Y, chi = variational_refine(S, phi, Y0, chi0, cfg)
    else:
        Y, chi = Y0, chi0

    if cfg.monochrome:
        chi = np.zeros_like(chi)

    V = np.real(chi) * cfg.chroma_gain
    U = -np.imag(chi) * cfg.chroma_gain
    return Y, U, V


def drizzle_frame(j0, Ys, chis, parities, cfg, scale=2):
    """Astronomy-style 'drizzle' stacking, adapted to interlaced NTSC:
    vertical super-resolution by SCATTER accumulation.

    Principle (Fruchter & Hook, HST): when many observations of the
    same scene exist at different sub-pixel offsets, depositing each
    source sample at its *mapped* position on a finer grid recovers
    detail beyond any single observation's sampling — provided the
    signal genuinely aliases (information exists between samples).
    For NTSC that is true ONLY vertically: 480 scan lines with no
    optical vertical prefilter alias heavily, while horizontally the
    4fsc sampling already oversamples the analog luma bandwidth.

    Offsets come from two sources, both already measured:
      * the intrinsic half-line parity offset between opposite fields
        (a free, guaranteed phase);
      * sub-pixel vertical motion (parabolic half-pel vectors).

    Robust per-pixel weights (Geman-McClure agreement with the target
    field's own decode, times block-matching confidence) reject
    motion/occlusion outliers, so this degrades gracefully to a plain
    2x weave-interpolation where no valid extra phases exist.

    Output grid: (scale * lines_per_field * 2 interleaved) rows —
    i.e. frame vertical resolution x scale — at native width.
    Scatter uses linear splitting between the two nearest fine rows
    (the 'pixfrac=1, linear kernel' flavour of drizzle).
    """
    L, W = Ys[j0].shape
    HF = 2 * L * scale                     # fine frame height
    accY = np.zeros((HF, W)); accC = np.zeros((HF, W), complex)
    accW = np.zeros((HF, W))
    eps_b = cfg.nr_eps if cfg.nr_eps > 0 else 3.0

    # target frame = fields j0 (parity 0 rows) and j0+1
    for jt in (j0, j0 + 1):
        pt = parities[jt]
        for k in range(max(0, j0 - 2 * cfg.nr_radius),
                       min(len(Ys), j0 + 2 + 2 * cfg.nr_radius)):
            pk = parities[k]
            if k in (j0, j0 + 1) and k != jt:
                continue  # the sibling field deposits on its own turn
            if k == jt:
                mdy = np.zeros((1, 1)); mdx = np.zeros((1, 1))
                conf = np.ones((1, 1))
            else:
                mdy, mdx, conf = estimate_motion(Ys[k], Ys[jt],
                                                 tile=cfg.mc_tile,
                                                 search=cfg.mc_search)
            vy, vx = _vectors_per_pixel(np.asarray(mdy, float),
                                        np.asarray(mdx, float),
                                        cfg.mc_tile, (L, W))
            th, tw = conf.shape
            yy, xx = np.mgrid[0:L, 0:W]
            cty = np.minimum(yy // cfg.mc_tile, th - 1)
            ctx = np.minimum(xx // cfg.mc_tile, tw - 1)
            cpx = conf[cty, ctx]

            # robust agreement with the target field's own decode,
            # evaluated at the integer-rounded warp (cheap)
            syi = np.clip(np.round(yy - vy).astype(int), 0, L - 1)
            sxi = np.clip(np.round(xx - vx).astype(int), 0, W - 1)
            d2 = ((Ys[k] - Ys[jt][syi, sxi]) ** 2
                  + np.abs(chis[k] - chis[jt][syi, sxi]) ** 2)
            w = cpx * eps_b ** 2 / (d2 + eps_b ** 2)

            # source sample (y, x) of field k lands, in TARGET FRAME
            # fine rows, at: (2*(y + vy) + pk) * scale ... using the
            # motion measured toward field jt (field-line units)
            yf = (2.0 * (yy + vy) + pk) * scale
            xs = np.clip(np.round(xx + vx).astype(int), 0, W - 1)
            y0 = np.floor(yf).astype(int)
            fy = yf - y0
            for off, ww in ((0, (1 - fy)), (1, fy)):
                yt = np.clip(y0 + off, 0, HF - 1)
                np.add.at(accY, (yt, xs), (w * ww) * Ys[k])
                np.add.at(accC.real, (yt, xs), (w * ww) * chis[k].real)
                np.add.at(accC.imag, (yt, xs), (w * ww) * chis[k].imag)
                np.add.at(accW, (yt, xs), w * ww)

    # fallback where coverage is thin: linear vertical interpolation
    # of the plain woven decode
    base = np.zeros((HF, W)); baseC = np.zeros((HF, W), complex)
    rows = (np.arange(HF) / scale - parities[j0]) / 2.0
    r0 = np.clip(np.floor(rows).astype(int), 0, L - 1)
    fr = np.clip(rows - r0, 0.0, 1.0)[:, None]
    r1 = np.minimum(r0 + 1, L - 1)
    for (dst, srcs) in ((base, (Ys[j0], Ys[j0 + 1])),
                        (baseC, (chis[j0], chis[j0 + 1]))):
        woven = np.zeros((2 * L, W), dst.dtype)
        woven[parities[j0]::2] = srcs[0]
        woven[parities[j0 + 1]::2] = srcs[1]
        fine_rows = np.arange(HF) / scale
        f0 = np.clip(np.floor(fine_rows).astype(int), 0, 2 * L - 2)
        ff = (fine_rows - f0)[:, None]
        dst[:] = woven[f0] * (1 - ff) + woven[f0 + 1] * ff

    lam = 0.35   # coverage confidence scale
    mix = accW / (accW + lam)
    Yf = np.where(accW > 0, accY / np.maximum(accW, 1e-9), 0.0)
    Cf = np.where(accW > 0, accC / np.maximum(accW + 0j, 1e-9), 0.0)
    Yout = mix * Yf + (1 - mix) * base
    Cout = mix * Cf + (1 - mix) * baseC
    return Yout, Cout


def psi_closed_form(S, phi, neighbors, chi_fallback):
    """N-step phase-shifting-interferometry init (optics import).

    Every static pixel is observed under several KNOWN carrier phases:
    its own field plus each MC-warped neighbor (phases measured via
    burst, never assumed). The model per observation k is

        S_k = Y + p·Re[c_k] − q·Im[c_k],      chi = p + iq

    i.e. exactly an N-step interferogram set. The weighted normal
    equations are a 3x3 per-pixel system solved in closed form
    (vectorised Cramer). This is the LINEAR skeleton of the temporal
    part of the variational energy: where the confidences are high
    (static, well-matched), it lands directly on the temporally exact
    solution before any spatial prior — a better linearisation point
    than the spectral init. Where total confidence is low, we blend
    back to the fallback (Dubois/holographic) init.

    Historical note: '3-phase decoding' papers (e.g. 3fsc-sampling
    demodulators, Philips WO1996013127) are the degenerate horizontal
    version of this idea, assuming (Y, chi) constant over 3 adjacent
    samples. Using FIELDS as the phase-shift axis removes that
    assumption for static content and lets motion gating handle the
    rest.
    """
    c0 = np.exp(1j * phi)
    obs = [(S, c0, np.ones_like(S))]
    # replicate the refine's robust temporal gate AT THE INIT POINT:
    # without it the closed form commits to motion-contaminated chi
    eps_t = 6.0
    for (S_w, c_w, conf) in neighbors:
        rt = (S_w - S) + np.real(chi_fallback * (c0 - c_w))
        obs.append((S_w, c_w, conf * eps_t ** 2 / (rt ** 2 + eps_t ** 2)))

    # accumulate weighted normal equations A^T W A (symmetric 3x3)
    a11 = a12 = a13 = a22 = a23 = a33 = 0.0
    b1 = b2 = b3 = 0.0
    for (Sk, ck, wk) in obs:
        rc, ic = np.real(ck), -np.imag(ck)      # basis [1, Re c, -Im c]
        a11 = a11 + wk
        a12 = a12 + wk * rc
        a13 = a13 + wk * ic
        a22 = a22 + wk * rc * rc
        a23 = a23 + wk * rc * ic
        a33 = a33 + wk * ic * ic
        b1 = b1 + wk * Sk
        b2 = b2 + wk * Sk * rc
        b3 = b3 + wk * Sk * ic

    det = (a11 * (a22 * a33 - a23 * a23)
           - a12 * (a12 * a33 - a23 * a13)
           + a13 * (a12 * a23 - a22 * a13))
    # phase diversity: with <3 well-separated phases the system is
    # near-singular (det -> 0); blend to fallback there
    ok = det > 1e-3 * np.maximum(a11, 1e-9) ** 3 * 0.01
    d = np.where(ok, det, 1.0)
    p = (b1 * (a22 * a33 - a23 * a23)
         - a12 * (b2 * a33 - a23 * b3)
         + a13 * (b2 * a23 - a22 * b3)) / d          # Y (unused here)
    q1 = (a11 * (b2 * a33 - b3 * a23)
          - b1 * (a12 * a33 - a23 * a13)
          + a13 * (a12 * b3 - b2 * a13)) / d         # p = Re chi
    q2 = (a11 * (a22 * b3 - a23 * b2)
          - a12 * (a12 * b3 - b2 * a13)
          + b1 * (a12 * a23 - a22 * a13)) / d        # q = Im chi
    chi_psi = (q1 + 1j * q2).astype(np.complex64)

    # confidence of the PSI solve: total neighbor weight (0 neighbors
    # => pure fallback) and diversity flag
    wsum = sum(w for (_, _, w) in obs[1:]) if neighbors else 0.0
    if np.isscalar(wsum):
        wsum = np.zeros_like(S)
    t = np.clip(wsum / 2.0, 0.0, 1.0) * ok
    return t * chi_psi + (1.0 - t) * chi_fallback


def yuv_to_rgb16(Y, U, V, black_ire=IRE_BLACK):
    """IRE-domain YUV -> 16-bit RGB (h, w, 3).

    black_ire: 7.5 for NTSC-M (setup pedestal), 0.0 for NTSC-J."""
    yn = (Y - black_ire) / (IRE_WHITE - black_ire)
    un = U / (IRE_WHITE - IRE_BLACK)
    vn = V / (IRE_WHITE - IRE_BLACK)
    yuv = np.stack([yn, un, vn], axis=-1)
    rgb = yuv @ YUV_TO_RGB.T
    return (np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def decode_frame(src: TbcSource, frame_index: int, cfg: DecoderConfig,
                 first_active_line: int | None = None, prev=None,
                 return_state: bool = False):
    """Decode one frame (two fields) -> RGB16 (H, W, 3).

    Default path weaves both fields into frame geometry *before*
    decoding, so the vertical priors operate at full frame resolution
    (per-field decoding lets the two fields drift apart at chroma
    transitions => interlace combing / hanging dots)."""
    (f0, _m0), (f1, _m1) = src.read_frame_fields(frame_index)
    p = src.params

    if not cfg.frame_decode:
        fal = (p.first_active_field_line if first_active_line is None
               else first_active_line)
        outs = []
        for fld in (f0, f1):
            Y, U, V = decode_field(fld, p, cfg)
            outs.append((Y[fal:], U[fal:], V[fal:]))
        lines = outs[0][0].shape[0]
        w = p.active_width
        Yf = np.zeros((2 * lines, w))
        Uf = np.zeros_like(Yf)
        Vf = np.zeros_like(Yf)
        Yf[0::2], Uf[0::2], Vf[0::2] = outs[0]
        Yf[1::2], Uf[1::2], Vf[1::2] = outs[1]
        rgb = yuv_to_rgb16(Yf, Uf, Vf)
        return (rgb, None) if return_state else rgb

    # ---- woven-frame path ------------------------------------------
    S, phi = prepare_frame(src, frame_index, first_active_line)
    Y0, chi0 = holographic_init(S, phi, p, cfg)
    if cfg.cg_iterations > 0 and not cfg.monochrome:
        mc = []
        if prev is not None and cfg.temporal_strength > 0.0:
            nbrs = prev if isinstance(prev, list) else [prev]
            mc = [motion_compensate_prev(nb, Y0, tile=cfg.mc_tile,
                                         search=cfg.mc_search)
                  for nb in nbrs]
        Y, chi = variational_refine(S, phi, Y0, chi0, cfg, neighbors=mc)
    else:
        Y, chi = Y0, chi0
    if cfg.monochrome:
        chi = np.zeros_like(chi)

    V = np.real(chi) * cfg.chroma_gain
    U = -np.imag(chi) * cfg.chroma_gain
    rgb = yuv_to_rgb16(Y, U, V)
    if return_state:
        return rgb, (Y, S, phi)
    return rgb


def prepare_field(src: TbcSource, field_index: int,
                  first_active_line: int | None = None):
    """One field's active area -> (S, phi) at field geometry.

    The active line range defaults to the .tbc.json metadata
    (firstActiveFieldLine / lastActiveFieldLine); an explicit
    first_active_line overrides it (tests)."""
    p = src.params
    fal = (p.first_active_field_line if first_active_line is None
           else first_active_line)
    lal = p.last_active_field_line or p.field_height
    a0, a1 = p.active_video_start, p.active_video_end
    ire = p.ire(src.read_field(field_index))
    theta = burst_lockin_phase(ire, p)[fal:lal]
    S = ire[fal:lal, a0:a1].astype(np.float32)
    x = np.arange(p.field_width)[a0:a1]
    phi = (theta[:, None] + (np.pi / 2.0) * x[None, :]).astype(np.float32)
    return S, phi


def prepare_frame(src: TbcSource, frame_index: int,
                  first_active_line: int | None = None):
    """Weave a frame's two fields into (S, phi) at frame geometry."""
    (f0, _m0), (f1, _m1) = src.read_frame_fields(frame_index)
    p = src.params
    fal = (p.first_active_field_line if first_active_line is None
           else first_active_line)
    a0, a1 = p.active_video_start, p.active_video_end
    S_list, th_list = [], []
    for fld in (f0, f1):
        ire = p.ire(fld)
        th_list.append(burst_lockin_phase(ire, p))
        S_list.append(ire[fal:, a0:a1])

    lines = S_list[0].shape[0]
    w = p.active_width
    S = np.zeros((2 * lines, w))
    theta = np.zeros(2 * lines)
    S[0::2], S[1::2] = S_list
    theta[0::2] = th_list[0][fal:]
    theta[1::2] = th_list[1][fal:]

    x = np.arange(p.field_width)[a0:a1]
    phi = theta[:, None] + (np.pi / 2.0) * x[None, :]
    return S, phi


def decode_sequence(src: TbcSource, start: int, length: int,
                    cfg: DecoderConfig, first_active_line: int | None = None):
    """Generator decoding frames [start, start+length).

    3D mode (cfg.temporal_strength > 0): every frame receives extra,
    motion-compensated data equations built from its temporal
    neighbors' *raw composites* and measured carrier phase maps. With
    cfg.bidirectional (default), both n-1 and n+1 contribute — their
    failure modes are complementary (scene cuts and occlusions break
    at most one side), and a static pixel gets three raw measurements.

    cfg.passes > 1 runs fixed-point refinement over the whole segment:
    each pass re-solves every frame using the neighbors' *refined*
    luma as the block-matching reference (better flow => better gating
    => better decode), Gauss-Seidel style — frames already re-solved
    in the current pass immediately benefit the next ones. The
    temporal data always comes from raw composites, never from
    neighbor estimates, so multiple passes cannot drift recursively.

    Because of the future-frame dependency this buffers the whole
    segment (~15 MB/frame); decode long discs in chunks with a couple
    of frames of overlap.
    """
    if cfg.temporal_strength <= 0.0 or cfg.cg_iterations <= 0:
        for i in range(start, start + length):
            yield i, decode_frame(src, i, cfg,
                                  first_active_line=first_active_line)
        return

    # ---- FIELD-BASED 3D pipeline, streamed in sliding windows -------
    # Real interlace: the two fields of a frame are 1/60 s apart. The
    # decode unit is therefore the FIELD; anything cross-field is a
    # robust, motion-gated *term*, never an assumption. Temporal
    # neighbors of field f:
    #   f±2  same parity, 1 frame apart  -> carrier flip 180 deg
    #   f±1  adjacent field, 1/60 s      -> carrier offset 3/4 cycle,
    #        |dc| = sqrt(2): non-degenerate, and it restores the woven
    #        vertical detail on static content through the optimiser
    # All offsets are measured per line via the burst, never assumed.
    #
    # Memory is bounded: frames are processed in windows of
    # cfg.chunk_frames with cfg.chunk_overlap frames of temporal
    # context on each side, and emitted as each window completes.
    p = src.params
    if cfg.extended_temporal:
        offs = [-3, -2, -1, 1, 2, 3] if cfg.bidirectional else [-3, -2, -1]
    else:
        offs = [-2, -1, 1, 2] if cfg.bidirectional else [-2, -1]

    def field_parity(f):
        """TRUE spatial parity of field f: 0 = top field. Read from the
        .tbc.json isFirstField metadata when available — assuming
        index%2 breaks on captures that start on a second field or
        have skipped fields (inverted weave, wrong half-line offsets
        everywhere)."""
        if 0 <= f < len(src.fields):
            return 0 if src.fields[f].is_first_field else 1
        return f % 2
    n_passes = max(1, cfg.passes)
    C = max(1, cfg.chunk_frames)
    OV = max(0, cfg.chunk_overlap)

    t0 = start
    while t0 < start + length:
        t1 = min(t0 + C, start + length)
        w0 = max(start, t0 - OV)
        w1 = min(start + length, t1 + OV)
        fidx = list(range(2 * w0, 2 * w1))
        SP = [prepare_field(src, f, first_active_line) for f in fidx]
        # self-calibration: measure the source's noise, scale the
        # robust gates accordingly (a fixed IRE gate is 5 sigma on a
        # clean disc but 1.4 sigma on a noisy one => over-gating
        # exactly where 3D helps most)
        import dataclasses as _dc
        sigma = float(np.median([estimate_noise_ire(S) for (S, _) in SP]))
        ccfg = _dc.replace(
            cfg,
            temporal_eps=(cfg.temporal_eps if cfg.temporal_eps > 0
                          else float(np.clip(7.0 * sigma, 4.0, 20.0))),
            nr_eps=(cfg.nr_eps if cfg.nr_eps > 0
                    else float(np.clip(3.0 * sigma, 3.0, 12.0))))
        inits = [holographic_init(S, phi, p, ccfg) for (S, phi) in SP]
        Ys = [Y.astype(np.float32) for (Y, _) in inits]
        chis = [chi.astype(np.complex64) for (_, chi) in inits]

        for _pass in range(n_passes):
            for j in range(len(fidx)):
                S, phi = SP[j]
                neighbors = []
                for o in offs:
                    k = j + o
                    if not (0 <= k < len(fidx)):
                        continue
                    state = (Ys[k], SP[k][0], SP[k][1])
                    # raw integer warp for the equations: EXACT for
                    # aligned static content; residual half-line bias
                    # on odd offsets is per-pixel gated. (An
                    # envelope-resampled variant was tried and
                    # rejected: its 1-line comb leaks vertical luma
                    # gradients into chroma and injects cross-colour
                    # on textured content — the artefact this decoder
                    # exists to remove.)
                    mo = estimate_motion(Ys[k], Ys[j],
                                         tile=ccfg.mc_tile,
                                         search=ccfg.mc_search)
                    S_w, c_w, conf_n = motion_compensate_prev(
                        state, Ys[j], tile=ccfg.mc_tile, motion=mo)
                    if ccfg.coherence_gate > 0.0:
                        # InSAR coherence between the current chroma
                        # and the warped neighbor chroma, floored so
                        # grey content (|chi|~0, gamma = pure noise)
                        # keeps the equation's LUMA benefit
                        row_off = (field_parity(fidx[j])
                                   - field_parity(fidx[k])) / 2.0
                        hh, ww = Ys[j].shape
                        Cw = (_warp_bilinear_tiles(
                                  chis[k].real, mo[0], mo[1], ccfg.mc_tile,
                                  row_offset=row_off, out_shape=(hh, ww))
                              + 1j * _warp_bilinear_tiles(
                                  chis[k].imag, mo[0], mo[1], ccfg.mc_tile,
                                  row_offset=row_off, out_shape=(hh, ww)))
                        g = complex_coherence(chis[j], Cw)
                        a = ccfg.coherence_gate
                        conf_n = conf_n * ((1.0 - a) + a * g)
                    neighbors.append((S_w, c_w, conf_n))
                if (_pass == 0 and ccfg.psi_init and neighbors):
                    chis[j] = psi_closed_form(S, phi, neighbors, chis[j])
                if _pass >= 1 and ccfg.nr_anchor > 0.0 and len(fidx) > 1:
                    anchor = synth_reference(j, Ys, chis, SP, ccfg,
                                             parities=[field_parity(f)
                                                       for f in fidx])
                    Ys[j], chis[j] = variational_refine_joint(
                        S, phi, Ys[j], chis[j], ccfg,
                        neighbors=neighbors, anchor=anchor)
                else:
                    Ys[j], chis[j] = variational_refine(
                        S, phi, Ys[j], chis[j], ccfg,
                        neighbors=neighbors)

        black = 0.0 if cfg.ntsc_j else 7.5
        if cfg.acc:
            # ACC: chroma gain from measured burst amplitude
            amps = []
            for f in fidx:
                ire = p.ire(src.read_field(f))
                amps.append(burst_amplitude_ire(ire, p))
            acc_gain = float(np.clip(20.0 / max(np.median(amps), 1.0),
                                     0.5, 2.0))
        else:
            acc_gain = 1.0

        if cfg.output_fidelity:
            # purist output: chroma keeps the fully guided separation,
            # luma reverts to exact data fidelity (no NR in deliverable)
            Yout = [SP[j][0] - np.real(chis[j] * np.exp(1j * SP[j][1]))
                    for j in range(len(fidx))]
        else:
            Yout = Ys
        parities = [field_parity(f) for f in fidx]
        for t in range(t0, t1):
            j0 = 2 * (t - w0)
            j1 = j0 + 1
            if cfg.drizzle:
                # drizzle is inherently a stacking output mode; it uses
                # the anchored (Ys) fields, not the purist ones
                Yd, Cd = drizzle_frame(j0, Ys, chis, parities, cfg)
                if cfg.monochrome:
                    Cd = np.zeros_like(Cd)
                g = cfg.chroma_gain * acc_gain
                yield t, yuv_to_rgb16(Yd, -np.imag(Cd) * g,
                                      np.real(Cd) * g, black_ire=black)
                continue
            lines = Yout[j0].shape[0]
            Yf = np.zeros((2 * lines, p.active_width))
            Cf = np.zeros((2 * lines, p.active_width), complex)
            p0, p1 = parities[j0], parities[j1]
            if p0 == p1:   # metadata anomaly: fall back to index order
                p0, p1 = 0, 1
            Yf[p0::2], Yf[p1::2] = Yout[j0], Yout[j1]
            Cf[p0::2], Cf[p1::2] = chis[j0], chis[j1]
            if cfg.monochrome:
                Cf = np.zeros_like(Cf)
            g = cfg.chroma_gain * acc_gain
            V = np.real(Cf) * g
            U = -np.imag(Cf) * g
            yield t, yuv_to_rgb16(Yf, U, V, black_ire=black)
        t0 = t1
