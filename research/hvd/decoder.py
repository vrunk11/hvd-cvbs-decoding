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
    fast: bool = False             # FAST MODE: same algorithm, cheaper
                                   # logistics — motion cache shared across
                                   # passes and the anchor blend; predicted+
                                   # verified ME for long offsets; tile-res
                                   # confidence maps; looser CG early-exit.
                                   # Target: >=3x speed, <=0.1 dB, no
                                   # perceptible change.
    cg_tol: float = 0.0            # CG relative gradient-norm early-exit
                                   # (0 = auto: 0.02 slow / 0.10 fast)
    diag_prior: float = 0.0        # oriented (+/-45 deg) chroma prior
                                   # weight, relative to mu_v and distance-
                                   # normalised by 1/2 (diagonal step =
                                   # sqrt(2) px); 0 disables
    trajectory_fit: bool = True    # fit ONE per-tile velocity across all
                                   # temporal offsets (they are 6 noisy
                                   # measurements of one physical motion)
                                   # and snap agreeing pairwise vectors to
                                   # the trajectory; disagreement is kept
                                   # (occlusion/acceleration is signal)
    psi_init: bool = False         # phase-shifting-interferometry init:
                                   # closed-form per-pixel weighted LS over
                                   # the >=3 carrier phases the neighbor
                                   # fields provide (optics N-step PSI);
                                   # temporally exact where static, blended
                                   # by total confidence
    symmetry_variant: bool = False # Transform-style spectral-symmetry
                                   # certified-chroma init variant
    lambda_c: float = 1.0      # chroma-smoothness vs luma-plausibility arbitration
    # mu_h = aniso * lambda_c. 0 = AUTO (default): the right split is a
    # property of the content's chroma orientation — thin horizontal
    # chroma bands / fsc-adjacent texture want the vertical prior weak
    # (-> 1.0: lines ~15% better, curtain ~17%), sharp horizontal chroma
    # transitions (colour bars) want 0.5 (a fixed 1.0 costs the chart
    # 1.1 dB in 2D). Auto measures the init's p90 |D chi0| gradient ratio
    # per solve and maps it into [0.5, 1.0] (_resolve_chroma_aniso).
    # Positive = forced fixed value (pre-auto reference behaviour at 0.5).
    chroma_aniso: float = 0.0  # (chroma is broader
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

# --------------------------------------------- motion compensation (3D)

def _box_blur(a, r=2):
    """Box blur via integral images (cumulative sums): O(n) fully
    vectorised, ~20x faster than per-row convolution and the natural
    formulation for a C++ port (two prefix-sum passes). Semantics
    match the original exactly: CONSTANT 1/(2r+1) normalisation per
    axis (zero-padded edges are attenuated, not renormalised), and
    default radius r=2 — the ME pre-blur depends on both."""
    h, w = a.shape[:2]
    c = np.zeros((h + 1, w), np.float64)
    np.cumsum(a, axis=0, out=c[1:])
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    sy = c[y1] - c[y0]
    c2 = np.zeros((h, w + 1), np.float64)
    np.cumsum(sy, axis=1, out=c2[:, 1:])
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    return (c2[:, x1] - c2[:, x0]) / float((2 * r + 1) ** 2)


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
    _, _, best0, cost0 = _bm_pass(A, B, tile, [0], [0])

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

    conf = _motion_conf(A, best, tile)
    return mdy + sdy, mdx + sdx, conf


def _motion_conf(A, best, tile):
    """Confidence layers shared by full and predicted ME: tile-energy
    explained, median-calibrated outlier rejection, global pair
    validity (scene-cut detector)."""
    h, w = A.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile

    def tilesum(D):
        Dp = np.zeros((th * tile, tw * tile))
        Dp[:h, :w] = D
        return Dp.reshape(th, tile, tw, tile).sum(axis=(1, 3))

    tmean = tilesum(A) / (tile * tile)
    tvar = tilesum(A ** 2) / (tile * tile) - tmean ** 2
    energy = (tvar + 1.0) * (tile * tile)
    conf = np.clip(1.0 - best / energy, 0.0, 1.0)
    r = best / (tile * tile)
    med = np.median(r)
    conf *= (med + 1.0) / (med + 1.0 + np.maximum(0.0, r - med))
    conf *= 6.0 / (6.0 + med)
    return conf


def verify_motion(Y_ref, Y_cur, pdy, pdx, tile=32):
    """FAST-mode ME for long temporal offsets: instead of a full
    pyramid search (~130 SSD evaluations), evaluate ONLY the
    trajectory-predicted vector and the zero vector (2 evaluations),
    keep the better, and compute the standard confidence layers. The
    trajectory supplies the hypothesis; this supplies the audit. Long
    offsets are exactly where full search is least reliable anyway
    (more occlusion), so the quality cost is imperceptible while the
    ME cost drops ~60x for those offsets."""
    A = _box_blur(Y_cur)
    B = _box_blur(Y_ref)
    pdyi = np.round(np.asarray(pdy)).astype(np.int32)
    pdxi = np.round(np.asarray(pdx)).astype(np.int32)
    mdy, mdx, best, _ = _bm_pass(A, B, tile, [0], [0],
                                 base_dy=pdyi, base_dx=pdxi)
    _, _, best0, _ = _bm_pass(A, B, tile, [0], [0])
    take0 = best0 <= best
    mdy = np.where(take0, 0, mdy).astype(float)
    mdx = np.where(take0, 0, mdx).astype(float)
    best = np.where(take0, best0, best)
    # keep the sub-pixel part of the prediction where it won
    mdy = np.where(take0, mdy, np.asarray(pdy))
    mdx = np.where(take0, mdx, np.asarray(pdx))
    return mdy, mdx, _motion_conf(A, best, tile)


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


# (mc_warp / envelope_of / motion_compensate_envelope removed in the
# cleanup pass: the envelope-resampled neighbour variant they implemented
# was tried and REJECTED — its 1-line comb leaks vertical luma gradients
# into chroma; see the raw-warp comment in decode_sequence and THEORY 9g.
# The negative result is recorded in prose; the bodies are gone.)


def _warp_bilinear_tiles(a, dyf, dxf, tile, row_offset=0.0, out_shape=None,
                         vpix=None):
    """Bilinear warp of a baseband array by per-tile float vectors,
    with a constant extra row offset (grid alignment). The output grid
    may differ from the source grid (envelope arrays have L-1 rows)."""
    ho, wo = out_shape if out_shape is not None else a.shape[:2]
    ho, wo = int(ho), int(wo)
    yy = np.arange(ho, dtype=np.float64)[:, None] * np.ones((1, wo))
    xx = np.ones((ho, 1)) * np.arange(wo, dtype=np.float64)[None, :]
    if vpix is None:
        vpix = _vectors_per_pixel(np.asarray(dyf, float),
                                  np.asarray(dxf, float), tile, (ho, wo))
    vdy, vdx = vpix
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


def motion_compensate_prev(prev, Y_cur_init, tile=32, search=16, motion=None, fast=False):
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
    if fast:
        # tile-resolution confidence, bilinearly interpolated between
        # tile centers (the full-res box blur only ever smoothed at
        # sub-tile scale; interpolating the 24x24 tile map is ~256x
        # cheaper and visually identical)
        cy, cx = _vectors_per_pixel(conf ** 2, conf ** 2, tile, (h, w))
        conf_px = cy
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        ty = np.minimum(yy // tile, conf.shape[0] - 1)
        tx = np.minimum(xx // tile, conf.shape[1] - 1)
        conf_px = _box_blur(conf[ty, tx] ** 2, r=8)
    S_w = warp_by_tiles(Sp, mdy, mdx, tile)
    phi_w = warp_by_tiles(phip, mdy, mdx, tile)
    return S_w, np.exp(1j * phi_w), conf_px


def synth_reference(j, Ys, chis, SP, cfg, parities=None,
                    motion_cache=None):
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
        if motion_cache is not None and (j, k) in motion_cache:
            mdy, mdx, cf = motion_cache[(j, k)]
        else:
            mdy, mdx, cf = estimate_motion(Ys[k], Ys[j], tile=tile,
                                           search=cfg.mc_search)
            if motion_cache is not None:
                motion_cache[(j, k)] = (mdy, mdx, cf)
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
        vpix = _vectors_per_pixel(np.asarray(mdy, float),
                                  np.asarray(mdx, float), tile, (hh, ww))
        Yw = _warp_bilinear_tiles(Ys[k], mdy, mdx, tile,
                                  row_offset=row_off, out_shape=(hh, ww),
                                  vpix=vpix)
        Cw = (_warp_bilinear_tiles(chis[k].real, mdy, mdx, tile,
                                   row_offset=row_off, out_shape=(hh, ww),
                                   vpix=vpix)
              + 1j * _warp_bilinear_tiles(chis[k].imag, mdy, mdx, tile,
                                          row_offset=row_off,
                                          out_shape=(hh, ww), vpix=vpix))
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


def _d1(a):
    """+45-degree diagonal forward difference (Neumann), for oriented
    priors (4D brainstorm idea #2: the (x,y,theta) dimension). Built
    so that _d1T is its exact adjoint (verified in run_tests)."""
    d = np.zeros_like(a)
    d[:-1, :-1] = a[1:, 1:] - a[:-1, :-1]
    return d


def _d1T(a):
    d = np.zeros_like(a)
    d[:-1, :-1] -= a[:-1, :-1]
    d[1:, 1:] += a[:-1, :-1]
    return d


def _d2(a):
    """-45-degree diagonal forward difference (Neumann)."""
    d = np.zeros_like(a)
    d[:-1, 1:] = a[1:, :-1] - a[:-1, 1:]
    return d


def _d2T(a):
    d = np.zeros_like(a)
    d[:-1, 1:] -= a[:-1, 1:]
    d[1:, :-1] += a[:-1, 1:]
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


def _resolve_chroma_aniso(chi0, cfg):
    """chroma_aniso == 0 => AUTO: dominant chroma-detail orientation of
    the init, via p98 one-step gradients (medians only see the flat
    background), mapped into [0.5, 1.0]. See HvdConfig.chroma_aniso for
    the measurements behind the map."""
    if cfg.chroma_aniso > 0.0:
        return cfg.chroma_aniso
    # Line-pair averages, not raw chi0: the init's dominant vertical
    # energy is cross-colour LEAK, alternating sign every line (carrier
    # flips 180 deg per field line) — on the colour-bar chart it made
    # |Dy chi0| read as large as the bar transitions (auto wrongly
    # picked ~1.0). Pair-averaging cancels the alternating leak exactly;
    # genuine chroma structure survives.
    _h2 = (chi0.shape[0] // 2) * 2
    _pav = 0.5 * (chi0[0:_h2:2] + chi0[1:_h2:2])
    gx = np.abs(np.diff(_pav[::2, ::3], axis=1)).ravel()
    gy = np.abs(np.diff(_pav[::2, 1::3], axis=0)).ravel()
    if gx.size == 0 or gy.size == 0:
        return 0.5
    # p98, not p90: structure can be SPARSE (colour-bar transitions are
    # ~4% of columns — a p90 reads noise/noise ~ 1 there and wrongly
    # picks aniso ~ 1, costing the chart 1.1 dB in 2D).
    px = float(np.percentile(gx, 98))
    py = float(np.percentile(gy, 98))
    r = py / max(px, 1e-6)
    import os as _os
    if _os.environ.get("HVD_DEBUG_ANISO"):
        print(f"    aniso auto: px={px:.2f} py={py:.2f} r={r:.2f} "
              f"-> {float(np.clip(0.5 + 1.1 * (r - 1.3), 0.5, 1.0)):.2f}")
    return float(np.clip(0.5 + 1.1 * (r - 1.3), 0.5, 1.0))

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
    _aniso = _resolve_chroma_aniso(chi0, cfg)
    mu_h = cfg.lambda_c * _aniso   # chroma is broader horizontally
    mu_v = cfg.lambda_c
    mu_d = cfg.lambda_c * cfg.diag_prior * 0.5
    # renormalise so the TOTAL chroma prior mass is unchanged by the
    # oriented terms: they redistribute smoothing across directions
    # (isotropy against diagonal cross-colour) without increasing it
    _tot0 = cfg.lambda_c * (1.0 + _aniso)
    _scale = _tot0 / (_tot0 + 2.0 * mu_d) if mu_d > 0 else 1.0
    mu_h, mu_v, mu_d = mu_h * _scale, mu_v * _scale, mu_d * _scale
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    carrier = np.exp(1j * phi)
    # per-neighbor: MC-warped composite, carrier difference, confidence
    nbr = [(S_w, carrier - c_w, conf) for (S_w, c_w, conf) in neighbors]
    chi = chi0.copy()

    _cg_total = (int(cfg.cg_iterations * 2 // 3) if cfg.fast
                 else cfg.cg_iterations)
    n_inner = max(1, _cg_total // max(1, irls_outer))

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
        cd1, cd2 = np.abs(_d1(chi)), np.abs(_d2(chi))
        wd1 = 1.0 / np.sqrt(1.0 + (cd1 / eps_c) ** 2)
        wd2 = 1.0 / np.sqrt(1.0 + (cd2 / eps_c) ** 2)
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
                           + mu_v * _dyT(wcy * _dy(chi))
                           + mu_d * (_d1T(wd1 * _d1(chi))
                                     + _d2T(wd2 * _d2(chi)))))
            for (S_w, dc, conf), wt in zip(nbr, wts):
                gC += (2.0 * nu * wt
                       * temporal_residual(chi, S_w, dc) * np.conj(dc))
            return gC

        def curv(dC):
            dY = -np.real(dC * carrier)
            h = (np.sum(wx * _dx(dY) ** 2) + np.sum(wy * _dy(dY) ** 2)
                 + mu_h * np.sum(wcx * np.abs(_dx(dC)) ** 2)
                 + mu_v * np.sum(wcy * np.abs(_dy(dC)) ** 2)
                 + mu_d * (np.sum(wd1 * np.abs(_d1(dC)) ** 2)
                           + np.sum(wd2 * np.abs(_d2(dC)) ** 2)))
            for (S_w, dc, conf), wt in zip(nbr, wts):
                h += nu * np.sum(wt * np.real(dC * dc) ** 2)
            return h

        g = grad(chi)
        d = -g
        gg = np.real(np.sum(np.conj(g) * g))
        tol = cfg.cg_tol or (0.10 if cfg.fast else 0.02)
        gg0 = gg
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
            if gg < tol * tol * gg0:
                break
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
    _aniso = _resolve_chroma_aniso(chi0, cfg)
    mu_h = cfg.lambda_c * _aniso
    mu_v = cfg.lambda_c
    mu_d = cfg.lambda_c * cfg.diag_prior * 0.5
    # renormalise so the TOTAL chroma prior mass is unchanged by the
    # oriented terms: they redistribute smoothing across directions
    # (isotropy against diagonal cross-colour) without increasing it
    _tot0 = cfg.lambda_c * (1.0 + _aniso)
    _scale = _tot0 / (_tot0 + 2.0 * mu_d) if mu_d > 0 else 1.0
    mu_h, mu_v, mu_d = mu_h * _scale, mu_v * _scale, mu_d * _scale
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    nu_a = cfg.nr_anchor if anchor is not None else 0.0
    if anchor is not None:
        Y_hat, chi_hat, w_a = anchor
    carrier = np.exp(1j * phi)
    nbr = [(S_w, c_w, conf) for (S_w, c_w, conf) in neighbors]

    Y = Y0.copy()
    chi = chi0.copy()
    _cg_total = (int(cfg.cg_iterations * 2 // 3) if cfg.fast
                 else cfg.cg_iterations)
    n_inner = max(1, _cg_total // max(1, irls_outer))

    for _outer in range(irls_outer):
        gxY, gyY = _dx(Y), _dy(Y)
        wx = eps / np.sqrt(gxY ** 2 + eps ** 2)
        wy = eps / np.sqrt(gyY ** 2 + eps ** 2)
        cd1, cd2 = np.abs(_d1(chi)), np.abs(_d2(chi))
        wd1 = 1.0 / np.sqrt(1.0 + (cd1 / eps_c) ** 2)
        wd2 = 1.0 / np.sqrt(1.0 + (cd2 / eps_c) ** 2)
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
                         + mu_v * _dyT(wcy * _dy(chi))
                         + mu_d * (_d1T(wd1 * _d1(chi))
                                   + _d2T(wd2 * _d2(chi))))
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
                  + mu_v * np.sum(wcy * np.abs(_dy(dC)) ** 2)
                  + mu_d * (np.sum(wd1 * np.abs(_d1(dC)) ** 2)
                            + np.sum(wd2 * np.abs(_d2(dC)) ** 2)))
            return h

        gY, gC = grad(Y, chi)
        dY, dC = -gY, -gC
        gg = np.sum(gY * gY) + np.real(np.sum(np.conj(gC) * gC))
        tol = cfg.cg_tol or (0.10 if cfg.fast else 0.02)
        gg0 = gg
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
            if gg < tol * tol * gg0:
                break
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
    # (the Y cofactor of the 3x3 solve is never needed — chi is what the
    # init wants and Y is re-derived from the identity downstream; the
    # full-array expansion it used to compute here was pure waste)
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
                    cfg: DecoderConfig, first_active_line: int | None = None,
                    roi=None):
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
        if roi is not None:
            # selective-3D support: run the ENTIRE field-based machinery
            # (motion, gates, solver, weave) on a rectangular crop, in
            # FIELD coordinates (fy0, fy1, x0, x1). Every downstream
            # step is shape-agnostic, so nothing else changes; emitted
            # frames are cropped to (2*(fy1-fy0), x1-x0).
            fy0, fy1, rx0, rx1 = roi
            SP = [(np.ascontiguousarray(S[fy0:fy1, rx0:rx1]),
                   np.ascontiguousarray(phi[fy0:fy1, rx0:rx1]))
                  for (S, phi) in SP]
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
        mcache = {}   # per-chunk motion cache: fast mode shares one
                      # estimate per field pair across passes AND with
                      # the anchor blend (identical inputs, ~4x fewer
                      # block-matching runs); slow mode keys per pass
        Ys = [Y.astype(np.float32) for (Y, _) in inits]
        chis = [chi.astype(np.complex64) for (_, chi) in inits]

        for _pass in range(n_passes):
            for j in range(len(fidx)):
                S, phi = SP[j]
                # --- trajectory-coherent motion (4D brainstorm) --
                # The offsets f±1..±3 are SIX independent pairwise
                # matches of what is physically ONE velocity per tile
                # (over 3/60 s). We therefore fit a per-tile velocity
                # v as the confidence-weighted median of d_k/k over
                # all offsets, then snap each pairwise vector to k·v
                # WHEN IT AGREES (<=1.5 px): six noisy measurements
                # collapse onto one trajectory ("the 1D filter that
                # follows the matter"), and disagreement is preserved
                # as-is — it is signal (occlusion, acceleration), not
                # noise, and the per-pixel gates handle it.
                pair_motion = {}
                full_os = ([-1, 1, 2] if ccfg.fast else offs)
                for o in offs:
                    k = j + o
                    if not (0 <= k < len(fidx)):
                        continue
                    ck = (j, k) if ccfg.fast else (j, k, _pass)
                    if ck in mcache:
                        pair_motion[o] = mcache[ck]
                    elif (o in full_os) or not ccfg.trajectory_fit:
                        pair_motion[o] = mcache[ck] = estimate_motion(
                            Ys[k], Ys[j], tile=ccfg.mc_tile,
                            search=ccfg.mc_search)
                if ccfg.fast and ccfg.trajectory_fit:
                    # FAST: long offsets get trajectory-PREDICTED
                    # vectors, audited by verify_motion (2 SSD evals)
                    have = sorted(o for o in pair_motion)
                    if have:
                        pj0 = field_parity(fidx[j])
                        vys = np.stack([(pair_motion[o][0]
                                         - (field_parity(fidx[j + o]) - pj0)
                                         / 2.0) / o for o in have])
                        vxs = np.stack([pair_motion[o][1] / o
                                        for o in have])
                        vym0 = np.median(vys, axis=0)
                        vxm0 = np.median(vxs, axis=0)
                        for o in offs:
                            k = j + o
                            if o in pair_motion or not (0 <= k < len(fidx)):
                                continue
                            ck = (j, k)
                            if ck in mcache:
                                pair_motion[o] = mcache[ck]
                                continue
                            h_o = (field_parity(fidx[k]) - pj0) / 2.0
                            pair_motion[o] = mcache[ck] = verify_motion(
                                Ys[k], Ys[j], o * vym0 + h_o, o * vxm0,
                                tile=ccfg.mc_tile)
                if len(pair_motion) >= 2 and ccfg.trajectory_fit:
                    os_ = sorted(pair_motion)
                    # odd offsets measure content displacement PLUS the
                    # half-line parity geometry: d_k = k*v + h_k with
                    # h_k = (p_k - p_j)/2. Remove the known h_k before
                    # fitting the trajectory, reinstate it after —
                    # otherwise the fit mixes two biased populations.
                    pj = field_parity(fidx[j])
                    # static content of parity p_k matched from parity
                    # p_j appears at dy = (p_k - p_j)/2 (derivation in
                    # THEORY 9e) — the sign matters, the first attempt
                    # doubled the bias instead of removing it
                    hk = {o: (field_parity(fidx[j + o]) - pj) / 2.0
                          for o in os_}
                    vy = np.stack([(pair_motion[o][0] - hk[o]) / o
                                   for o in os_])
                    vx = np.stack([pair_motion[o][1] / o for o in os_])
                    wv = np.stack([pair_motion[o][2] for o in os_])
                    # weighted median via argsort would be heavy; the
                    # plain median over <=6 samples is robust enough,
                    # with low-confidence samples pushed toward the
                    # median of the rest by NaN-masking
                    vy = np.where(wv > 0.15, vy, np.nan)
                    vx = np.where(wv > 0.15, vx, np.nan)
                    with np.errstate(all="ignore"):
                        vym = np.nanmedian(vy, axis=0)
                        vxm = np.nanmedian(vx, axis=0)
                    vym = np.nan_to_num(vym)
                    vxm = np.nan_to_num(vxm)
                    # consensus requirement: the snap only applies where
                    # >= 3 offsets agree with the fitted trajectory —
                    # structured motion qualifies, noise-dominated
                    # matching leaves the pairwise vectors untouched
                    agree = 0
                    for o in os_:
                        mdy, mdx, _ = pair_motion[o]
                        agree = agree + ((np.abs(mdy - (o * vym + hk[o]))
                                          + np.abs(mdx - o * vxm)) <= 1.5)
                    consensus = agree >= 3
                    for o in os_:
                        mdy, mdx, cf = pair_motion[o]
                        py, px = o * vym + hk[o], o * vxm
                        ok = consensus & (
                            (np.abs(mdy - py) + np.abs(mdx - px)) <= 1.5)
                        pair_motion[o] = (np.where(ok, py, mdy),
                                          np.where(ok, px, mdx), cf)

                neighbors = []
                # Half-line validity envelope for ODD offsets (opposite
                # parity) — closes a hole the C++ port measured (THEORY
                # 9h): on detail at/beyond the opposite parity's vertical
                # Nyquist (a 1-frame-line coloured ledge, thin blinds) the
                # feature is INVISIBLE to that parity, whose equations
                # then vote "background"; their residual OSCILLATES with x
                # (|dchi*cos|), so the robust weight lets confident wrong
                # votes through at the cosine zeros — measured, 3D made
                # 1-frame-line chroma detail WORSE than 2D. Only the
                # phase-independent BASEBAND envelope catches all the
                # votes (a composite-based envelope is unusable: within a
                # field the carrier flips 180 deg per line, so |dS/dline|
                # ~ 2|chi| on flat saturated colour — measured -8.6 dB on
                # the chart). One-sided max diff (a central diff is
                # exactly zero ON a one-row feature), horizontal-only
                # smoothing (a 2D blur dilutes a one-row footprint ~2.5x),
                # and a 0.25 FLOOR: on step-edge-heavy content the odd
                # equations are biased but informative and hard-gating
                # them cost 4.4 dB of 3D gain on the chart.
                _dyu = np.abs(np.diff(Ys[j], axis=0, prepend=Ys[j][:1]))
                _dyd = np.abs(np.diff(Ys[j], axis=0, append=Ys[j][-1:]))
                _dcu = np.abs(np.diff(chis[j], axis=0,
                                      prepend=chis[j][:1]))
                _dcd = np.abs(np.diff(chis[j], axis=0,
                                      append=chis[j][-1:]))
                _m = np.maximum(_dyu ** 2 + _dcu ** 2,
                                _dyd ** 2 + _dcd ** 2)
                _hh, _ww = _m.shape
                _c = np.zeros((_hh, _ww + 1))
                np.cumsum(_m, axis=1, out=_c[:, 1:])
                _x0 = np.clip(np.arange(_ww) - 2, 0, _ww)
                _x1 = np.clip(np.arange(_ww) + 3, 0, _ww)
                _vg = np.sqrt((_c[:, _x1] - _c[:, _x0])
                              / np.maximum(_x1 - _x0, 1))
                _og = np.maximum(
                    0.35,
                    ccfg.temporal_eps ** 2
                    / (ccfg.temporal_eps ** 2 + _vg * _vg))
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
                    mo = pair_motion[o]
                    S_w, c_w, conf_n = motion_compensate_prev(
                        state, Ys[j], tile=ccfg.mc_tile, motion=mo,
                        fast=ccfg.fast)
                    if ccfg.coherence_gate > 0.0 and not (
                            ccfg.fast and _pass == 0):
                        # fast: coherence only from pass 1 — at pass 0
                        # the chi fields are still inits and the
                        # measurement is barely informative
                        # InSAR coherence between the current chroma
                        # and the warped neighbor chroma, floored so
                        # grey content (|chi|~0, gamma = pure noise)
                        # keeps the equation's LUMA benefit
                        # AUDIT FIX (THEORY 9g): the warp convention
                        # assumes motion WITHOUT the half-line parity
                        # component (row_offset supplies it) — true while
                        # the margin rule zeroed static estimates, broken
                        # once the trajectory snap deliberately
                        # RE-INTRODUCED h_k into the pairwise vectors:
                        # adding row_offset on top double-compensated, and
                        # the gate compared chi fields shifted a FULL line
                        # on exactly the static consensus tiles it exists
                        # to protect. The snapped/subpixel motion already
                        # carries the total inter-grid displacement, so
                        # the correct warp is sy = y - dy, no row offset.
                        row_off = 0.0
                        hh, ww = Ys[j].shape
                        vpx = _vectors_per_pixel(
                            np.asarray(mo[0], float),
                            np.asarray(mo[1], float),
                            ccfg.mc_tile, (hh, ww))
                        Cw = (_warp_bilinear_tiles(
                                  chis[k].real, mo[0], mo[1], ccfg.mc_tile,
                                  row_offset=row_off, out_shape=(hh, ww),
                                  vpix=vpx)
                              + 1j * _warp_bilinear_tiles(
                                  chis[k].imag, mo[0], mo[1], ccfg.mc_tile,
                                  row_offset=row_off, out_shape=(hh, ww),
                                  vpix=vpx))
                        g = complex_coherence(chis[j], Cw)
                        a = ccfg.coherence_gate
                        conf_n = conf_n * ((1.0 - a) + a * g)
                    if (o % 2) != 0:
                        conf_n = conf_n * _og
                    neighbors.append((S_w, c_w, conf_n))
                if (_pass == 0 and ccfg.psi_init and neighbors):
                    chis[j] = psi_closed_form(S, phi, neighbors, chis[j])
                if _pass >= 1 and ccfg.nr_anchor > 0.0 and len(fidx) > 1:
                    anchor = synth_reference(
                        j, Ys, chis, SP, ccfg,
                        parities=[field_parity(f) for f in fidx],
                        motion_cache=(mcache if ccfg.fast else None))
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
            wcols = Yout[j0].shape[1]   # = active_width, or roi width
            Yf = np.zeros((2 * lines, wcols))
            Cf = np.zeros((2 * lines, wcols), complex)
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


def decode_sequence_selective(src: TbcSource, start: int, length: int,
                              cfg: DecoderConfig,
                              first_active_line: int | None = None,
                              max_area: float = 0.6, tile: int = 32,
                              halo_px: int = 48, stats: dict | None = None):
    """Selective 3D: full-frame 2D decode everywhere + the COMPLETE
    field-based 3D machinery (decode_sequence, cropped via roi=) on the
    single bounding box covering the most Y/C-ambiguous tiles, feather-
    blended in. Falls back to plain 2D when the flagged area exceeds
    max_area (a crop that big saves nothing) or to full 3D when
    max_area >= 1.

    The ambiguity score is the one validated against measured per-tile
    2D-vs-3D gain on real photo content (PORTING.md Sec. 19):
    sqrt(luma_HF_energy * chroma_HF_energy) of the holographic init,
    r=0.60 with true gain. The same measurement showed the gain is
    DIFFUSE (top 30% of tiles carry ~48% of it), so this mode is an
    explicit speed/quality dial, not a free lunch: expect roughly half
    the 3D improvement at a fraction of the 3D cost inside the budget
    you give it via max_area. Halo must cover mc_search + one tile so
    motion estimation inside the crop sees the same evidence it would
    full-frame."""
    if cfg.temporal_strength <= 0.0 or cfg.cg_iterations <= 0:
        for i in range(start, start + length):
            yield i, decode_frame(src, i, cfg,
                                  first_active_line=first_active_line)
        return
    if max_area >= 1.0:
        yield from decode_sequence(src, start, length, cfg,
                                   first_active_line)
        return

    # ---- ambiguity from the first frame's init (cheap, no solve) ----
    S, phi = prepare_frame(src, start, first_active_line)
    p = src.params
    Y0, chi0 = holographic_init(S, phi, p, cfg)
    boxes = _ambiguous_boxes(Y0, chi0, tile=tile, max_area=max_area,
                             halo_px=halo_px)
    if stats is not None:
        stats["boxes"] = boxes
    if boxes == "full":
        # ambiguity present but too widespread to crop profitably:
        # fall back to FULL 3D, never to 2D. Selective mode's contract
        # is "at least full-3D quality, cheaper when possible" -- the
        # earlier 2D fallback here is exactly how real footage with a
        # border stripe ended up with its artifact zones untreated.
        yield from decode_sequence(src, start, length, cfg,
                                   first_active_line)
        return
    if not boxes:
        for i in range(start, start + length):
            yield i, decode_frame(src, i, cfg,
                                  first_active_line=first_active_line)
        return

    seqs = []
    for (y0, y1, x0, x1) in boxes:            # frame coords, y even
        roi = (y0 // 2, (y1 + 1) // 2, x0, x1)
        seqs.append(decode_sequence(src, start, length, cfg,
                                    first_active_line, roi=roi))
    masks = [None] * len(boxes)
    for i in range(start, start + length):
        rgb2d = decode_frame(src, i, cfg,
                             first_active_line=first_active_line)
        out = rgb2d.astype(np.float64)
        for bi, (y0, y1, x0, x1) in enumerate(boxes):
            _, crop3d = next(seqs[bi])
            ch = crop3d.shape[0]
            yy1 = min(y0 + ch, out.shape[0])
            if masks[bi] is None:
                hh = yy1 - y0
                ww = min(x1, out.shape[1]) - x0
                hy = min(halo_px, hh // 3)
                hx = min(halo_px, ww // 3)
                ry = np.ones(hh)
                ry[:hy] = np.linspace(0.0, 1.0, hy)
                ry[hh - hy:] = np.linspace(1.0, 0.0, hy)
                rx = np.ones(ww)
                rx[:hx] = np.linspace(0.0, 1.0, hx)
                rx[ww - hx:] = np.linspace(1.0, 0.0, hx)
                masks[bi] = np.outer(ry, rx)[..., None]
            m = masks[bi]
            sl = (slice(y0, y0 + m.shape[0]), slice(x0, x0 + m.shape[1]))
            out[sl] = (out[sl] * (1.0 - m)
                       + crop3d[:m.shape[0], :m.shape[1]].astype(np.float64)
                       * m)
        yield i, np.clip(out + 0.5, 0, 65535).astype(np.uint16)


def _ambiguity_flags(Y0, chi0, tile=32):
    """Shared detector core: per-tile ambiguity score (sqrt(luma_HF x
    chroma_HF), the §19-validated proxy), 4x-median threshold, >=2-of-8
    neighbor density filter, border-normalised smoothing. Returns
    (flagged tile map or None, h, w)."""
    h, w = Y0.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile
    ph, pw = th * tile, tw * tile

    def tmean(a):
        ap = np.zeros((ph, pw))
        ap[:h, :w] = a
        return ap.reshape(th, tile, tw, tile).mean(axis=(1, 3))

    k = max(3, tile // 2)
    c = np.ones(k) / k

    def _box2d(a):
        # count-normalised borders: plain 'same' convolution zero-pads,
        # manufacturing fake HF along row 0 / col 0 (measured: whole
        # first row+column of tiles flagged on a flat scene).
        ones = np.ones(a.shape[0])
        ny = np.convolve(ones, c, mode="same")
        t1 = np.apply_along_axis(
            lambda r: np.convolve(r, c, mode="same"), 0, a) / ny[:, None]
        ones = np.ones(a.shape[1])
        nx = np.convolve(ones, c, mode="same")
        return np.apply_along_axis(
            lambda r: np.convolve(r, c, mode="same"), 1, t1) / nx[None, :]

    lpY = _box2d(Y0)
    mag = np.abs(chi0).astype(np.float64)
    lpC = _box2d(mag)
    score = np.sqrt(np.maximum(tmean((Y0 - lpY) ** 2), 0.0)
                    * np.maximum(tmean((mag - lpC) ** 2), 0.0))
    med = float(np.median(score))
    thr = 4.0 * med
    flagged = score >= thr
    # density: keep tiles with >= 2 of 8 flagged neighbors (noise flags
    # isolated tiles; real ambiguous content flags clusters)
    nb = np.zeros_like(score)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sh = np.roll(np.roll(flagged, dy, axis=0), dx, axis=1)
            if dy == -1:
                sh[-1, :] = False
            elif dy == 1:
                sh[0, :] = False
            if dx == -1:
                sh[:, -1] = False
            elif dx == 1:
                sh[:, 0] = False
            nb += sh
    flagged = flagged & (nb >= 2)
    # exclude the outer 1-tile border ring: border tiles straddle the
    # capture-mask transition (black pillars, head-switch area) and are
    # never useful ROI targets themselves -- the boxes' halo still
    # covers in-picture artifacts near the border. A flagged border
    # stripe otherwise bridges or widens every band (measured on real
    # capture: left pillar edge).
    flagged[0, :] = False
    flagged[-1, :] = False
    flagged[:, 0] = False
    flagged[:, -1] = False
    if not flagged.any():
        # No localized peaks. Disambiguate uniform-CLEAN from uniform-
        # AMBIGUOUS with an absolute level: measured medians (IRE^2,
        # tile 16, field geometry) -- SMPTE chart 0.62, flat noise
        # sigma=2.5 1.00, real LD footage with frame-wide blinds 5.66,
        # diffuse photo texture 8.87. Above 3.0 the frame is globally
        # ambiguous and deserves FULL 3D, not the 2D fallback; a purely
        # relative threshold cannot tell these two apart.
        if med > 3.0:
            return "uniform", h, w
        return None, h, w
    return flagged, h, w


def _ambiguous_boxes(Y0, chi0, tile=32, max_area=0.6, halo_px=48,
                     halo_y=32, max_boxes=4):
    """Row-banded boxes: one per vertical band of flagged tile-rows,
    instead of a single hull. The single-hull version silently fell
    back to 2D on exactly the field-reported case it was built for --
    thin horizontal artifact zones at the TOP, MIDDLE and BOTTOM of the
    frame: their common hull is ~the whole frame, > max_area, box=None,
    and "selective 3D" quietly became plain 2D while full 3D fixed the
    zones. Bands merge to at most max_boxes; max_area limits the TOTAL
    cropped area."""
    flagged, h, w = _ambiguity_flags(Y0, chi0, tile)
    if flagged is None:
        return None
    if isinstance(flagged, str):     # "uniform": frame-wide ambiguity
        return "full"
    # A tile-row only counts as a band row with >= 2 flagged tiles: a
    # one-tile-wide vertical stripe (measured on real capture: the hard
    # edge of the black side pillar flags column 0 top to bottom)
    # otherwise BRIDGES every artifact band into one giant one that
    # overflows the area cap.
    rows = np.nonzero(flagged.sum(axis=1) >= 2)[0]
    if rows.size == 0:
        return None
    bands = []
    r0 = prev = rows[0]
    for r in rows[1:]:
        if r - prev <= 1:
            prev = r
            continue
        bands.append((r0, prev))
        r0 = prev = r
    bands.append((r0, prev))
    while len(bands) > max_boxes:
        gaps = [bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1)]
        i = int(np.argmin(gaps))
        bands[i] = (bands[i][0], bands[i + 1][1])
        del bands[i + 1]
    boxes = []
    total = 0
    for (tr0, tr1) in bands:
        cols = np.nonzero(flagged[tr0:tr1 + 1].any(axis=0))[0]
        # One box PER contiguous column run (gap <= 2), single-tile runs
        # dropped. History matters here: v18's min..max span let one
        # stray tile widen a band to full width; the first fix kept only
        # the LONGEST run, which on the real capture cut off the right
        # half of the circled sill artifact (its band held runs at
        # ~x256-576 and ~x560-720; only the first survived) -- the user-
        # visible symptom was "selective precisely avoids my artifacts".
        # Emitting every multi-tile run keeps all artifact clusters and
        # still drops isolated strays.
        runs = []
        c0 = cprev = cols[0]
        for cc in cols[1:]:
            if cc - cprev <= 2:
                cprev = cc
                continue
            runs.append((c0, cprev))
            c0 = cprev = cc
        runs.append((c0, cprev))
        # vertical halo only needs to cover motion search (mc_search=16)
        # plus margin: 48 tripled a one-tile band's height and pushed the
        # dispersed three-band case past the area cap.
        y0 = max(0, tr0 * tile - halo_y)
        y1 = min(h, (tr1 + 1) * tile + halo_y)
        y0 -= y0 % 2
        band_boxes = []
        for (c0, c1) in runs:
            if c1 - c0 < 1:      # single-tile run: stray, skip
                continue
            x0 = max(0, c0 * tile - halo_px)
            x1 = min(w, (c1 + 1) * tile + halo_px)
            # halo expansion can make neighbouring runs overlap; merge
            # so the blend isn't applied twice over the same pixels
            if band_boxes and x0 <= band_boxes[-1][3]:
                band_boxes[-1] = (y0, y1, band_boxes[-1][2], x1)
            else:
                band_boxes.append((y0, y1, x0, x1))
        for b in band_boxes:
            boxes.append(b)
            total += (b[1] - b[0]) * (b[3] - b[2])
    if not boxes:
        return None
    while len(boxes) > 2 * max_boxes:
        # merge the horizontally-nearest pair within the same band
        best = None
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] != boxes[j][0]:
                    continue
                gap = max(boxes[j][2] - boxes[i][3],
                          boxes[i][2] - boxes[j][3])
                if best is None or gap < best[0]:
                    best = (gap, i, j)
        if best is None:
            break
        _, i, j = best
        bi, bj = boxes[i], boxes[j]
        merged = (bi[0], bi[1], min(bi[2], bj[2]), max(bi[3], bj[3]))
        total += ((merged[1] - merged[0]) * (merged[3] - merged[2])
                  - (bi[1] - bi[0]) * (bi[3] - bi[2])
                  - (bj[1] - bj[0]) * (bj[3] - bj[2]))
        boxes[i] = merged
        del boxes[j]
    if total > max_area * h * w:
        return "full"
    return boxes
