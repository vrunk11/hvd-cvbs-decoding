"""
hvd_pal.decoder — Holographic-Variational PAL chroma decoder (research).

Architecture summary (full derivations in THEORY-PAL.md):

1.  THE PAL HOLOGRAM IDENTITY.  A 4fsc PAL field is an off-axis
    hologram recorded with a reference wave whose sense is CONJUGATED
    on alternate lines (the V-switch):

        S = Y + Re[ chi * c ],   chi = V - i*U   (one global phasor)
        c(x,l) =  exp(+i*phi)   on unswitched lines  (s = +1)
               = -exp(-i*phi)   on switched lines    (s = -1)

    Folding the V-switch into an EFFECTIVE CARRIER of unit modulus is
    the whole architectural trick: every downstream equation of the
    NTSC decoder (init, arbitration, gradients) only ever touches the
    carrier through Re[chi*c] and conj(c), so the identical variational
    machinery applies with c generalised from exp(i*phi).

2.  In optics language the V-switch is 2-step PHASE-CONJUGATE
    holography: alternate lines record the hologram with the conjugate
    reference. Combining a line pair cancels the twin image to first
    order — which is precisely why the analogue delay-line PAL decoder
    turns differential phase (hue) errors into small saturation
    errors. Here that cancellation is not an averaging circuit but a
    property the vertical chroma prior recovers automatically
    (delay-line PAL = the quadratic, confidence-free limit of this
    decoder's vertical prior).

3.  SWINGING-BURST JOINT ESTIMATION. The PAL burst swings +/-45 deg
    about -U line to line. A lock-in on the burst therefore measures
    BOTH the carrier phase (mean direction) and the V-switch parity
    (sign of the swing) in one complex number per line — a joint
    phase+polarisation estimator (polarimetry analogy). Parity is
    measured, never derived from line indices or trusted metadata.

4.  Same arbitration principle as NTSC: Y is eliminated by
    substitution (Y := S - Re[chi*c]) and the energy is minimised over
    chi alone, IRLS + CG, luma/chroma Charbonnier priors, parallel-
    level-sets structure coupling. PAL specifics: Hanover bars are a
    line-alternating conjugation error in chi -> maximal vertical
    gradient -> maximally penalised by the vertical prior -> the
    optimiser suppresses them structurally.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

from .tbc import TbcSource, VideoParameters

YUV_TO_RGB = np.array(
    [[1.0, 0.0, 1.13983],
     [1.0, -0.39465, -0.58060],
     [1.0, 2.03211, 0.0]]
)

IRE_BLACK = 0.0        # PAL: no setup
IRE_WHITE = 100.0
NOMINAL_BURST_IRE = 21.43

LINE_ADV = 3.0 * np.pi / 2.0   # 270 deg/line model on the stored grid


@dataclass
class DecoderConfig:
    lambda_c: float = 1.0          # chroma-smoothness vs luma-plausibility
    chroma_aniso: float = 0.0      # mu_h = aniso * lambda_c. 0 = AUTO:
                                   # measured from the init's dominant
                                   # chroma-detail orientation, PAL-adapted
                                   # (see _resolve_chroma_aniso). Positive =
                                   # forced. PAL chroma is ~1.3 MHz on both
                                   # axes (no NTSC I/Q asymmetry) so the
                                   # auto map centres nearer isotropy than
                                   # NTSC's 0.5.
    charbonnier_eps: float = 0.5   # luma edge-preservation scale (IRE)
    chroma_eps: float = 1.0        # chroma edge-preservation scale (IRE)
    structure_coupling: float = 0.25
    cg_iterations: int = 60        # 0 => pure holographic init
    cg_tol: float = 0.02
    init_lpf_h_mhz: float = 1.3    # PAL chroma bandwidth
    init_lpf_v_cph: float = 60.0
    frame_decode: bool = True      # weave fields, solve at frame geometry
                                   # (2D path; the 3D path decodes FIELDS
                                   # and weaves after, as the NTSC decoder)
    acc: bool = True               # Automatic Colour Control from burst
    chroma_gain: float = 1.0
    monochrome: bool = False
    symmetry_variant: bool = False # Transform-PAL certified-chroma init
                                   # variant (native on PAL, unlike NTSC)
    # ---- 3D (temporal equations), same-parity offsets only ----------
    temporal_strength: float = 0.0 # nu; 0 = 2D. Recommended 3D value 0.5
                                   # (see coherence_gate sweep note)
    temporal_eps: float = 0.0      # motion gate scale (IRE); 0 = auto
                                   # (~7x measured noise, as NTSC)
    temporal_offsets: tuple = (-4, -3, -2, -1, 1, 2, 3, 4)
                                   # FULL neighborhood to the f+/-8 horizon's
                                   # useful part. Even offsets are same-parity
                                   # (f+/-2: 270 deg |dc|=sqrt2; f+/-4:
                                   # 180 deg |dc|=2, the 2-frame PAL comb).
                                   # ODD offsets are cross-parity AND
                                   # cross-V-sense: the PAL derivation
                                   # (THEORY 5c) shows their |dc| OSCILLATES
                                   # with x between 0 and 2 (mean-square 2,
                                   # same as a constant sqrt2) — the
                                   # arbitration handles this exactly
                                   # (per-pixel inertness, never corruption)
                                   # and the half-line vertical bias is
                                   # gated per pixel (NTSC 9h gate, ported).
    trajectory_fit: bool = True    # fit ONE per-tile velocity across all
                                   # offsets (up to 8 pairwise matches of
                                   # one physical motion), remove the known
                                   # half-line term h_k before fitting,
                                   # snap agreeing vectors under consensus
                                   # (>= 3 offsets within 1.5 px)
    drizzle: bool = False          # astronomy-style vertical 2x stacking
                                   # output (PAL's 576 active lines alias
                                   # vertically exactly like NTSC's 480;
                                   # horizontal SR stays pointless at 4fsc)
    chunk_frames: int = 6          # streaming window (frames)
    chunk_overlap: int = 2         # temporal context overlap (frames)
    # ---- re-encode anchor loop (decode -> NR -> re-encode) ----------
    passes: int = 1                # >= 2 enables the anchor loop: pass 1
                                   # decodes with exact fidelity, then a
                                   # motion-compensated robust temporal
                                   # blend of the DECODES is re-encoded to
                                   # composite for an honesty check, and
                                   # pass 2+ re-solves jointly over (Y,chi)
                                   # softly anchored to the blend
    nr_anchor: float = 1.0         # anchor strength (nu_a)
    nr_eps: float = 0.0            # blend/re-encode robustness (IRE);
                                   # 0 = auto (~3x measured noise)
    nr_radius: int = 4             # temporal blend radius in FIELDS.
                                   # PAL default 4 (not NTSC's 2): the
                                   # separation-leakage anti-correlation
                                   # lives at the 180 deg carrier
                                   # relation, which PAL reaches at
                                   # f+/-4 (2 frames), not f+/-2
    output_fidelity: bool = True   # DEFAULT purist: re-impose
                                   # Y = S - Re[chi c] on output; the NR
                                   # machinery then only GUIDES the Y/C
                                   # separation — zero temporal smoothing
                                   # in the deliverable (--soft-output
                                   # opts out)
    coherence_gate: float = 1.0    # InSAR complex-coherence gating weight
                                   # (swept on the static/pan bench pair:
                                   # 1.0 with nu=0.5 makes 3D >= 2D on BOTH
                                   # — pan +0.3/+0.9 dB, static +1.8/+2.7;
                                   # weaker gates trade pan losses for
                                   # small static luma gains — recorded in
                                   # THEORY 5b)
    mc_tile: int = 32
    mc_search: int = 16


# ------------------------------------------------------- self-calibration

def estimate_noise_ire(S: np.ndarray) -> float:
    """Stride-4 horizontal second difference: 360 deg of carrier over 4
    samples cancels chroma AND smooth luma exactly (V-switch is a
    per-line constant, so the identity holds on PAL unchanged)."""
    d = S[:, :-8] - 2.0 * S[:, 4:-4] + S[:, 8:]
    q = np.percentile(np.abs(d - np.median(d)), 25)
    return float(q / 0.3186 / np.sqrt(6.0))


# ---------------------------------------- swinging-burst joint lock-in

def _tridiag_smooth(d, a, lam):
    """(diag(a) + lam*L) x = a*d, Thomas algorithm (as NTSC)."""
    n = len(d)
    diag = a + 2.0 * lam
    diag[0] -= lam
    diag[-1] -= lam
    off = np.full(n - 1, -lam)
    rhs = a * d
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


def _wrap(a):
    return np.angle(np.exp(1j * a))


def burst_lockin(field_ire: np.ndarray, p: VideoParameters):
    """Joint carrier-phase + V-switch-parity estimation from the
    swinging burst.

    Lock-in output per line:  z_l ~ (B/2) * exp(i*(theta_l + pi/2 - s_l*pi/4))

    i.e. the measured angle swings -/+45 deg about theta+90 for
    s = +1 / -1.  One complex number per line carries both unknowns:
    the mean direction is the carrier phase trajectory, the sign of
    the alternating swing is the V-switch parity (measured, not
    assumed — Bruch-sequence metadata is never trusted).

    Returns (theta[l], s[l], amp[l]).  theta is trajectory-smoothed
    around the 270 deg/line model (tridiagonal weighted solve + one
    IRLS Huber pass, as the NTSC decoder), which also absorbs the
    genuine slow +0.57 deg/line drift of the 25 Hz offset component.
    """
    h, _ = field_ire.shape
    x = np.arange(p.colour_burst_start, p.colour_burst_end)
    ref = np.exp(-1j * (np.pi / 2.0) * x)
    seg = field_ire[:, p.colour_burst_start:p.colour_burst_end]
    seg = seg - seg.mean(axis=1, keepdims=True)
    z = (seg * ref[None, :]).mean(axis=1)
    amp = np.abs(z)
    a_meas = np.angle(z)

    lines = np.arange(h)
    good = amp > (amp.max() * 0.2 if amp.max() > 0 else 1.0)
    if not good.any():
        theta = LINE_ADV * lines
        s = np.where(lines % 2 == 0, 1.0, -1.0)
        return theta, s, amp

    idx = np.where(good)[0]
    ref_line = idx[0]

    # Two parity hypotheses for the reference line; score each by how
    # well the alternating +/-45 deg swing explains all good lines.
    best = None
    for s_ref in (+1.0, -1.0):
        theta_ref = a_meas[ref_line] - np.pi / 2.0 + s_ref * np.pi / 4.0
        model = theta_ref + LINE_ADV * (lines - ref_line)
        s_hyp = s_ref * np.where((lines - ref_line) % 2 == 0, 1.0, -1.0)
        dev = _wrap(a_meas - (model + np.pi / 2.0 - s_hyp * np.pi / 4.0))
        score = float(np.sum(np.cos(dev)[good] * amp[good]))
        if best is None or score > best[0]:
            best = (score, s_hyp, model)
    _, s, model = best

    # Per-line phase measurement with the parity swing removed, then
    # robust trajectory smoothing of the deviation from the model.
    theta_meas = a_meas - np.pi / 2.0 + s * np.pi / 4.0
    d = _wrap(theta_meas - model)
    a_w = np.where(good, np.clip(amp / (np.median(amp[good]) + 1e-9),
                                 0.0, 2.0), 0.0)
    lam = 25.0
    xs = _tridiag_smooth(d, a_w, lam)
    r = np.abs(d - xs)
    a2 = a_w * 0.15 / np.maximum(r, 0.15)
    xs = _tridiag_smooth(d, a2, lam)
    theta = model + xs
    return theta, s, amp


def burst_amplitude_ire(field_ire: np.ndarray, p: VideoParameters) -> float:
    """Measured burst amplitude (IRE) for ACC. |z| = B/2 regardless of
    the swing direction (the +/-45 deg rotation does not change the
    modulus), so the NTSC estimator carries over verbatim."""
    x = np.arange(p.colour_burst_start, p.colour_burst_end)
    seg = field_ire[:, p.colour_burst_start:p.colour_burst_end]
    seg = seg - seg.mean(axis=1, keepdims=True)
    z = (seg * np.exp(-1j * (np.pi / 2.0) * x)[None, :]).mean(axis=1)
    amp = 2.0 * np.abs(z)
    good = amp > 0.25 * np.median(amp[amp > 0]) if (amp > 0).any() else amp > -1
    return float(np.median(amp[good])) if good.any() else NOMINAL_BURST_IRE


def effective_carrier(theta: np.ndarray, s: np.ndarray, width: int) -> np.ndarray:
    """The V-switch folded into a unit-modulus per-pixel carrier:

        c = exp(+i*phi) where s=+1,  -exp(-i*phi) where s=-1

    so that S = Y + Re[chi * c] with ONE chroma phasor chi = V - iU for
    the whole image. All decoder mathematics is generic in c."""
    x = np.arange(width)
    phi = theta[:, None] + (np.pi / 2.0) * x[None, :]
    cpos = np.exp(1j * phi)
    return np.where(s[:, None] > 0, cpos, -np.conj(cpos))


# ------------------------------------------- holographic reconstruction

def _gaussian_lpf_kernel_fft(shape, p: VideoParameters, cfg: DecoderConfig):
    h, w = shape
    fx = fftfreq(w, d=1.0 / p.sample_rate)
    fy = fftfreq(h, d=1.0)
    cutoff_x = cfg.init_lpf_h_mhz * 1e6
    cutoff_y = cfg.init_lpf_v_cph / (2.0 * h)
    gx = np.exp(-0.5 * (fx / cutoff_x) ** 2)
    gy = np.exp(-0.5 * (fy / cutoff_y) ** 2)
    return gy[:, None] * gx[None, :]


def _box_blur(a, r=2):
    """Integral-image box blur, constant per-axis normalisation
    (semantics identical to the NTSC reference)."""
    k = 2 * r + 1
    pad = np.zeros((a.shape[0] + 1, a.shape[1]), a.dtype)
    np.cumsum(a, axis=0, out=pad[1:])
    up = np.zeros_like(a)
    lo = np.clip(np.arange(a.shape[0]) - r, 0, None)
    hi = np.clip(np.arange(a.shape[0]) + r + 1, None, a.shape[0])
    up = (pad[hi] - pad[lo]) / k
    pad2 = np.zeros((up.shape[0], up.shape[1] + 1), a.dtype)
    np.cumsum(up, axis=1, out=pad2[:, 1:])
    lo2 = np.clip(np.arange(a.shape[1]) - r, 0, None)
    hi2 = np.clip(np.arange(a.shape[1]) + r + 1, None, a.shape[1])
    return (pad2[:, hi2] - pad2[:, lo2]) / k


def holographic_init(S: np.ndarray, carrier: np.ndarray,
                     p: VideoParameters, cfg: DecoderConfig):
    """Digital-holography reconstruction with Dubois-style adaptive
    sideband arbitration, generic in the effective carrier.

    Demodulating by conj(c) moves chi to DC on every line REGARDLESS of
    the V-switch (|c| = 1), while luma lands at +/-fsc horizontally —
    with a vertical structure that differs from NTSC: the per-line
    conjugation splits luma leakage across two vertical alias positions
    (quarter-line offset). The dual anisotropic crops + smoothest-
    residual-luma arbitration handle both, unchanged."""
    demod = S * np.conj(carrier)
    D = fft2(demod)

    variants = []
    for hx, vy in ((0.8, 120.0), (1.8, 30.0)):
        c2 = dataclasses.replace(cfg, init_lpf_h_mhz=hx, init_lpf_v_cph=vy)
        G = _gaussian_lpf_kernel_fft(S.shape, p, c2)
        chi_v = 2.0 * ifft2(D * G)
        Yv = S - np.real(chi_v * carrier)
        E = _box_blur(np.abs(_dx(Yv)) + np.abs(_dy(Yv)), r=3)
        variants.append((chi_v, 1.0 / (E + 0.5)))

    # --- Transform PAL as a CERTIFIER (native on PAL) ----------------
    # BBC Transform PAL extracts chroma by thresholding spectral
    # symmetry about the carrier — the technique is native to PAL (no
    # NTSC quadrature blocker). Here the repaired form from the NTSC
    # work: min(|Z(+k)|,|Z(-k)|) per point-reflected bin pair of the
    # DEMODULATED spectrum is a chroma lower bound luma almost never
    # fakes; the asymmetric remainder is left to the arbitration.
    # NTSC verdict was "subsumed, neutral"; PAL's cleaner symmetry
    # justified re-measuring (see THEORY 6) — bench: run_tests 5.
    if cfg.symmetry_variant:
        mag = np.abs(D)
        mag_r = np.roll(np.roll(mag[::-1, ::-1], 1, axis=0), 1, axis=1)
        sym = np.minimum(mag, mag_r) / (mag + 1e-6)
        c3 = dataclasses.replace(cfg, init_lpf_h_mhz=1.3, init_lpf_v_cph=60.0)
        G3 = _gaussian_lpf_kernel_fft(S.shape, p, c3)
        chi_s = 2.0 * ifft2(D * sym * G3)
        Ys_ = S - np.real(chi_s * carrier)
        Es = _box_blur(np.abs(_dx(Ys_)) + np.abs(_dy(Ys_)), r=3)
        variants.append((chi_s, 1.0 / (Es + 0.5)))

    wsum = sum(w for (_, w) in variants)
    chi = sum(cv * w for (cv, w) in variants) / wsum
    Y = S - np.real(chi * carrier)
    return Y, chi


# ------------------------------------------------- gradient operators

def _dx(a):
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
    """chroma_aniso == 0 => AUTO, ported from NTSC with the PAL leak
    structure accounted for. The init's dominant vertical energy is
    cross-colour LEAK; under the effective carrier the leak's line-to-
    line behaviour mixes a sign flip with a conjugation (conj(c)
    alternates between e^{-i phi} and -e^{+i phi}), so a plain 2-line
    average does not cancel it the way it does on NTSC. A 4-LINE
    average does (the conjugation pattern repeats over 2 lines and the
    270 deg/line carrier walk completes 1080 = 0 mod 360 over 4):
    genuine chroma structure survives, the leak cancels. p98 one-step
    gradients on the 4-line averages, mapped into [0.55, 1.0] centred
    nearer isotropy than NTSC (PAL U/V bandwidths are equal)."""
    if cfg.chroma_aniso > 0.0:
        return cfg.chroma_aniso
    h4 = (chi0.shape[0] // 4) * 4
    q = 0.25 * (chi0[0:h4:4] + chi0[1:h4:4] + chi0[2:h4:4] + chi0[3:h4:4])
    gx = np.abs(np.diff(q[:, ::3], axis=1)).ravel()
    gy = np.abs(np.diff(q[:, 1::3], axis=0)).ravel()
    if gx.size == 0 or gy.size == 0:
        return 0.7
    px = float(np.percentile(gx, 98))
    py = float(np.percentile(gy, 98))
    r = py / max(px, 1e-6)
    import os as _os
    val = float(np.clip(0.7 + 0.9 * (r - 1.3), 0.55, 1.0))
    if _os.environ.get("HVD_PAL_DEBUG_ANISO"):
        print(f"    aniso auto: px={px:.2f} py={py:.2f} r={r:.2f} -> {val:.2f}")
    return val


# ------------------------------------------------- variational refinement

def variational_refine(S, carrier, Y0, chi0, cfg: DecoderConfig,
                       irls_outer: int = 4, neighbors=None):
    """Arbitration with luma eliminated, generic in the carrier.

    Identical structure to the NTSC solver; PAL enters ONLY through c:

    * dot crawl  = carrier-frequency ripple in Y  -> huge luma prior
      cost -> migrates into chi (unchanged from NTSC);
    * cross-colour = 2fsc oscillation in chi -> huge chroma prior cost
      -> migrates back into Y (unchanged);
    * HANOVER BARS (new, PAL-only) = a line-alternating conjugation
      error in chi -> the highest possible VERTICAL frequency in chi
      -> maximally penalised by mu_v -> the optimiser converges to the
      line-pair-coherent solution. Differential phase errors therefore
      come out as slight desaturation, not hue error or bars — the
      delay-line guarantee, recovered as a property of the prior
      rather than a hardwired 2-line average.
    """
    eps = cfg.charbonnier_eps
    eps_c = cfg.chroma_eps
    mu_h = cfg.lambda_c * _resolve_chroma_aniso(chi0, cfg)
    mu_v = cfg.lambda_c
    chi = chi0.copy()
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    eps_t = cfg.temporal_eps
    if nu > 0.0 and eps_t <= 0.0:
        eps_t = max(1.0, 7.0 * estimate_noise_ire(S))
    # per-neighbor: MC-warped composite, carrier difference, confidence.
    # For SAME-PARITY PAL neighbors the static-pixel geometry is
    # c_k = c*exp(+i Delta) on s=+1 lines and c*exp(-i Delta) on s=-1
    # lines (the V-switch conjugates the rotation) — |dc| is parity-
    # independent, and using the ACTUAL warped carrier arrays makes the
    # bookkeeping exact rather than assumed.
    nbr = [(S_w, carrier - c_w, conf) for (S_w, c_w, conf) in neighbors]

    n_inner = max(1, cfg.cg_iterations // max(1, irls_outer))

    def luma(chi):
        return S - np.real(chi * carrier)

    def temporal_residual(chi, S_w, dc):
        # r = (S_w - S) + Re[chi (c - c_w)]; zeroing it for a static
        # pixel with the f+/-4 flip (dc = 2c) forces Re[chi c] =
        # (S - S_w)/2 — the 2-frame PAL 3D comb, embedded in the
        # arbitration; f+/-2 (|dc| = sqrt2) adds a second, rotated
        # equation NTSC does not have at that distance.
        return (S_w - S) + np.real(chi * dc)

    for _outer in range(irls_outer):
        Y = luma(chi)
        gxY, gyY = _dx(Y), _dy(Y)
        wx = eps / np.sqrt(gxY ** 2 + eps ** 2)
        wy = eps / np.sqrt(gyY ** 2 + eps ** 2)

        cx, cy = np.abs(_dx(chi)), np.abs(_dy(chi))
        k = cfg.structure_coupling
        wcx = eps_c / np.sqrt(cx ** 2 + k * gxY ** 2 + eps_c ** 2)
        wcy = eps_c / np.sqrt(cy ** 2 + k * gyY ** 2 + eps_c ** 2)

        # temporal confidence: Geman-McClure on the composite-domain
        # residual, pre-gated by block-matching confidence (as NTSC)
        wts = []
        if nu > 0.0:
            for S_w, dc, conf in nbr:
                rt = temporal_residual(chi, S_w, dc)
                wts.append(conf * eps_t ** 2 / (rt ** 2 + eps_t ** 2))

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
            if gg < cfg.cg_tol ** 2 * gg0 or gg < 1e-10 * S.size:
                break

    return luma(chi), chi


# --------------------------------------------- delay-line PAL baseline

def delayline_baseline(S, carrier, p: VideoParameters, cfg: DecoderConfig):
    """Classic analogue delay-line PAL decoder, expressed in chi-space:
    per-line horizontal demodulation + low-pass, then the 2-line
    average. In effective-carrier terms the 64 us delay-line + add/
    subtract network is EXACTLY chi_dl[l] = (chi_raw[l] + chi_raw[l-1])/2
    (the V-switch conjugation is already inside c). Comparison
    reference for the tests — the incumbent every PAL decoder must
    beat."""
    demod = S * np.conj(carrier)
    # horizontal-only low-pass at the chroma bandwidth
    w = S.shape[1]
    fx = fftfreq(w, d=1.0 / p.sample_rate)
    G = np.exp(-0.5 * (fx / (cfg.init_lpf_h_mhz * 1e6)) ** 2)
    chi_raw = 2.0 * np.fft.ifft(np.fft.fft(demod, axis=1) * G[None, :], axis=1)
    chi_dl = chi_raw.copy()
    chi_dl[1:] = 0.5 * (chi_raw[1:] + chi_raw[:-1])
    Y = S - np.real(chi_dl * carrier)
    return Y, chi_dl


# ------------------------------------------------------------ frame path

def prepare_frame(src: TbcSource, frame_index: int, cfg: DecoderConfig):
    """Weave the two fields' ACTIVE regions into frame geometry and
    build the woven effective carrier from per-field burst lock-ins."""
    p = src.params
    (f0, m0), (f1, m1) = src.read_frame_fields(frame_index)
    S0, S1 = p.ire(f0), p.ire(f1)

    th0, s0, _ = burst_lockin(S0, p)
    th1, s1, _ = burst_lockin(S1, p)

    fal = p.first_active_field_line
    lal = p.last_active_field_line or p.field_height
    a0, a1 = p.active_video_start, p.active_video_end

    rows0 = np.arange(fal, lal)
    n = len(rows0)
    Sf = np.empty((2 * n, a1 - a0))
    th = np.empty(2 * n)
    sw = np.empty(2 * n)
    Sf[0::2] = S0[rows0, a0:a1]
    Sf[1::2] = S1[rows0, a0:a1]
    # theta refers to sample 0 of the stored line; shift to the active
    # start so phi = theta_active + (pi/2)*x with x from 0
    th[0::2] = th0[rows0] + (np.pi / 2.0) * a0
    th[1::2] = th1[rows0] + (np.pi / 2.0) * a0
    sw[0::2] = s0[rows0]
    sw[1::2] = s1[rows0]

    carrier = effective_carrier(th, sw, a1 - a0)

    gain = 1.0
    if cfg.acc:
        amp = 0.5 * (burst_amplitude_ire(S0, p) + burst_amplitude_ire(S1, p))
        gain = float(np.clip(NOMINAL_BURST_IRE / max(amp, 1e-6), 0.5, 2.0))
    return Sf, carrier, gain


def decode_frame(src: TbcSource, frame_index: int, cfg: DecoderConfig):
    """Decode one frame -> (Y, U, V) at active frame geometry (IRE)."""
    p = src.params
    S, carrier, acc_gain = prepare_frame(src, frame_index, cfg)
    Y0, chi0 = holographic_init(S, carrier, p, cfg)
    if cfg.cg_iterations > 0:
        Y, chi = variational_refine(S, carrier, Y0, chi0, cfg)
    else:
        Y, chi = Y0, chi0
    # purity contract: the delivered pair reconstructs S exactly
    Y = S - np.real(chi * carrier)
    g = cfg.chroma_gain * acc_gain
    V = np.real(chi) * g
    U = -np.imag(chi) * g
    if cfg.monochrome:
        U = np.zeros_like(U)
        V = np.zeros_like(V)
    return Y, U, V


def yuv_to_rgb16(Y, U, V, black_ire=IRE_BLACK):
    yuv = np.stack([(Y - black_ire) / (IRE_WHITE - black_ire),
                    U / (IRE_WHITE - black_ire),
                    V / (IRE_WHITE - black_ire)], axis=-1)
    rgb = yuv @ YUV_TO_RGB.T
    return (np.clip(rgb, 0, 1) * 65535.0).astype(np.uint16)


# --------------------------------------------- motion (video-coding imports)
# Ported from the NTSC engine: coarse-to-fine tiled block matching with
# zero-motion margin rule, vector-median outlier-snap, OBMC-style
# per-pixel vector interpolation, and the confidence stack (tile-energy,
# median-calibrated outlier rejection, scene-cut global validity).

def _decimate(a, f):
    h, w = a.shape
    hh, ww = h // f * f, w // f * f
    return a[:hh, :ww].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def _bm_pass(A, B, tile, dys, dxs, base_dy=None, base_dx=None):
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
            cost = se * (1.0 + 0.02 * (abs(dy) + abs(dx)))
            upd = cost < best
            best[upd] = cost[upd]
            if base_dy is None:
                mdy[upd] = tot_dy
                mdx[upd] = tot_dx
            else:
                mdy[upd] = (base_dy + dy)[upd]
                mdx[upd] = (base_dx + dx)[upd]
    return mdy, mdx, best, cost0


def _motion_conf(A, best, tile):
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


def _tile_ssd(A, B, dy_t, dx_t, tile):
    """Per-tile SSD of A vs B warped by per-tile integer vectors."""
    h, w = A.shape
    th, tw = dy_t.shape
    ms = int(max(np.abs(dy_t).max(), np.abs(dx_t).max())) + 1
    Bp = np.pad(B, ms, mode="edge")
    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // tile, th - 1)
    tx = np.minimum(xx // tile, tw - 1)
    Bs = Bp[ms + yy - dy_t[ty, tx], ms + xx - dx_t[ty, tx]]
    D = (A - Bs) ** 2
    ph, pw = th * tile, tw * tile
    Dp = np.zeros((ph, pw))
    Dp[:h, :w] = D
    return Dp.reshape(th, tile, tw, tile).sum(axis=(1, 3))


def subpel_refine(Y_ref, Y_cur, mdy, mdx, tile=32):
    """Parabolic half-pel refinement of integer tile vectors (the
    NTSC drizzle prerequisite, ported). Three tile-SSD samples per
    axis around the integer optimum; the parabola vertex gives the
    sub-pixel component, clamped to +/-0.6. This is what lets the
    matcher SEE the half-line parity offset (0.5 field-line) that
    integer BM + the zero-margin rule deliberately snap away — used
    for baseband stacking (drizzle, blends) only, never for the raw
    equations (integer carrier copies are exact there)."""
    A = _box_blur(Y_cur)
    B = _box_blur(Y_ref)
    mdy = np.asarray(mdy, int); mdx = np.asarray(mdx, int)
    out_dy = mdy.astype(float); out_dx = mdx.astype(float)
    cmin = None
    # 5-point scan per axis BEFORE the parabola: the zero-motion
    # margin rule (correct for the raw equations) traps genuine
    # sub-line motion one integer off — the parabola then converges in
    # the wrong basin (measured: expected +0.75, parabola around the
    # snapped 0 returned -0.29). Re-minimise over +/-2 first.
    for axis in (0, 1):
        costs = []
        for d in (-2, -1, 0, 1, 2):
            if axis == 0:
                costs.append(_tile_ssd(A, B, mdy + d, mdx, tile))
            else:
                costs.append(_tile_ssd(A, B, mdy, mdx + d, tile))
        C = np.stack(costs)
        ibest = np.argmin(C, axis=0)
        ii, jj = np.mgrid[0:C.shape[1], 0:C.shape[2]]
        ic = np.clip(ibest, 1, 3)
        c0 = C[ic, ii, jj]
        cm = C[ic - 1, ii, jj]
        cp = C[ic + 1, ii, jj]
        den = cm - 2.0 * c0 + cp
        delta = (ic - 2) + np.clip(
            np.where(den > 1e-9,
                     0.5 * (cm - cp) / np.maximum(den, 1e-9), 0.0),
            -0.6, 0.6)
        # parabola-vertex cost: the cost AT the sub-pixel optimum, not
        # at the integer sample — on a genuine half-line displacement
        # the integer cost sits on the plateau and would slander a
        # perfectly registrable neighbor
        cvert = np.where(den > 1e-9,
                         c0 - (cm - cp) ** 2 / (8.0 * np.maximum(den, 1e-9)),
                         c0)
        cmin = cvert if cmin is None else np.minimum(cmin, cvert)
        if axis == 0:
            out_dy += delta
        else:
            out_dx += delta
    return out_dy, out_dx, cmin


def estimate_motion(Y_ref, Y_cur, tile=32, search=16):
    """Coarse-to-fine integer-pel BM + zero-motion margin rule. The
    margin rule matters MORE on PAL than NTSC: a spurious 1-sample
    shift rotates the carrier 90 deg here too, but the same-parity
    temporal geometry (270/180 deg) means a wrong small vector
    corrupts BOTH equations at once."""
    A = _box_blur(Y_cur)
    B = _box_blur(Y_ref)
    h, w = A.shape
    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile

    f = 4
    cs = max(1, int(np.ceil(search / f)))
    Ad, Bd = _decimate(A, f), _decimate(B, f)
    cdy, cdx, _, _ = _bm_pass(Ad, Bd, max(4, tile // f),
                              range(-cs, cs + 1), range(-cs, cs + 1))
    hh, ww = cdy.shape
    gy = (np.arange(th) * hh // th).clip(0, hh - 1)
    gx = (np.arange(tw) * ww // tw).clip(0, tw - 1)
    base_dy = (cdy[gy][:, gx] * f).astype(np.int32)
    base_dx = (cdx[gy][:, gx] * f).astype(np.int32)

    mdy, mdx, best, _ = _bm_pass(A, B, tile, range(-3, 4), range(-3, 4),
                                 base_dy=base_dy, base_dx=base_dx)
    _, _, best0, cost0 = _bm_pass(A, B, tile, [0], [0])
    take0 = best0 < best
    mdy[take0] = 0
    mdx[take0] = 0
    best[take0] = best0[take0]
    keep_zero = best > 0.85 * cost0
    mdy[keep_zero] = 0
    mdx[keep_zero] = 0
    best[keep_zero] = cost0[keep_zero]
    return mdy.astype(float), mdx.astype(float), _motion_conf(A, best, tile)


def _vectors_per_pixel(mdy, mdx, tile, shape):
    h, w = shape
    th, tw = mdy.shape

    def _med3(V):
        if V.shape[0] < 3 or V.shape[1] < 3:
            return V
        Vp = np.pad(V, 1, mode="edge")
        stack = [Vp[1 + dy:1 + dy + V.shape[0], 1 + dx:1 + dx + V.shape[1]]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        return np.median(np.stack(stack), axis=0)

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
    fy = np.clip((yy - cy[iy]) / tile, 0.0, 1.0) if th > 1 else np.zeros(h)
    fx = np.clip((xx - cx[ix]) / tile, 0.0, 1.0) if tw > 1 else np.zeros(w)

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
    """Integer-pel warp with OBMC-smoothed vectors. Used for the RAW
    measurements (composite + effective carrier): integer shifts copy
    carrier VALUES exactly — no complex interpolation error, and the
    V-switch parity of every source line travels with its carrier
    sample (nothing to book-keep)."""
    h, w = a.shape[:2]
    vdy, vdx = _vectors_per_pixel(np.asarray(mdy, float),
                                  np.asarray(mdx, float), tile, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    sy = np.clip(yy - np.round(vdy).astype(int), 0, h - 1)
    sx = np.clip(xx - np.round(vdx).astype(int), 0, w - 1)
    return a[sy, sx]


def complex_coherence(z1, z2, r=6):
    """InSAR-style local complex coherence between chroma fields (as
    NTSC): phase-sensitive where an SSD is not."""
    num = np.abs(_box_blur((z1 * np.conj(z2)).real, r)
                 + 1j * _box_blur((z1 * np.conj(z2)).imag, r))
    den = np.sqrt(_box_blur(np.abs(z1) ** 2, r)
                  * _box_blur(np.abs(z2) ** 2, r)) + 1e-6
    return np.clip(num / den, 0.0, 1.0)


def motion_compensate_neighbor(nb, Y_cur_init, cfg, motion=None):
    """Warp a neighbor FIELD's raw measurements (S, effective carrier)
    toward the current field. Matching runs on decoded-luma inits;
    the equations get raw composite, never recycled estimates."""
    Y_nb, S_nb, c_nb = nb
    if motion is None:
        motion = estimate_motion(Y_nb, Y_cur_init,
                                 tile=cfg.mc_tile, search=cfg.mc_search)
    mdy, mdx, conf = motion
    h, w = Y_nb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ty = np.minimum(yy // cfg.mc_tile, conf.shape[0] - 1)
    tx = np.minimum(xx // cfg.mc_tile, conf.shape[1] - 1)
    conf_px = _box_blur(conf[ty, tx] ** 2, r=8)
    S_w = warp_by_tiles(S_nb, mdy, mdx, cfg.mc_tile)
    c_w = (warp_by_tiles(c_nb.real, mdy, mdx, cfg.mc_tile)
           + 1j * warp_by_tiles(c_nb.imag, mdy, mdx, cfg.mc_tile))
    return S_w, c_w, conf_px, motion


# ---------------------------------- polarimetric decomposition (PolSAR)

def polarimetric_maps(S, carrier, p: VideoParameters, cfg: DecoderConfig):
    """Pauli-style co-pol / cross-pol decomposition over V-switch line
    pairs (polarimetric SAR import, THEORY 4). With the V-switch
    folded into the effective carrier, adjacent lines are the two
    'polarisation channels':

        chi_co = (chi_raw[l] + chi_raw[l+1]) / 2   pair-coherent chroma
        chi_x  = (chi_raw[l] - chi_raw[l+1]) / 2   conjugation-error ch.

    CRUCIAL: chi_raw must be the RAW per-line demodulation (horizontal
    low-pass only). The decoder's own chroma has already had the
    line-alternating component arbitrated away by the vertical prior —
    measuring the map on it reads ~0 by construction (found the hard
    way: 0.15 IRE measured on chi0 vs 7.6 IRE predicted; the init's
    60 c/ph vertical crop had eaten the evidence).

    |chi_x| is OPERATOR-FACING: differential-phase distortion (what
    would have been Hanover bars) lights up at |chi|*sin(delta);
    legitimate vertical chroma detail feeds both channels. The decoder
    does not consume this map — it tells the archivist WHERE and HOW
    MUCH the source chain is phase-distorted, per pixel."""
    demod = S * np.conj(carrier)
    w = S.shape[1]
    fx = fftfreq(w, d=1.0 / p.sample_rate)
    G = np.exp(-0.5 * (fx / (cfg.init_lpf_h_mhz * 1e6)) ** 2)
    chi_raw = 2.0 * np.fft.ifft(np.fft.fft(demod, axis=1) * G[None, :],
                                axis=1)
    co = 0.5 * (chi_raw[:-1] + chi_raw[1:])
    cx = 0.5 * (chi_raw[:-1] - chi_raw[1:])
    return np.abs(co), np.abs(cx)


# ------------------------------------------------------- 3D sequence path

def prepare_field(src: TbcSource, field_index: int, cfg: DecoderConfig):
    """One field's active region + effective carrier (field geometry)."""
    p = src.params
    S_full = p.ire(src.read_field(field_index))
    th, sw, _ = burst_lockin(S_full, p)
    fal = p.first_active_field_line
    lal = p.last_active_field_line or p.field_height
    a0, a1 = p.active_video_start, p.active_video_end
    rows = np.arange(fal, lal)
    S = S_full[rows, a0:a1]
    carrier = effective_carrier(th[rows] + (np.pi / 2.0) * a0,
                                sw[rows], a1 - a0)
    return S, carrier, burst_amplitude_ire(S_full, p)


def _halfline_gate(Y0, chi0, eps_t):
    """Odd-offset validity envelope (NTSC 9h gate, ported verbatim in
    form): thin horizontal detail at/beyond the opposite parity's
    vertical Nyquist is INVISIBLE to that parity — its equations vote
    "background", and on PAL the vote's residual oscillates doubly
    (the |dchi cos| mechanism of NTSC PLUS the odd-offset |dc|
    oscillation of THEORY 5c), so the robust weight passes confident
    wrong votes at the zeros. Deterministic per-pixel gate: BASEBAND
    one-sided-max envelope of sqrt(dY^2 + |dchi|^2) (central diffs are
    exactly zero ON a one-row feature), horizontal-only smoothing (a
    2D blur dilutes a one-row footprint), floor 0.35 (step edges are
    displaced-but-informative; hard-gating them cost NTSC 4.4 dB of
    3D gain)."""
    dyu = np.abs(np.diff(Y0, axis=0, prepend=Y0[:1]))
    dyd = np.abs(np.diff(Y0, axis=0, append=Y0[-1:]))
    dcu = np.abs(np.diff(chi0, axis=0, prepend=chi0[:1]))
    dcd = np.abs(np.diff(chi0, axis=0, append=chi0[-1:]))
    m = np.maximum(dyu ** 2 + dcu ** 2, dyd ** 2 + dcd ** 2)
    hh, ww = m.shape
    c = np.zeros((hh, ww + 1))
    np.cumsum(m, axis=1, out=c[:, 1:])
    x0 = np.clip(np.arange(ww) - 2, 0, ww)
    x1 = np.clip(np.arange(ww) + 3, 0, ww)
    vg = np.sqrt((c[:, x1] - c[:, x0]) / np.maximum(x1 - x0, 1))
    return np.maximum(0.35, eps_t ** 2 / (eps_t ** 2 + vg * vg))


def decode_sequence(src: TbcSource, start: int, length: int,
                    cfg: DecoderConfig):
    """Streaming 3D decode: FIELDS are the decode unit (weave after).
    Processes the segment in chunks of `chunk_frames` with
    `chunk_overlap` frames of temporal CONTEXT on each side (context
    fields feed equations and the anchor blend but are not emitted),
    so memory is bounded for arbitrarily long sources. Yields woven
    (Y, U, V) frames; with cfg.drizzle, vertically 2x-stacked frames.
    """
    n_frames = (src.num_frames - start) if length < 0 else length
    step = max(1, cfg.chunk_frames)
    done = 0
    while done < n_frames:
        cn = min(step, n_frames - done)
        yield from _decode_chunk(src, start + done, cn, cfg)
        done += cn


def _decode_chunk(src: TbcSource, start: int, n_frames: int,
                  cfg: DecoderConfig):
    f0 = start * 2
    n_fields = n_frames * 2
    max_off = max((abs(o) for o in cfg.temporal_offsets), default=0)
    ctx = max(max_off, 2 * cfg.chunk_overlap, cfg.nr_radius)
    lo = max(0, f0 - ctx)
    hi = min(src.num_fields, f0 + n_fields + ctx)

    fields = {}
    for i in range(lo, hi):
        S, c, burst = prepare_field(src, i, cfg)
        Y0, chi0 = holographic_init(S, c, src.params, cfg)
        fields[i] = dict(S=S, c=c, Y0=Y0, chi0=chi0, burst=burst)

    parities = {i: (0.0 if src.fields[i].is_first_field else 1.0)
                for i in range(lo, hi)}

    # ---- shared motion cache (NTSC fast-mode item, default here):
    # ONE estimate per ordered field pair, matched on the pass-1
    # inits, reused by the equations, the anchor blend, and drizzle.
    mcache = {}

    def motion_for(i, k):
        if (i, k) not in mcache:
            mcache[(i, k)] = estimate_motion(
                fields[k]["Y0"], fields[i]["Y0"],
                tile=cfg.mc_tile, search=cfg.mc_search)
        return mcache[(i, k)]

    def pair_motion_for(i):
        """Per-field pairwise motion over all offsets, with the
        trajectory-coherent snap (4D brainstorm import). Odd offsets
        measure content displacement PLUS the half-line parity term
        h_k = (p_k - p_i)/2 — removed before fitting the per-tile
        velocity, reinstated after (NTSC 9e; the sign matters)."""
        pm = {}
        for o in cfg.temporal_offsets:
            k = i + o
            if k in fields:
                pm[o] = motion_for(i, k)
        if len(pm) >= 2 and cfg.trajectory_fit:
            os_ = sorted(pm)
            pi = parities[i]
            hk = {o: (parities[i + o] - pi) / 2.0 for o in os_}
            vy = np.stack([(pm[o][0] - hk[o]) / o for o in os_])
            vx = np.stack([pm[o][1] / o for o in os_])
            wv = np.stack([pm[o][2] for o in os_])
            vy = np.where(wv > 0.15, vy, np.nan)
            vx = np.where(wv > 0.15, vx, np.nan)
            import warnings as _w
            with np.errstate(all="ignore"), _w.catch_warnings():
                _w.simplefilter("ignore")
                vym = np.nan_to_num(np.nanmedian(vy, axis=0))
                vxm = np.nan_to_num(np.nanmedian(vx, axis=0))
            agree = 0
            for o in os_:
                mdy, mdx, _ = pm[o]
                agree = agree + ((np.abs(mdy - (o * vym + hk[o]))
                                  + np.abs(mdx - o * vxm)) <= 1.5)
            consensus = agree >= 3
            for o in os_:
                mdy, mdx, cf = pm[o]
                py, px = o * vym + hk[o], o * vxm
                okm = consensus & (
                    (np.abs(mdy - py) + np.abs(mdx - px)) <= 1.5)
                pm[o] = (np.where(okm, py, mdy),
                         np.where(okm, px, mdx), cf)
        return pm

    def neighbors_for(i):
        F = fields[i]
        out = []
        if cfg.temporal_strength <= 0.0:
            return out
        eps_t = cfg.temporal_eps
        if eps_t <= 0.0:
            eps_t = max(1.0, 7.0 * estimate_noise_ire(F["S"]))
        og = _halfline_gate(F["Y0"], F["chi0"], eps_t)
        pm = pair_motion_for(i)
        for o in cfg.temporal_offsets:
            k = i + o
            if k not in fields:
                continue
            N = fields[k]
            S_w, c_w, conf, _ = motion_compensate_neighbor(
                (N["Y0"], N["S"], N["c"]), F["Y0"], cfg, motion=pm[o])
            if o % 2 != 0:
                conf = conf * og
            if cfg.coherence_gate > 0.0:
                gam = complex_coherence(F["chi0"], S_w * np.conj(c_w))
                g = cfg.coherence_gate
                conf = conf * ((1.0 - g) + g * np.maximum(gam, 0.25))
            out.append((S_w, c_w, conf))
        return out

    refined = {}
    for i in range(lo, hi):
        F = fields[i]
        emit_range = f0 <= i < f0 + n_fields
        neighbors = neighbors_for(i) if emit_range else []
        Y, chi = variational_refine(F["S"], F["c"], F["Y0"], F["chi0"],
                                    cfg, neighbors=neighbors)
        refined[i] = (Y, chi)
        F["nbr"] = neighbors

    # ---- passes >= 2: decode -> MC temporal NR -> re-encode anchor ---
    for _pass in range(1, cfg.passes):
        Ys = {i: refined[i][0] for i in refined}
        chis = {i: refined[i][1] for i in refined}
        new_refined = {}
        for i in range(f0, f0 + n_fields):
            F = fields[i]
            anchor = synth_reference(i, Ys, chis, F["S"], F["c"],
                                     parities, cfg, motion_cache=mcache)
            Y, chi = variational_refine_joint(
                F["S"], F["c"], Ys[i], chis[i], cfg,
                neighbors=F["nbr"], anchor=anchor)
            if cfg.output_fidelity:
                # purity contract: the NR machinery only GUIDES the
                # separation; delivered luma = raw composite minus
                # reconstructed chroma
                Y = F["S"] - np.real(chi * F["c"])
            new_refined[i] = (Y, chi)
        refined.update(new_refined)

    p = src.params
    for fr in range(start, start + n_frames):
        i0 = fr * 2
        m0 = src.fields[i0]
        first, second = ((i0, i0 + 1) if m0.is_first_field
                         else (i0 + 1, i0))
        gain = 1.0
        if cfg.acc:
            amp = 0.5 * (fields[first]["burst"] + fields[second]["burst"])
            gain = float(np.clip(NOMINAL_BURST_IRE / max(amp, 1e-6),
                                 0.5, 2.0))
        g = cfg.chroma_gain * gain

        if cfg.drizzle:
            Ys = {i: refined[i][0] for i in refined}
            chis = {i: refined[i][1] for i in refined}
            Yd, Cd = drizzle_frame(first, Ys, chis, parities, cfg,
                                   motion_cache=mcache)
            U, V = -np.imag(Cd) * g, np.real(Cd) * g
            if cfg.monochrome:
                U[:] = 0.0; V[:] = 0.0
            yield Yd, U, V
            continue

        Y1, chi1 = refined[first]
        Y2, chi2 = refined[second]
        h, w = Y1.shape
        Y = np.empty((2 * h, w)); U = np.empty((2 * h, w)); V = np.empty((2 * h, w))
        Y[0::2], Y[1::2] = Y1, Y2
        U[0::2], U[1::2] = -np.imag(chi1) * g, -np.imag(chi2) * g
        V[0::2], V[1::2] = np.real(chi1) * g, np.real(chi2) * g
        if cfg.monochrome:
            U[:] = 0.0; V[:] = 0.0
        yield Y, U, V


def drizzle_frame(j0, Ys, chis, parities, cfg, scale=2,
                  motion_cache=None):
    """Astronomy drizzle (Fruchter & Hook), adapted to interlaced PAL:
    vertical 2x super-resolution by scatter accumulation. Vertical
    only, by the same physics as NTSC: PAL's 576 active lines have no
    optical vertical prefilter and alias heavily, while the 4fsc
    horizontal sampling already oversamples the analog luma bandwidth
    (the ~5 MHz PAL luma is oversampled ~1.7x at 17.7 MHz — slightly
    less headroom than NTSC's ratio, still no aliased detail to
    recover). Offsets: the intrinsic half-line parity phase between
    opposite fields (a free, guaranteed phase) + sub-pixel vertical
    motion. Robust weights degrade gracefully to weave-interpolation
    where coverage is thin.

    Boundary statement unchanged from NTSC: this is reconstruction,
    NOT decoding — an opt-in archival enhancement outside the purity
    contract."""
    L, W = Ys[j0].shape
    HF = 2 * L * scale
    accY = np.zeros((HF, W)); accC = np.zeros((HF, W), complex)
    accW = np.zeros((HF, W))
    eps_b = cfg.nr_eps if cfg.nr_eps > 0 else 3.0

    idxs = sorted(Ys.keys())
    for jt in (j0, j0 + 1):
        ks = [k for k in range(max(idxs[0], j0 - cfg.nr_radius),
                               min(idxs[-1] + 1, j0 + 2 + cfg.nr_radius))
              if k not in (j0, j0 + 1) or k == jt]

        # ---- trajectory-based sub-pixel registration -----------------
        # Genuine sub-line displacements sit ON the integer-BM plateau
        # (a 0.5-line shift costs the same at 0 and 1), so per-pair
        # matching honestly collapses for exactly the neighbors whose
        # deposits fill the uncovered fine-row residues (measured:
        # f+/-2 conf 0.13 while f+/-4 is exact). The trajectory
        # assumption rescues them: fit ONE per-tile velocity from the
        # offsets that DO measure (long/integer ones constrain it
        # sharply), then hand each neighbor a predicted base
        # o*v + h_k and let the 5-point parabola polish it.
        pms = {}
        for k in ks:
            if k == jt:
                continue
            if motion_cache is not None and (jt, k) in motion_cache:
                pms[k] = motion_cache[(jt, k)]
            else:
                pms[k] = estimate_motion(Ys[k], Ys[jt], tile=cfg.mc_tile,
                                         search=cfg.mc_search)
                if motion_cache is not None:
                    motion_cache[(jt, k)] = pms[k]
        vy_t = vx_t = None
        if pms:
            import warnings as _w
            oss = sorted(pms)
            hks = {k: (parities[k] - parities[jt]) / 2.0 for k in oss}
            sty = np.stack([(pms[k][0] - hks[k]) / (k - jt) for k in oss])
            stx = np.stack([pms[k][1] / (k - jt) for k in oss])
            stw = np.stack([pms[k][2] for k in oss])
            sty = np.where(stw > 0.3, sty, np.nan)
            stx = np.where(stw > 0.3, stx, np.nan)
            with np.errstate(all="ignore"), _w.catch_warnings():
                _w.simplefilter("ignore")
                vy_t = np.nan_to_num(np.nanmedian(sty, axis=0))
                vx_t = np.nan_to_num(np.nanmedian(stx, axis=0))

        for k in ks:
            pk = parities[k]
            if k == jt:
                sdy = np.zeros((1, 1)); sdx = np.zeros((1, 1))
                conf = np.ones((1, 1))
            else:
                o = k - jt
                base_dy = np.round(o * vy_t + hks[k]).astype(int)
                base_dx = np.round(o * vx_t).astype(int)
                sdy, sdx, cmin = subpel_refine(Ys[k], Ys[jt],
                                               base_dy, base_dx,
                                               tile=cfg.mc_tile)
                # confidence from the PARABOLA-VERTEX cost: the
                # integer-sample cost on a half-line plateau slanders
                # a perfectly registrable neighbor (measured: f+/-2
                # conf 0.13 at the integer, while its sub-pixel
                # registration is exact)
                conf = _motion_conf(_box_blur(Ys[jt]), cmin, cfg.mc_tile)
            vy, vx = _vectors_per_pixel(np.asarray(sdy, float),
                                        np.asarray(sdx, float),
                                        cfg.mc_tile, (L, W))
            th, tw = conf.shape
            yy, xx = np.mgrid[0:L, 0:W]
            cty = np.minimum(yy // cfg.mc_tile, th - 1)
            ctx_ = np.minimum(xx // cfg.mc_tile, tw - 1)
            cpx = conf[cty, ctx_]

            # agreement check at BILINEAR sub-pixel registration:
            # rounding here blinds the check to exactly the half-line
            # registrations drizzle exists to exploit, crushing the
            # good deposits (measured before this fix)
            Yw_c = _warp_bilinear_tiles(Ys[k], sdy, sdx, cfg.mc_tile,
                                        out_shape=(L, W))
            Cw_c = (_warp_bilinear_tiles(chis[k].real, sdy, sdx,
                                         cfg.mc_tile, out_shape=(L, W))
                    + 1j * _warp_bilinear_tiles(chis[k].imag, sdy, sdx,
                                                cfg.mc_tile,
                                                out_shape=(L, W)))
            d2 = ((Yw_c - Ys[jt]) ** 2 + np.abs(Cw_c - chis[jt]) ** 2)
            # d2 lives on the jt grid; deposits are per source pixel —
            # pull the weight back through the (rounded) map
            syi = np.clip(np.round(yy + vy).astype(int), 0, L - 1)
            sxi = np.clip(np.round(xx + vx).astype(int), 0, W - 1)
            w = cpx * eps_b ** 2 / (d2[syi, sxi] + eps_b ** 2)

            # 9g-class correction, derived for PAL before it bit:
            # the SUB-PIXEL vector measures content displacement
            # INCLUDING the half-line parity term h_k = (p_k - p_t)/2,
            # so the deposit must remove it before adding back the
            # source field's own parity phase — otherwise static
            # cross-parity content lands a full frame-line off
            # (2(y + h_k) + p_k instead of 2y + p_k). Verified:
            # static deposit reduces to 2y + p_k, pure motion m to
            # 2(y + m) + p_k, combined to 2(y + m) + p_k.
            hk_t = (pk - parities[jt]) / 2.0
            yf = (2.0 * (yy + vy - hk_t) + pk) * scale
            xs = np.clip(np.round(xx + vx).astype(int), 0, W - 1)
            y0_ = np.floor(yf).astype(int)
            fy = yf - y0_
            for off, wr in ((0, (1 - fy)), (1, fy)):
                yt = np.clip(y0_ + off, 0, HF - 1)
                np.add.at(accY, (yt, xs), (w * wr) * Ys[k])
                np.add.at(accC.real, (yt, xs), (w * wr) * chis[k].real)
                np.add.at(accC.imag, (yt, xs), (w * wr) * chis[k].imag)
                np.add.at(accW, (yt, xs), w * wr)

    base = np.zeros((HF, W)); baseC = np.zeros((HF, W), complex)
    for (dst, srcs) in ((base, (Ys[j0], Ys[j0 + 1])),
                        (baseC, (chis[j0], chis[j0 + 1]))):
        woven = np.zeros((2 * L, W), dst.dtype)
        woven[int(parities[j0])::2] = srcs[0]
        woven[int(parities[j0 + 1])::2] = srcs[1]
        fine_rows = np.arange(HF) / scale
        fl = np.clip(np.floor(fine_rows).astype(int), 0, 2 * L - 2)
        ff = (fine_rows - fl)[:, None]
        dst[:] = woven[fl] * (1 - ff) + woven[fl + 1] * ff

    lam = 0.35
    mix = accW / (accW + lam)
    Yf = np.where(accW > 0, accY / np.maximum(accW, 1e-9), 0.0)
    Cf = np.where(accW > 0, accC / np.maximum(accW + 0j, 1e-9), 0.0)
    return mix * Yf + (1 - mix) * base, mix * Cf + (1 - mix) * baseC



# ------------------------------------ re-encode anchor loop (passes >= 2)

def _warp_bilinear_tiles(a, dyf, dxf, tile, row_offset=0.0, out_shape=None,
                         vpix=None):
    """Bilinear sub-pixel warp of a BASEBAND array by per-tile float
    vectors, with a constant extra row offset for cross-parity grid
    alignment. Legal only on baseband quantities (decoded Y, chi):
    the raw composite/carrier must use integer-pel value copies."""
    ho, wo = out_shape if out_shape is not None else a.shape[:2]
    ho, wo = int(ho), int(wo)
    yy = np.arange(ho, dtype=np.float64)[:, None] * np.ones((1, wo))
    xx = np.ones((ho, 1)) * np.arange(wo, dtype=np.float64)[None, :]
    if vpix is None:
        vpix = _vectors_per_pixel(np.asarray(dyf, float),
                                  np.asarray(dxf, float), tile, (ho, wo))
    vdy, vdx = vpix
    sy = np.clip(yy - vdy + row_offset, 0, a.shape[0] - 1.001)
    sx = np.clip(xx - vdx, 0, a.shape[1] - 1.001)
    y0 = sy.astype(int); x0 = sx.astype(int)
    fy = sy - y0; fx = sx - x0
    return (a[y0, x0] * (1 - fy) * (1 - fx) + a[y0 + 1, x0] * fy * (1 - fx)
            + a[y0, x0 + 1] * (1 - fy) * fx + a[y0 + 1, x0 + 1] * fy * fx)


def synth_reference(j, Ys, chis, S_j, carrier_j, parities, cfg,
                    motion_cache=None):
    """The re-encode loop: build a denoised reference for field j by
    motion-compensated robust temporal blending of the DECODED fields,
    re-encode it, and measure trust in the composite domain.

    PAL blend geometry (differs from NTSC, worth recording): the
    separation-leakage anti-correlation that makes the blend powerful
    lives at the 180 deg carrier relation — which PAL reaches at f+/-4
    (2 frames), not f+/-1 like NTSC. f+/-2 neighbors are 270 deg
    (leak rotates 90 deg in chi: partial cancellation), odd offsets
    add cross-parity vertical coverage. Hence nr_radius default 4.

    Returns (Y_hat, chi_hat, conf_ref): conf_ref = 1 where the
    re-encoded synthesis Y_hat + Re[chi_hat c] explains the raw
    measurement S_j, -> 0 where it does not. The anchor never trusts
    a reference the data contradicts."""
    eps_b = cfg.nr_eps
    if eps_b <= 0.0:
        eps_b = max(1.5, 3.0 * estimate_noise_ire(S_j))
    accY = Ys[j].copy()
    accC = chis[j].copy()
    accW = np.ones_like(Ys[j])
    tile = cfg.mc_tile
    hh, ww = Ys[j].shape
    for k in range(j - cfg.nr_radius, j + cfg.nr_radius + 1):
        if k == j or k not in Ys:
            continue
        if motion_cache is not None and (j, k) in motion_cache:
            mdy, mdx, cf = motion_cache[(j, k)]
        else:
            mdy, mdx, cf = estimate_motion(Ys[k], Ys[j], tile=tile,
                                           search=cfg.mc_search)
            if motion_cache is not None:
                motion_cache[(j, k)] = (mdy, mdx, cf)
        yy, xx = np.mgrid[0:hh, 0:ww]
        ty = np.minimum(yy // tile, cf.shape[0] - 1)
        tx = np.minimum(xx // tile, cf.shape[1] - 1)
        conf = _box_blur(cf[ty, tx] ** 2, r=8)
        # decoded fields are baseband: bilinear sub-pixel is legal, and
        # the half-line offset between opposite-parity fields is
        # compensated exactly on the sampling grid
        row_off = (parities[j] - parities[k]) / 2.0
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

    S_hat = Y_hat + np.real(chi_hat * carrier_j)
    r = _box_blur(np.abs(S_j - S_hat), r=1)
    conf_ref = eps_b ** 2 / (r ** 2 + eps_b ** 2)
    return Y_hat, chi_hat, conf_ref


def variational_refine_joint(S, carrier, Y0, chi0, cfg: DecoderConfig,
                             irls_outer: int = 4, neighbors=None,
                             anchor=None):
    """Pass-2+ solver with RELAXED data fidelity, generic in c:

        E = ||S - Y - Re[chi c]||^2                      (soft data)
          + nu   sum_k wt_k (S_k - Y - Re[chi c_k])^2    (raw temporal)
          + nu_a sum w_a [(Y - Y_hat)^2 + |chi - chi_hat|^2]  (anchor)
          + spatial Charbonnier priors on grad Y, grad chi

    The exact-fidelity pass-1 forces Y to inherit ALL composite noise;
    relaxing is only safe with a trustworthy prior, and the decode ->
    MC temporal NR -> re-encode loop manufactures exactly that:
    (Y_hat, chi_hat) aggregates up to 2*nr_radius+1 raw measurements
    per pixel, and w_a (the composite-domain re-encode check) disables
    the anchor wherever the synthesis fails to explain the raw data."""
    eps = cfg.charbonnier_eps
    eps_c = cfg.chroma_eps
    mu_h = cfg.lambda_c * _resolve_chroma_aniso(chi0, cfg)
    mu_v = cfg.lambda_c
    neighbors = neighbors or []
    nu = cfg.temporal_strength if neighbors else 0.0
    eps_t = cfg.temporal_eps
    if nu > 0.0 and eps_t <= 0.0:
        eps_t = max(1.0, 7.0 * estimate_noise_ire(S))
    nu_a = cfg.nr_anchor if anchor is not None else 0.0
    if anchor is not None:
        Y_hat, chi_hat, w_a = anchor
    nbr = list(neighbors)

    Y = Y0.copy()
    chi = chi0.copy()
    n_inner = max(1, cfg.cg_iterations // max(1, irls_outer))

    for _outer in range(irls_outer):
        gxY, gyY = _dx(Y), _dy(Y)
        wx = eps / np.sqrt(gxY ** 2 + eps ** 2)
        wy = eps / np.sqrt(gyY ** 2 + eps ** 2)
        cx, cy = np.abs(_dx(chi)), np.abs(_dy(chi))
        kc = cfg.structure_coupling
        wcx = eps_c / np.sqrt(cx ** 2 + kc * gxY ** 2 + eps_c ** 2)
        wcy = eps_c / np.sqrt(cy ** 2 + kc * gyY ** 2 + eps_c ** 2)
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
            if gg < cfg.cg_tol ** 2 * gg0 or gg < 1e-10 * S.size:
                break

    return Y, chi
