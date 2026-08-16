#!/usr/bin/env python3
"""End-to-end self-tests for the PAL research decoder.

Full suite:      python3 run_tests.py            (~15-20 min, CPU)
Chosen sections: python3 run_tests.py 6b 6d 9

Sections: 1 adjoints | 2 burst lock-in | 3 round-trip vs delay-line |
4 differential phase | 5 3D static | 6 3D motion | 6b thin detail |
6c chunked streaming | 6d drizzle | 7 aniso+symmetry | 8 polsar |
9 anchor loop
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hvd_pal.tbc import TbcSource, VideoParameters
from hvd_pal import encode as enc
from hvd_pal import decoder as dec

ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and cond

rng = np.random.default_rng(0)

def make_source(tmp, **kw):
    p = VideoParameters()
    img = enc.make_test_pattern(700, 560, "ebu")
    path = os.path.join(tmp, "t.tbc")
    enc.write_tbc(path, [img], p, **kw)
    return TbcSource.open(path), img, p

def ref_yuv(img, shape):
    Hh, Ww = img.shape[:2]
    ys = (np.arange(shape[0]) * Hh // shape[0]).clip(0, Hh - 1)
    xs = (np.arange(shape[1]) * Ww // shape[1]).clip(0, Ww - 1)
    f = img[ys][:, xs] @ enc.RGB_TO_YUV.T
    Y = enc.IRE_BLACK + f[..., 0] * 100.0
    U = f[..., 1] * 100.0
    V = f[..., 2] * 100.0
    return Y, U, V

def psnr(a, b, peak=100.0):
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(peak * peak / max(mse, 1e-12))

def ref_field(im, fld, shape):
    Hh, Ww = im.shape[:2]
    ys = (np.arange(2*shape[0])*Hh//(2*shape[0])).clip(0, Hh-1)[fld::2]
    xs = (np.arange(shape[1])*Ww//shape[1]).clip(0, Ww-1)
    f = im[ys][:, xs] @ enc.RGB_TO_YUV.T
    return f[..., 0]*100.0, f[..., 1]*100.0, f[..., 2]*100.0

def weave_ref(g1, g2, shape):
    h = shape[0]//2
    Yt = np.empty(shape); Ut = np.empty(shape); Vt = np.empty(shape)
    a = ref_field(g1, 0, (h, shape[1])); b = ref_field(g2, 1, (h, shape[1]))
    Yt[0::2], Yt[1::2] = a[0], b[0]
    Ut[0::2], Ut[1::2] = a[1], b[1]
    Vt[0::2], Vt[1::2] = a[2], b[2]
    return Yt, Ut, Vt

def shifted(im, px):
    return np.roll(im, int(px), axis=1)

base = enc.make_test_pattern(760, 560, "ebu")
SHAPE = (582, 922)

SECTIONS = {}
def section(name):
    def deco(fn):
        SECTIONS[name] = fn
        return fn
    return deco

@section("1")
def _sec_1():
    # ---- 1. adjoints -----------------------------------------------------
    a = rng.normal(size=(64, 80)); b = rng.normal(size=(64, 80))
    e1 = abs(np.sum(dec._dx(a) * b) - np.sum(a * dec._dxT(b)))
    e2 = abs(np.sum(dec._dy(a) * b) - np.sum(a * dec._dyT(b)))
    check("adjoint Dx/DxT, Dy/DyT", max(e1, e2) < 1e-9, f"err={max(e1,e2):.2e}")

@section("2")
def _sec_2():
    # ---- 2. burst lock-in ------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        src, img, p = make_source(tmp, noise_ire=0.8)
        S0 = p.ire(src.read_field(0))
        th, s, amp = dec.burst_lockin(S0, p)
        # ground truth for field phase_id=1: voff=0 -> s=+1 on even lines
        s_true = np.where(np.arange(p.field_height) % 2 == 0, 1.0, -1.0)
        th_true = enc._phase_origin(1) + dec.LINE_ADV * np.arange(p.field_height)
        perr = np.abs(np.angle(np.exp(1j * (th - th_true)))).max()
        check("swinging-burst parity (clean)", np.all(s == s_true))
        check("swinging-burst phase (clean)", perr < 0.02, f"max err={np.degrees(perr):.3f} deg")

    with tempfile.TemporaryDirectory() as tmp:
        src, img, p = make_source(tmp, noise_ire=0.8, burst_dropout=0.25)
        S0 = p.ire(src.read_field(0))
        th, s, amp = dec.burst_lockin(S0, p)
        s_true = np.where(np.arange(p.field_height) % 2 == 0, 1.0, -1.0)
        th_true = enc._phase_origin(1) + dec.LINE_ADV * np.arange(p.field_height)
        perr = np.abs(np.angle(np.exp(1j * (th - th_true)))).max()
        check("parity under 25% burst dropout", np.mean(s == s_true) == 1.0)
        check("phase under 25% burst dropout", perr < 0.06, f"max err={np.degrees(perr):.2f} deg")

@section("3")
def _sec_3():
    # ---- 3. round-trip PSNR vs delay-line --------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        src, img, p = make_source(tmp, noise_ire=0.8)
        cfg = dec.DecoderConfig()
        S, carrier, gain = dec.prepare_frame(src, 0, cfg)
        Yt, Ut, Vt = ref_yuv(img, S.shape)

        Yd, chid = dec.delayline_baseline(S, carrier, p, cfg)
        Y0, chi0 = dec.holographic_init(S, carrier, p, cfg)
        Yr, chir = dec.variational_refine(S, carrier, Y0, chi0, cfg)

        def score(Y, chi):
            U = -np.imag(chi); V = np.real(chi)
            return psnr(Y, Yt), psnr(np.stack([U, V]), np.stack([Ut, Vt]))
        pd = score(Yd, chid); ph = score(Y0, chi0); pv = score(Yr, chir)
        print(f"  delay-line     Y {pd[0]:5.1f} dB  C {pd[1]:5.1f} dB")
        print(f"  holo init      Y {ph[0]:5.1f} dB  C {ph[1]:5.1f} dB")
        print(f"  holo+variational Y {pv[0]:5.1f} dB  C {pv[1]:5.1f} dB")
        check("beats delay-line (luma)", pv[0] > pd[0] + 0.5)
        check("beats delay-line (chroma)", pv[1] > pd[1] + 0.5)
        check("refine >= init", pv[0] >= ph[0] - 0.1 and pv[1] >= ph[1] - 0.1)
        # purity: delivered pair reconstructs S exactly
        recon = Yr + np.real(chir * carrier)
        check("purity contract (lossless split)", np.abs(recon - S).max() < 1e-9)

@section("4")
def _sec_4():
    # ---- 4. differential phase torture (the PAL raison d'etre) -----------
    with tempfile.TemporaryDirectory() as tmp:
        DELTA = 10.0
        src, img, p = make_source(tmp, noise_ire=0.4, diff_phase_deg=DELTA)
        cfg = dec.DecoderConfig()
        S, carrier, gain = dec.prepare_frame(src, 0, cfg)
        Yt, Ut, Vt = ref_yuv(img, S.shape)
        Y0, chi0 = dec.holographic_init(S, carrier, p, cfg)
        Yr, chir = dec.variational_refine(S, carrier, Y0, chi0, cfg)
        U = -np.imag(chir); V = np.real(chir)

        # measure on the saturated top bars only
        sat = np.hypot(Ut, Vt) > 15.0
        hue_err = np.angle(np.exp(1j * (np.arctan2(Vt, Ut) - np.arctan2(V, U))))
        med_hue = np.degrees(np.median(np.abs(hue_err[sat])))
        sat_ratio = np.median(np.hypot(U, V)[sat] / np.hypot(Ut, Vt)[sat])
        # Hanover-bar energy: line-alternating hue component
        hue = np.arctan2(V, U)
        alt = np.abs(np.angle(np.exp(1j * (hue[1:] - hue[:-1]))))
        hanover = np.degrees(np.median(alt[sat[1:] & sat[:-1]]))
        print(f"  {DELTA:.0f} deg differential phase -> hue err {med_hue:.2f} deg, "
              f"saturation {100*sat_ratio:.1f}%, line-alt hue {hanover:.2f} deg")
        check("hue error converted to saturation (PAL guarantee)",
              med_hue < DELTA / 4.0, f"({med_hue:.2f} < {DELTA/4:.1f} deg)")
        check("no Hanover bars", hanover < 3.0, f"({hanover:.2f} deg line-alt)")
        check("saturation loss ~= cos(delta)",
              abs(sat_ratio - np.cos(np.deg2rad(DELTA))) < 0.06,
              f"({sat_ratio:.3f} vs {np.cos(np.deg2rad(DELTA)):.3f})")

@section("5")
def _sec_5():
    # ---- 5. 3D static gain (same-parity equations f+/-2, f+/-4) ----------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        path = os.path.join(tmp, "s.tbc")
        enc.write_tbc(path, [base[:, 30:730]]*4, p, noise_ire=3.0,
                      rng=np.random.default_rng(5))
        src = TbcSource.open(path)
        Yt, Ut, Vt = weave_ref(base[:, 30:730], base[:, 30:730], SHAPE)
        Y2, U2, V2 = next(dec.decode_sequence(src, 1, 1,
                          dec.DecoderConfig(temporal_strength=0.0)))
        Y3, U3, V3 = next(dec.decode_sequence(src, 1, 1,
                          dec.DecoderConfig(temporal_strength=0.5)))
        s2 = (psnr(Y2, Yt), psnr(np.stack([U2, V2]), np.stack([Ut, Vt])))
        s3 = (psnr(Y3, Yt), psnr(np.stack([U3, V3]), np.stack([Ut, Vt])))
        print(f"  static 3 IRE: 2D Y {s2[0]:.1f} C {s2[1]:.1f} | "
              f"3D Y {s3[0]:.1f} C {s3[1]:.1f}")
        check("3D static gain (Y)", s3[0] > s2[0] + 1.5,
              f"(+{s3[0]-s2[0]:.1f} dB)")
        check("3D static gain (C)", s3[1] > s2[1] + 1.0,
              f"(+{s3[1]-s2[1]:.1f} dB)")

@section("6")
def _sec_6():
    # ---- 6. 3D motion safety (5 px/frame pan, true interlace) ------------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        path = os.path.join(tmp, "m.tbc")
        frames = [(shifted(base, 5*f)[:, 30:730], shifted(base, 5*f+2)[:, 30:730])
                  for f in range(4)]
        enc.write_tbc(path, frames, p, noise_ire=1.5,
                      rng=np.random.default_rng(9))
        src = TbcSource.open(path)
        Yt, Ut, Vt = weave_ref(shifted(base, 5)[:, 30:730],
                               shifted(base, 7)[:, 30:730], SHAPE)
        Y2, U2, V2 = next(dec.decode_sequence(src, 1, 1,
                          dec.DecoderConfig(temporal_strength=0.0)))
        Y3, U3, V3 = next(dec.decode_sequence(src, 1, 1,
                          dec.DecoderConfig(temporal_strength=0.5)))
        m2 = (psnr(Y2, Yt), psnr(np.stack([U2, V2]), np.stack([Ut, Vt])))
        m3 = (psnr(Y3, Yt), psnr(np.stack([U3, V3]), np.stack([Ut, Vt])))
        print(f"  pan 5px/fr:   2D Y {m2[0]:.1f} C {m2[1]:.1f} | "
              f"3D Y {m3[0]:.1f} C {m3[1]:.1f}")
        check("3D never worse on motion (Y)", m3[0] > m2[0] - 0.2,
              f"({m3[0]-m2[0]:+.1f} dB)")
        check("3D never worse on motion (C)", m3[1] > m2[1] - 0.2,
              f"({m3[1]-m2[1]:+.1f} dB)")

@section("6b")
def _sec_6b():
    # ---- 6b. thin horizontal detail (the NTSC 9h torture, on PAL) --------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        thin = base.copy()
        for r0 in range(120, 480, 40):
            thin[r0:r0+1, 100:660] = [0.1, 0.7, 0.2]
        path = os.path.join(tmp, "th.tbc")
        enc.write_tbc(path, [thin[:, 30:730]]*6, p, noise_ire=1.5,
                      rng=np.random.default_rng(3))
        src = TbcSource.open(path)
        Yt, Ut, Vt = weave_ref(thin[:, 30:730], thin[:, 30:730], SHAPE)
        Y2, U2, V2 = next(dec.decode_sequence(src, 2, 1,
                          dec.DecoderConfig(temporal_strength=0.0)))
        Y3, U3, V3 = next(dec.decode_sequence(src, 2, 1,
                          dec.DecoderConfig(temporal_strength=0.5)))
        t2 = (psnr(Y2, Yt), psnr(np.stack([U2, V2]), np.stack([Ut, Vt])))
        t3 = (psnr(Y3, Yt), psnr(np.stack([U3, V3]), np.stack([Ut, Vt])))
        print(f"  thin 1-line detail: 2D Y {t2[0]:.1f} C {t2[1]:.1f} | "
              f"3D-full Y {t3[0]:.1f} C {t3[1]:.1f}")
        check("3D full offsets safe on thin detail (Y)", t3[0] > t2[0] + 2.0,
              f"(+{t3[0]-t2[0]:.1f} dB — no NTSC-9h regression)")
        check("3D full offsets safe on thin detail (C)", t3[1] > t2[1] + 2.0,
              f"(+{t3[1]-t2[1]:.1f} dB)")

@section("6c")
def _sec_6c():
    # ---- 6c. chunked streaming equivalence -------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        path = os.path.join(tmp, "c.tbc")
        frames = [(shifted(base, 5*f)[:, 30:730],
                   shifted(base, 5*f+2)[:, 30:730]) for f in range(5)]
        enc.write_tbc(path, frames, p, noise_ire=1.5,
                      rng=np.random.default_rng(9))
        src = TbcSource.open(path)
        cA = dec.DecoderConfig(temporal_strength=0.5, cg_iterations=30,
                               chunk_frames=10)
        cB = dec.DecoderConfig(temporal_strength=0.5, cg_iterations=30,
                               chunk_frames=2)
        outA = [Y for Y, U, V in dec.decode_sequence(src, 1, 3, cA)]
        outB = [Y for Y, U, V in dec.decode_sequence(src, 1, 3, cB)]
        d = max(np.abs(a - b).max() for a, b in zip(outA, outB))
        check("chunked streaming bit-exact (pass 1)", d < 1e-9,
              f"(max diff {d:.1e} IRE)")

@section("6d")
def _sec_6d():
    # ---- 6d. drizzle: static vertical stacking ---------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        path = os.path.join(tmp, "dz.tbc")
        enc.write_tbc(path, [base[:, 30:730]]*5, p, noise_ire=3.0,
                      rng=np.random.default_rng(5))
        src = TbcSource.open(path)
        Yd, Ud, Vd = next(dec.decode_sequence(src, 2, 1,
            dec.DecoderConfig(temporal_strength=0.5, drizzle=True,
                              cg_iterations=40)))
        Yw, Uw, Vw = next(dec.decode_sequence(src, 2, 1,
            dec.DecoderConfig(temporal_strength=0.5, cg_iterations=40)))
        check("drizzle output geometry 2x", Yd.shape[0] == 2*Yw.shape[0])
        yuv = base[:, 30:730] @ enc.RGB_TO_YUV.T
        Ybt = yuv[..., 0]*100.0
        ys = (np.arange(Yd.shape[0]) * Ybt.shape[0] // Yd.shape[0]).clip(
            0, Ybt.shape[0]-1)
        xs = (np.arange(Yd.shape[1]) * Ybt.shape[1] // Yd.shape[1]).clip(
            0, Ybt.shape[1]-1)
        Yfine = Ybt[ys][:, xs]
        fine = np.arange(Yd.shape[0]) / 2.0
        f0 = np.clip(np.floor(fine).astype(int), 0, Yw.shape[0]-2)
        ff = (fine - f0)[:, None]
        Yup = Yw[f0]*(1-ff) + Yw[f0+1]*ff
        sl = (slice(20, Yd.shape[0]-20), slice(20, Yd.shape[1]-20))
        pu, pd = psnr(Yup[sl], Yfine[sl]), psnr(Yd[sl], Yfine[sl])
        print(f"  drizzle static: weave-upscale {pu:.1f} dB, drizzle {pd:.1f} dB")
        check("drizzle beats weave-upscale (static)", pd > pu + 1.0,
              f"(+{pd-pu:.1f} dB)")

@section("7")
def _sec_7():
    # ---- 7. auto aniso + symmetry variant (subsumption check) ------------
    with tempfile.TemporaryDirectory() as tmp:
        src, img, p = make_source(tmp, noise_ire=0.8)
        cfg = dec.DecoderConfig()
        S, carrier, gain = dec.prepare_frame(src, 0, cfg)
        Yt, Ut, Vt = ref_yuv(img, S.shape)
        a = dec._resolve_chroma_aniso(
            dec.holographic_init(S, carrier, p, cfg)[1], cfg)
        check("auto aniso in range", 0.55 <= a <= 1.0, f"(a={a:.2f})")
        res = {}
        for name in ("plain", "symmetry"):
            c = dec.DecoderConfig(symmetry_variant=(name == "symmetry"))
            Y0, chi0 = dec.holographic_init(S, carrier, p, c)
            Yr, chir = dec.variational_refine(S, carrier, Y0, chi0, c)
            res[name] = psnr(np.stack([-np.imag(chir), np.real(chir)]),
                             np.stack([Ut, Vt]))
        d = res["symmetry"] - res["plain"]
        print(f"  Transform-PAL certifier variant: {d:+.2f} dB chroma")
        check("symmetry variant measured (recorded neutral)", abs(d) < 0.5,
              f"({d:+.2f} dB — subsumption holds on PAL as on NTSC)")

@section("8")
def _sec_8():
    # ---- 8. PolSAR cross-pol diagnostic map ------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        med = {}
        for dp in (0.0, 10.0):
            path = os.path.join(tmp, f"d{int(dp)}.tbc")
            enc.write_tbc(path, [base[:, 30:730]], p, noise_ire=0.4,
                          diff_phase_deg=dp)
            s3 = TbcSource.open(path)
            S, carrier, _ = dec.prepare_frame(s3, 0, dec.DecoderConfig())
            co, cx = dec.polarimetric_maps(S, carrier, s3.params,
                                           dec.DecoderConfig())
            med[dp] = float(np.median(cx[co > 15.0]))
        print(f"  cross-pol |chi_x|: clean {med[0.0]:.2f} IRE, "
              f"10 deg diff-phase {med[10.0]:.2f} IRE")
        check("PolSAR map discriminates distortion",
              med[10.0] > 8.0 * med[0.0],
              f"({med[10.0]/max(med[0.0],1e-6):.0f}x separation)")

@section("9")
def _sec_9():
    # ---- 9. re-encode anchor loop (passes >= 2) --------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = VideoParameters()
        path = os.path.join(tmp, "a.tbc")
        enc.write_tbc(path, [base[:, 30:730]]*6, p, noise_ire=3.0,
                      rng=np.random.default_rng(5))
        src = TbcSource.open(path)
        Yt, Ut, Vt = weave_ref(base[:, 30:730], base[:, 30:730], SHAPE)
        r = {}
        # acc=False throughout: ACC is a pure output gain (neutral on this
        # unit-level source) and would otherwise scale U/V away from the
        # exact chi the purity identity is stated in
        for name, cfg in (
                ("p1", dec.DecoderConfig(temporal_strength=0.5, acc=False)),
                ("p2", dec.DecoderConfig(temporal_strength=0.5, passes=2,
                                         acc=False)),
                ("p2s", dec.DecoderConfig(temporal_strength=0.5, passes=2,
                                          output_fidelity=False, acc=False))):
            Y, U, V = next(dec.decode_sequence(src, 2, 1, cfg))
            r[name] = (psnr(Y, Yt), psnr(np.stack([U, V]), np.stack([Ut, Vt])),
                       (Y, U, V))
        print(f"  anchor loop (3 IRE): 1-pass Y {r['p1'][0]:.1f} C {r['p1'][1]:.1f}"
              f" | 2-pass purist Y {r['p2'][0]:.1f} C {r['p2'][1]:.1f}"
              f" | 2-pass soft Y {r['p2s'][0]:.1f} C {r['p2s'][1]:.1f}")
        check("anchor loop chroma gain (purist, no output NR)",
              r["p2"][1] > r["p1"][1] + 2.0,
              f"(+{r['p2'][1]-r['p1'][1]:.1f} dB)")
        check("anchor loop soft-output luma gain",
              r["p2s"][0] > r["p1"][0] + 2.0,
              f"(+{r['p2s'][0]-r['p1'][0]:.1f} dB)")
        # purist purity: woven output must reconstruct the woven composite
        Sf, carrier, _ = dec.prepare_frame(src, 2, dec.DecoderConfig(acc=False))
        Yp, Up, Vp = r["p2"][2]
        chi_p = Vp + 1j * (-Up)
        recon = Yp + np.real(chi_p * carrier)
        check("purist output still splits S losslessly",
              np.abs(recon - Sf).max() < 1e-6,
              f"(max err {np.abs(recon - Sf).max():.1e} IRE)")


if __name__ == "__main__":
    want = sys.argv[1:] or list(SECTIONS.keys())
    for name in want:
        SECTIONS[name]()
    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    sys.exit(0 if ok else 1)
