#!/usr/bin/env python3
"""
fixed cross（primary 経路）の全域 Eq 残差 ||A - T~||_F を、全域評価なしに Gaussian probe で推定する試験（2026-10-06）。
  A = sum_i a_i(qx) d_i(qz) b_i(qy) m_i(x,z,y)  … dense moment から O(1) で sketch できる
  T~ = fixed cross の TT                      … TT 縮約で sketch できる
  推定 = ||(A - T~) Om'||_F / sqrt(s)、Om' は構築と独立な seed。
  比較: 厳密残差（全域実体化）。probe seed を変えて推定/厳密の分布（min/median/max）と時間を記録する。
  対象: (a) 前回 Eq 履歴 pivot の cross（合格ケース）、(b) 前回 F rank 25/16 pivot の cross（不合格ケース）。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import toy_moment_hybrid_timing as T
from toy_cross_oversampling import tt_svd_capped
Nx, Ny, Nz = T.Nx, T.Ny, T.Nz


def sketch_A(mf, Abas, Dbas, Bbas, Om):
    """(A Om)[(x,qx), s] を dense moment から。Om: (Nz,3,Ny,3,s)"""
    nI = mf.shape[0]; s = Om.shape[-1]
    M = mf.reshape(nI, Nx, Nz * Ny)
    DB = np.einsum("iq,ip->iqp", Dbas, Bbas).reshape(nI, 9)
    Om_r = Om.transpose(0, 2, 1, 3, 4).reshape(Nz * Ny, 9, s)
    Omt = np.einsum("ir,srl->isl", DB, Om_r)
    Yi = M @ Omt
    return np.einsum("iq,ixl->xql", Abas, Yi).reshape(Nx * 3, s)


def sketch_TT(cores, Om):
    """(T~ Om)[(x,qx), s] を TT 縮約で。"""
    G1, G2, G3 = cores
    r1, r2 = G1.shape[2], G2.shape[2]; s = Om.shape[-1]
    Om2 = Om.reshape(Nz * 3, Ny * 3, s)
    W = np.einsum("bm,nms->bns", G3.reshape(r2, Ny * 3), Om2)        # (r2, n2, s)
    V = np.einsum("anb,bns->as", G2, W)                                # (r1, s)
    return G1.reshape(Nx * 3, r1) @ V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--current-shift", type=float, default=10.0)
    ap.add_argument("--shift", type=float, default=9.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="docs/toy_residual_probe_results.json")
    args = ap.parse_args()
    Abas, Dbas, Bbas = T.direction_basis()
    rho, u = T.build_fields(args.eps, shift_steps=args.current_shift)
    mf = T.coefficient_fields(rho.transpose(0, 2, 1), tuple((rho * uu).transpose(0, 2, 1) for uu in u))
    Tex = T.exact_feq_paired(mf, Abas, Dbas, Bbas); nTex = np.linalg.norm(Tex)
    eq_tol_rel = 1.6288e-8; budget_eq = 0.5e-9 * nTex / T.OMEGA
    rho_p, u_p = T.build_fields(args.eps, shift_steps=args.shift)
    mf_p = T.coefficient_fields(rho_p.transpose(0, 2, 1), tuple((rho_p * uu).transpose(0, 2, 1) for uu in u_p))
    Tex_p = T.exact_feq_paired(mf_p, Abas, Dbas, Bbas)
    f_p = T.equilibrium(rho_p, u_p) + T.nonequilibrium_CE(rho_p, u_p); fp = T.to_paired(f_p); del f_p
    cases = {
        "prevEq_pivots": T.pivots_from_tt(T.tt_svd(Tex_p, (Nx * 3, Nz * 3, Ny * 3), budget_eq)),
        "prevF_rank25_16_pivots": T.pivots_from_tt(tt_svd_capped(fp, (Nx * 3, Nz * 3, Ny * 3), 1e-8 * np.linalg.norm(fp), (25, 16))),
    }
    del Tex_p, fp
    res = dict(config=vars(args), eq_tol_rel=eq_tol_rel, cases=[])
    for label, piv in cases.items():
        cores, nraw, conds = T.cross_feq(mf, Abas, Dbas, Bbas, piv)
        exact = np.linalg.norm(T.tt_full(cores) - Tex.reshape(-1))
        for s in (8, 16):
            ratios = []; t_probe = []
            for sd in range(args.seeds):
                rng = np.random.default_rng(2026100602 + sd)
                t0 = time.perf_counter()
                Om = rng.standard_normal((Nz, 3, Ny, 3, s))
                R = sketch_A(mf, Abas, Dbas, Bbas, Om) - sketch_TT(cores, Om)
                est = np.linalg.norm(R) / np.sqrt(s)
                t_probe.append(time.perf_counter() - t0)
                ratios.append(est / exact)
            ratios = np.array(ratios)
            rec = dict(case=label, ranks=[len(piv["I1"]), len(piv["J2"])], exact_rel=float(exact / nTex), exact_passed=bool(exact / nTex <= eq_tol_rel),
                       probe_cols=s, seeds=args.seeds, ratio_min=float(ratios.min()), ratio_median=float(np.median(ratios)), ratio_max=float(ratios.max()),
                       ratio_std=float(ratios.std()), probe_seconds_median=float(np.median(t_probe)),
                       frac_seeds_underestimating_more_than_20pct=float((ratios < 0.8).mean()))
            res["cases"].append(rec); print(json.dumps(rec), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
