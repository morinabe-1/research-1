#!/usr/bin/env python3
"""
構造付き randomized TT-SVD（fallback 候補）の、sheared 場での精度・事後台帳・時間の確認（2026-10-06、レビュー 4.2/7 対応）。
  場: phase0 (Eq rank 26/10) と current-shift 10 の sheared 場 (Eq rank 43/13)
  設定: k ∈ {40, 58}, power ∈ {1, 2}
  記録: 関数単体時間、出力 rank、厳密全域残差 / ||Eq||、事後台帳（射影残差推定 + tail）、台帳が厳密残差を覆うか
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import toy_moment_hybrid_timing as T
Nx, Ny, Nz = T.Nx, T.Ny, T.Nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--out", default="docs/toy_randomized_fallback_results.json")
    args = ap.parse_args()
    Abas, Dbas, Bbas = T.direction_basis()
    eq_tol_rel = 1.6288e-8
    res = dict(config=dict(eps=args.eps, eq_tol_rel=eq_tol_rel), cases=[])
    for fld_label, shift in [("phase0", 0.0), ("sheared_shift10", 10.0)]:
        rho, u = T.build_fields(args.eps, shift_steps=shift)
        mf = T.coefficient_fields(rho.transpose(0, 2, 1), tuple((rho * uu).transpose(0, 2, 1) for uu in u))
        Tex = T.exact_feq_paired(mf, Abas, Dbas, Bbas); nTex = np.linalg.norm(Tex)
        ref = T.tt_svd(Tex, (Nx * 3, Nz * 3, Ny * 3), eq_tol_rel * nTex)
        ref_ranks = [c.shape[2] for c in ref[:-1]]
        tol_abs = 0.5 * eq_tol_rel * nTex       # Eq 許容の半分を構築に配分（残りは射影残差の不確かさへ）
        for k in (40, 58):
            for power in (1, 2):
                rng = np.random.default_rng(1)
                t0 = time.perf_counter()
                cores, diag = T.structured_tt_svd(mf, Abas, Dbas, Bbas, tol_abs, k=k, power=power, rng=rng, return_diag=True)
                dt = time.perf_counter() - t0
                exact = np.linalg.norm(T.tt_full(cores) - Tex.reshape(-1))
                rec = dict(field=fld_label, tt_svd_ref_ranks=ref_ranks, k=k, power=power, seconds=dt, ranks=diag["ranks"],
                           exact_rel_err=float(exact / nTex), exact_passed=bool(exact / nTex <= eq_tol_rel),
                           tail_only_rel=float(diag["tail_only"] / nTex),
                           projection_residual_estimate_rel=float(diag["projection_residual_estimate"] / nTex),
                           ledger_total_estimate_rel=float(diag["ledger_total_estimate"] / nTex),
                           ledger_covers_exact=bool(diag["ledger_total_estimate"] >= exact),
                           tail_only_covers_exact=bool(diag["tail_only"] >= exact))
                res["cases"].append(rec); print(json.dumps(rec), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
