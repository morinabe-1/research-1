#!/usr/bin/env python3
"""
固定 pivot cross の精度を、pivot の出自と過剰標本化で比べる合成場試験（2026-10-06）。

研究側 rev2 では、前回受理 F1（rank 25/16）の pivot で現在の Eq を fixed cross した結果、
面 y2 の population 相対誤差 3.06e-8 が velocity guard 2e-5 を 3.33e-5 で超え、
two-site warm TCI（9.4 秒、11.46M 要求）へ退避した。ここでは合成場で
  (a) 近傍状態の完全 f（非平衡込み）を rank (25,16) に打ち切った TT の pivot  … rev2 の状況を模す
  (b) 同じ f を (25+os, 16+os/2) に打ち切った TT の pivot を使い、pivot 行列を pinv で解く … 過剰標本化
  (c) 現在の Eq 自身の TT-SVD の pivot … 上限性能
について Eq cross の全域誤差、面 y2 の population/velocity 相対誤差、時間を測る。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import toy_moment_hybrid_timing as T

Nx, Ny, Nz = T.Nx, T.Ny, T.Nz


def tt_svd_capped(X, dims, tol_abs, caps):
    d = len(dims); eps = tol_abs / np.sqrt(max(d - 1, 1))
    M = np.ascontiguousarray(X).reshape(dims[0], -1); r = 1; cores = []
    for k in range(d - 1):
        M = M.reshape(r * dims[k], -1)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        rn = len(s)
        while rn > 1 and tail[rn - 1] <= eps:
            rn -= 1
        rn = min(rn, caps[k])
        cores.append(U[:, :rn].reshape(r, dims[k], rn)); M = s[:rn, None] * Vt[:rn]; r = rn
    cores.append(M.reshape(r, dims[-1], 1))
    return cores


def cross_feq_pinv(mf, Abas, Dbas, Bbas, piv, rcond):
    """T.cross_feq と同じ構成だが、pivot 行列を pinv(rcond) で解く。"""
    I1 = piv["I1"]; I2p = piv["I2_pairs"]; J2 = piv["J2"]; J1p = piv["J1_pairs"]
    r1, r2 = len(I1), len(J2)
    x1, qx1 = np.unravel_index(I1, (Nx, 3)); y2, qy2 = np.unravel_index(J2, (Ny, 3))
    n2_J1 = J1p[:, 0]; b_J1 = J1p[:, 1]
    z1, qz1 = np.unravel_index(n2_J1, (Nz, 3)); yJ1, qyJ1 = y2[b_J1], qy2[b_J1]
    a_I2 = I2p[:, 0]; n2_I2 = I2p[:, 1]
    xI2, qxI2 = x1[a_I2], qx1[a_I2]; z2, qz2 = np.unravel_index(n2_I2, (Nz, 3))
    coef1 = Dbas[:, qz1] * Bbas[:, qyJ1]
    C1 = np.einsum("ij,iq,ixj->xqj", coef1, Abas, mf[:, :, z1, yJ1]).reshape(Nx * 3, r1)
    P1 = C1[I1]
    C2 = np.einsum("ia,ib,iazb,iq->azqb", Abas[:, qx1], Bbas[:, qy2], mf[:, x1][:, :, :, y2], Dbas).reshape(r1, Nz * 3, r2)
    P2 = C2.reshape(r1 * Nz * 3, r2)[np.ravel_multi_index((a_I2, n2_I2), (r1, Nz * 3))]
    coef3 = Abas[:, qxI2] * Dbas[:, qz2]
    C3 = np.einsum("ic,icy,ip->cyp", coef3, mf[:, xI2, z2, :], Bbas).reshape(r2, Ny * 3)
    core1 = (C1 @ np.linalg.pinv(P1, rcond=rcond)).reshape(1, Nx * 3, r1)
    core2 = (C2.reshape(-1, r2) @ np.linalg.pinv(P2, rcond=rcond)).reshape(r1, Nz * 3, r2)
    core3 = C3.reshape(r2, Ny * 3, 1)
    return [core1, core2, core3], C1.size + C2.size + C3.size, (np.linalg.cond(P1), np.linalg.cond(P2))


def plane_metrics(feq_tt, Tex_paired, yidx):
    """面 yidx の population 相対 L2、および f* 相当の velocity 相対 L2（Eq 誤差のみ、ω 倍）。"""
    G1, G2, G3 = feq_tt
    r1, r2 = G1.shape[2], G2.shape[2]
    g3 = G3.reshape(r2, Ny, 3)[:, yidx, :]
    T12 = (G1.reshape(Nx * 3, r1) @ G2.reshape(r1, Nz * 3 * r2)).reshape(Nx * 3 * Nz * 3, r2)
    pl_tt = (T12 @ g3).reshape(Nx, 3, Nz, 3, 3)                         # (x,qx,z,qz,qy)
    pl_ex = Tex_paired.reshape(Nx, 3, Nz, 3, Ny, 3)[:, :, :, :, yidx, :]
    pop_rel = np.linalg.norm(pl_tt - pl_ex) / np.linalg.norm(pl_ex)
    # moment: rho, j from (qx,qz,qy)
    def mom(pl):
        rho = pl.sum(axis=(1, 3, 4))
        jx = np.einsum("xqzrp,q->xz", pl, T.cax); jz = np.einsum("xqzrp,r->xz", pl, T.cax); jy = np.einsum("xqzrp,p->xz", pl, T.cax)
        return rho, np.stack([jx, jy, jz], -1)
    rho_e, j_e = mom(pl_ex); rho_t, j_t = mom(pl_ex + T.OMEGA * (pl_tt - pl_ex))   # f* の誤差は ω 倍の Eq 誤差
    u_e = j_e / rho_e[..., None]; u_t = j_t / rho_t[..., None]
    vel_rel = np.linalg.norm(u_t - u_e) / np.linalg.norm(u_e)
    return pop_rel, vel_rel, np.linalg.norm(u_e) / np.linalg.norm(pl_ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--shift", type=float, default=10.0, help="前回状態の位相進み（ステップ数）")
    ap.add_argument("--current-shift", type=float, default=0.0, help="現在状態の位相進み（ステップ数）。0 なら phase0 の場")
    ap.add_argument("--out", default="docs/toy_cross_oversampling_results.json")
    args = ap.parse_args()
    Abas, Dbas, Bbas = T.direction_basis()

    # 現在状態
    rho, u = T.build_fields(args.eps, shift_steps=args.current_shift)
    feq = T.equilibrium(rho, u); f = feq + T.nonequilibrium_CE(rho, u)
    fstar = (1 - T.OMEGA) * f + T.OMEGA * feq; nf = np.linalg.norm(fstar)
    mf = T.coefficient_fields(rho.transpose(0, 2, 1), tuple((rho * uu).transpose(0, 2, 1) for uu in u))
    Tex = T.exact_feq_paired(mf, Abas, Dbas, Bbas); nTex = np.linalg.norm(Tex)
    budget_eq = 0.5e-9 * nf / T.OMEGA
    eq_tol_rel = 1.6288e-8     # 研究側 normalizer の Eq 相対許容
    # 近傍状態（前回受理 F1 を模す）: 完全 f、1e-8 で打ち切り、rank cap
    rho_p, u_p = T.build_fields(args.eps, shift_steps=args.shift)
    f_p = T.equilibrium(rho_p, u_p) + T.nonequilibrium_CE(rho_p, u_p)
    fp_paired = T.to_paired(f_p); nfp = np.linalg.norm(fp_paired)
    del f_p
    res = dict(config=dict(eps=args.eps, shift=args.shift, current_shift=args.current_shift, eq_tol_rel=eq_tol_rel), cases=[])

    def run_case(label, cores_for_pivots, solver, rcond=None):
        piv = T.pivots_from_tt(cores_for_pivots)
        t0 = time.perf_counter()
        if solver == "solve":
            tt, nraw, conds = T.cross_feq(mf, Abas, Dbas, Bbas, piv)
        else:
            tt, nraw, conds = cross_feq_pinv(mf, Abas, Dbas, Bbas, piv, rcond)
        dt = time.perf_counter() - t0
        err = np.linalg.norm(T.tt_full(tt) - Tex.reshape(-1))
        p2, v2, uscale2 = plane_metrics(tt, Tex, 2)
        p31, v31, _ = plane_metrics(tt, Tex, 31)
        rec = dict(label=label, pivot_ranks=[len(piv["I1"]), len(piv["J2"])], n_raw=int(nraw), seconds=dt,
                   cond_P1=float(conds[0]), cond_P2=float(conds[1]),
                   cond_cap_1e12_passed=bool(max(conds) <= 1e12),
                   eq_rel_err=float(err / nTex), eq_rel_tol_passed=bool(err / nTex <= eq_tol_rel),
                   both_passed=bool(err / nTex <= eq_tol_rel and max(conds) <= 1e12),
                   face_y2_pop_rel=float(p2), face_y2_vel_rel_fstar=float(v2), face_y31_pop_rel=float(p31), face_y31_vel_rel_fstar=float(v31),
                   face_y2_vel_amplification=float(v2 / (T.OMEGA * p2)) if p2 > 0 else None)
        res["cases"].append(rec); print(json.dumps(rec), flush=True)

    # (a) rev2 の状況: 前回 F（完全 f、近傍状態）の pivot、rank cap (25,16)、通常 solve
    prevF_25_16 = tt_svd_capped(fp_paired, (Nx * 3, Nz * 3, Ny * 3), 1e-8 * nfp, (25, 16))
    run_case("a_prevF_rank25_16_solve", prevF_25_16, "solve")
    # (b) 過剰標本化: 前回 F を (25+os, 16+os//2) まで保持し pinv
    for os_ in (4, 8, 12):
        prevF_os = tt_svd_capped(fp_paired, (Nx * 3, Nz * 3, Ny * 3), 1e-10 * nfp, (25 + os_, 16 + os_ // 2))
        run_case(f"b_prevF_oversample_{os_}_pinv", prevF_os, "pinv", rcond=1e-12)
    # (c) 現在 Eq 自身の TT-SVD の pivot（上限性能）
    ref_now = T.tt_svd(Tex, (Nx * 3, Nz * 3, Ny * 3), budget_eq)
    run_case("c_currentEq_pivots_solve", ref_now, "solve")
    # (d) 前回 Eq（近傍状態の Eq）の pivot、絶対予算 budget_eq、rank 上限なし
    mf_p = T.coefficient_fields(rho_p.transpose(0, 2, 1), tuple((rho_p * uu).transpose(0, 2, 1) for uu in u_p))
    Tex_p = T.exact_feq_paired(mf_p, Abas, Dbas, Bbas)
    prevEq = T.tt_svd(Tex_p, (Nx * 3, Nz * 3, Ny * 3), budget_eq)
    run_case("d_prevEq_pivots_solve", prevEq, "solve")
    # ---- 交絡を外した比較（レビュー 6.2）: 打切り予算・rank 上限・solver を揃えて出自だけを変える ----
    # (e) 前回 F を Eq と同じ絶対予算 budget_eq、rank 上限なしで打ち切る → solve / pinv
    prevF_same = T.tt_svd(fp_paired, (Nx * 3, Nz * 3, Ny * 3), budget_eq)
    run_case("e_prevF_same_budget_nocap_solve", prevF_same, "solve")
    run_case("e_prevF_same_budget_nocap_pinv", prevF_same, "pinv", rcond=1e-12)
    # (f) 前回 Eq を F と同じ rank 上限 (25,16) で打ち切る → solve
    prevEq_cap = tt_svd_capped(Tex_p, (Nx * 3, Nz * 3, Ny * 3), budget_eq, (25, 16))
    run_case("f_prevEq_rankcap_25_16_solve", prevEq_cap, "solve")
    # (g) 前回 Eq、同予算、pinv（solver の影響）
    run_case("g_prevEq_pivots_pinv", prevEq, "pinv", rcond=1e-12)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
