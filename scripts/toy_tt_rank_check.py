#!/usr/bin/env python3
"""
合成場によるTT rank診断（方向性レビュー 2026-10-05 付属）

目的:
  README記載のN128壁帯 (128 x 32 x 128, D3Q27, BGK tau=2.036) と同じ形の
  合成分布関数を作り、TT-SVD で以下を測る。
    (1) 添字配置 (xzy|q, xyz|q, q因子化, quantics) ごとの rank と要素数
    (2) 衝突後分布 f* に対する R項寄与 Δf*_R の、同じ絶対誤差予算での rank
    (3) 線形部 f*_lin の rank
  実研究の場・コード・TCIではなく、平衡+Chapman-Enskog一次の合成場に対する
  最適打ち切り (TT-SVD) の値である。採否の根拠ではなく、仮説の篩として使う。

実行例:
  python3 scripts/toy_tt_rank_check.py --eps 0.01 0.1 0.3 1.0 --layout-eps 0.1 1.0
"""
import argparse, itertools, json, sys, time
import numpy as np

Nx, Ny, Nz = 128, 32, 128
TAU = 2.036
H = 128.0          # 半チャネル高さ (細格子単位)。帯は下壁から32層
UMAX = 0.05
MODES = [(1, 1), (2, 1), (1, 2), (3, 2)]   # (kx, kz) の4モード、phase 0

C = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)  # (27,3), q=9(cx+1)+3(cy+1)+(cz+1)
w1 = {-1: 1/6, 0: 2/3, 1: 1/6}
W = np.array([w1[a] * w1[b] * w1[c] for a, b, c in C.astype(int)])


def build_fields(eps):
    x = np.arange(Nx); y = np.arange(Ny) + 0.5; z = np.arange(Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    U = UMAX * (1 - (1 - Y / H) ** 2)

    def msum(coefs):
        s = 0
        for (m, n), a in zip(MODES, coefs):
            s = s + a * np.cos(2 * np.pi * m * X / Nx) * np.cos(2 * np.pi * n * Z / Nz)
        return s

    g = np.sin(np.pi * Y / (2 * Ny))          # 壁で0、帯上端で1
    gy = np.sin(np.pi * Y / Ny)               # 両端で0
    ux = U + eps * UMAX * msum([1.0, 0.8, 0.6, 0.5]) * g
    uy = 0.3 * eps * UMAX * msum([0.6, 0.5, 0.4, 0.3]) * gy
    uz = eps * UMAX * msum([0.5, 0.7, 0.9, 0.4]) * g
    rho = 1 + 3 * (eps * UMAX) ** 2 * msum([1, 1, 1, 1])
    return rho, (ux, uy, uz), U


def equilibrium(rho, u):
    ustack = np.stack(u, axis=-1)                       # (Nx,Ny,Nz,3)
    cu = ustack @ C.T                                   # (Nx,Ny,Nz,27)
    uu = (ustack ** 2).sum(-1)[..., None]
    return W * rho[..., None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * uu)


def nonequilibrium_CE(rho, u):
    # f^(1) ≈ -3 tau w_q rho (c c - I/3) : grad u   (一次Chapman-Enskog, O(u^3)無視)
    grads = [[None] * 3 for _ in range(3)]
    for b in range(3):
        gx, gy, gz = np.gradient(u[b], axis=(0, 1, 2))
        grads[0][b], grads[1][b], grads[2][b] = gx, gy, gz
    out = np.empty(rho.shape + (27,))
    for q in range(27):
        A = 0
        for a in range(3):
            for b in range(3):
                coef = C[q, a] * C[q, b] - (1 / 3 if a == b else 0.0)
                if coef != 0:
                    A = A + coef * grads[a][b]
        out[..., q] = -3 * TAU * W[q] * rho * A
    return out


def r_term(rho, u, U):
    # R = rho u' u'^T, u' = u - U e_x ;  Δf*_R = (1/tau) w (4.5 cc:R - 1.5 trR)
    up = [u[0] - U, u[1], u[2]]
    ccR = 0
    trR = 0
    for a in range(3):
        for b in range(3):
            Rab = rho * up[a] * up[b]
            ccR = ccR + (C[:, a] * C[:, b]) * Rab[..., None]
            if a == b:
                trR = trR + Rab
    return (1 / TAU) * W * (4.5 * ccR - 1.5 * trR[..., None])


def tt_svd(T, dims, tol_abs):
    d = len(dims)
    eps = tol_abs / np.sqrt(max(d - 1, 1))
    M = np.ascontiguousarray(T).reshape(dims[0], -1)
    r = 1
    ranks = [1]
    cores = []
    for k in range(d - 1):
        M = M.reshape(r * dims[k], -1)
        Uk, s, Vt = np.linalg.svd(M, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        rnew = len(s)
        while rnew > 1 and tail[rnew - 1] <= eps:
            rnew -= 1
        cores.append(Uk[:, :rnew].reshape(r, dims[k], rnew))
        M = s[:rnew, None] * Vt[:rnew]
        r = rnew
        ranks.append(r)
    cores.append(M.reshape(r, dims[-1], 1))
    ranks.append(1)
    elems = sum(c.size for c in cores)
    return ranks, elems, cores


def tt_full(cores):
    G = cores[0].reshape(cores[0].shape[1], -1)
    for c in cores[1:]:
        r, n, r2 = c.shape
        G = (G @ c.reshape(r, n * r2)).reshape(-1, r2)
    return G.reshape(-1)


def layouts(T):
    """T: (Nx,Ny,Nz,27). 各配置の (名前, 配列, dims, 空間コアの判定) を返す。"""
    out = []
    out.append(("xzy|q", T.transpose(0, 2, 1, 3), (Nx, Nz, Ny, 27), "qlast"))
    out.append(("xyz|q", T, (Nx, Ny, Nz, 27), "qlast"))
    Tq = T.reshape(Nx, Ny, Nz, 3, 3, 3)       # (x,y,z,qx,qy,qz)
    out.append(("x qx z qz y qy", Tq.transpose(0, 3, 2, 5, 1, 4), (Nx, 3, Nz, 3, Ny, 3), "fact"))
    out.append(("x qx y qy z qz", Tq.transpose(0, 3, 1, 4, 2, 5), (Nx, 3, Ny, 3, Nz, 3), "fact"))
    out.append(("xzy|qx qy qz", Tq.transpose(0, 2, 1, 3, 4, 5), (Nx, Nz, Ny, 3, 3, 3), "fact"))
    Tb = T.reshape((2,) * 7 + (2,) * 5 + (2,) * 7 + (27,))   # x bits(0-6), y bits(7-11), z bits(12-18), q(19)
    xb = list(range(0, 7)); yb = list(range(7, 12)); zb = list(range(12, 19))
    out.append(("quantics x z y |q", Tb.transpose(xb + zb + yb + [19]), (2,) * 19 + (27,), "qtt"))
    inter = [v for pair in zip(xb, zb) for v in pair]
    out.append(("quantics (xz)interleave y |q", Tb.transpose(inter + yb + [19]), (2,) * 19 + (27,), "qtt"))
    return out


def summarize(name, ranks, elems, dims, kind, dense):
    if kind == "qlast":
        spatial = sum(ranks[k] * dims[k] * ranks[k + 1] for k in range(len(dims) - 1))
    else:
        spatial = None
    return dict(layout=name, ranks=ranks, max_rank=max(ranks), elements=int(elems),
                elements_over_dense=elems / dense, spatial_fiber_points=spatial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.1, 0.3, 1.0])
    ap.add_argument("--layout-eps", nargs="+", type=float, default=[0.1, 1.0])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--out", default="docs/toy_tt_rank_results.json")
    args = ap.parse_args()

    dense = Nx * Ny * Nz * 27
    results = dict(config=dict(Nx=Nx, Ny=Ny, Nz=Nz, tau=TAU, H=H, Umax=UMAX, modes=MODES, tol=args.tol),
                   amplitude_sweep=[], layout_sweep=[])

    for eps in args.eps:
        t0 = time.time()
        rho, u, U = build_fields(eps)
        feq = equilibrium(rho, u)
        f = feq + nonequilibrium_CE(rho, u)
        fstar = (1 - 1 / TAU) * f + (1 / TAU) * feq
        dfR = r_term(rho, u, U)
        flin = fstar - dfR
        nf = np.linalg.norm(fstar)
        budget = args.tol * nf
        rec = dict(eps=eps, up_over_U=float(np.sqrt(((u[0] - U) ** 2 + u[1] ** 2 + u[2] ** 2).mean()) / np.sqrt((U ** 2).mean())),
                   dfR_over_fstar=float(np.linalg.norm(dfR) / nf),
                   rho_min=float(rho.min()), umax=float(max(np.abs(a).max() for a in u)))
        for label, arr in [("f_input", f), ("f_star", fstar), ("f_star_lin", flin), ("df_star_R", dfR)]:
            T = arr.transpose(0, 2, 1, 3)   # xzy|q
            ranks, elems, cores = tt_svd(T, (Nx, Nz, Ny, 27), budget)
            rec[label] = dict(ranks=ranks, elements=int(elems), spatial_fiber_points=int(sum(ranks[k] * d * ranks[k + 1] for k, d in enumerate((Nx, Nz, Ny)))))
            if label == "f_star":
                err = np.linalg.norm(tt_full(cores) - T.reshape(-1)) / nf
                rec[label]["achieved_rel_err"] = float(err)
        rec["seconds"] = round(time.time() - t0, 1)
        results["amplitude_sweep"].append(rec)
        print(json.dumps(rec), flush=True)

    for eps in args.layout_eps:
        rho, u, U = build_fields(eps)
        feq = equilibrium(rho, u)
        f = feq + nonequilibrium_CE(rho, u)
        fstar = (1 - 1 / TAU) * f + (1 / TAU) * feq
        budget = args.tol * np.linalg.norm(fstar)
        for name, arr, dims, kind in layouts(fstar):
            t0 = time.time()
            ranks, elems, _ = tt_svd(arr, dims, budget)
            rec = summarize(name, ranks, elems, dims, kind, dense)
            rec.update(eps=eps, seconds=round(time.time() - t0, 1))
            results["layout_sweep"].append(rec)
            print(json.dumps(rec), flush=True)

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print("written", args.out)


if __name__ == "__main__":
    main()
