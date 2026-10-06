#!/usr/bin/env python3
"""
moment-dense / population-TT ハイブリッド衝突の合成場計時（方針決定 2026-10-05 付属）

考え方:
  BGK の非線形性は moment 空間 (rho, j: 4場) にだけある。population (27場) を TT に置き、
  moment は TT から厳密に縮約して dense に持つ。平衡分布は
      f_eq = sum_{i=1}^{10} v_i(q) m_i(x)     (v_i: 固定の方向ベクトル, m_i: dense moment 係数場)
  という q-rank 10 の厳密な構造を持つので、点標本の oracle は O(1) になり、
  TT 化は構造付き randomized TT-SVD で行える。f* = (1-omega) f + omega f_eq は TT 和と丸め。
  丸めは SVD 切捨てで Frobenius 誤差が保証される。f_eq の TT 化誤差は厳密参照が安価なので
  全域または面で直接検証できる。

計測 (BLAS/OMP 1 thread):
  dense:   dense population からの BGK 衝突 (参照)、および dense streaming (27 roll)
  hybrid:  (1) paired TT から 4 moment の dense 化  (2) 10 係数場  (3) 構造付き randomized TT-SVD
           (4) 全域厳密残差 (任意の研究検証)  (5) TT 和 + 丸め  (6) f* の全域・面誤差
合成場・TT-SVD の限界は toy_tt_rank_check.py と同じ。実研究の実装・TCI・検査契約を再現しない。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, itertools, json, time
import numpy as np

Nx, Ny, Nz = 128, 32, 128
TAU = 2.036
OMEGA = 1.0 / TAU
H = 128.0
UMAX = 0.05
MODES = [(1, 1), (2, 1), (1, 2), (3, 2)]
C = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)  # q = 9(cx+1)+3(cy+1)+(cz+1)
w1 = np.array([1 / 6, 2 / 3, 1 / 6])
W = np.array([w1[a + 1] * w1[b + 1] * w1[c + 1] for a, b, c in C.astype(int)])
cax = np.array([-1.0, 0.0, 1.0])


def build_fields(eps, shift_steps=0.0):
    """shift_steps > 0 なら各モードの位相を k_x U(y) shift_steps だけ進め、近傍状態を模す。"""
    x = np.arange(Nx); y = np.arange(Ny) + 0.5; z = np.arange(Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    U = UMAX * (1 - (1 - Y / H) ** 2)

    def msum(coefs):
        s = 0
        for (m, n), a in zip(MODES, coefs):
            kx = 2 * np.pi * m / Nx
            s = s + a * np.cos(kx * X - kx * U * shift_steps) * np.cos(2 * np.pi * n * Z / Nz)
        return s
    g = np.sin(np.pi * Y / (2 * Ny)); gy = np.sin(np.pi * Y / Ny)
    ux = U + eps * UMAX * msum([1.0, 0.8, 0.6, 0.5]) * g
    uy = 0.3 * eps * UMAX * msum([0.6, 0.5, 0.4, 0.3]) * gy
    uz = eps * UMAX * msum([0.5, 0.7, 0.9, 0.4]) * g
    rho = 1 + 3 * (eps * UMAX) ** 2 * msum([1, 1, 1, 1])
    return rho, (ux, uy, uz)


def equilibrium(rho, u):
    ustack = np.stack(u, axis=-1)
    cu = ustack @ C.T
    uu = (ustack ** 2).sum(-1)[..., None]
    return W * rho[..., None] * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * uu)


def nonequilibrium_CE(rho, u):
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


# ---------- 方向基底 v_i(q) = a_i(qx) d_i(qz) b_i(qy) （w を含む） ----------
def direction_basis():
    one = np.ones(3); c1 = 3 * cax; c2 = 4.5 * cax ** 2 - 1.5
    wx = w1
    # 係数場の順: rho, jx, jy, jz, Qxx, Qyy, Qzz, Qxy, Qxz, Qyz
    fac = [(one, one, one), (c1, one, one), (one, one, c1), (one, c1, one),
           (c2, one, one), (one, one, c2), (one, c2, one),
           (c1, one, c1), (c1, c1, one), (one, c1, c1)]       # (qx, qz, qy)
    A = np.array([wx * f[0] for f in fac])   # (10,3) over qx
    D = np.array([wx * f[1] for f in fac])   # (10,3) over qz
    B = np.array([wx * f[2] for f in fac])   # (10,3) over qy
    return A, D, B


def coefficient_fields(rho, j):
    inv = 1.0 / rho
    jx, jy, jz = j
    return np.stack([rho, jx, jy, jz, jx * jx * inv, jy * jy * inv, jz * jz * inv,
                     jx * jy * inv, jx * jz * inv, jy * jz * inv], axis=0)   # (10,Nx,Nz,Ny)


# ---------- dense 参照 ----------
def dense_collision(f):
    """f: (Nx,Ny,Nz,27) dense. 戻り f*. 27 方向ループで一時配列を抑える。"""
    rho = f.sum(-1)
    jx = f @ C[:, 0]; jy = f @ C[:, 1]; jz = f @ C[:, 2]
    inv = 1.0 / rho
    ux, uy, uz = jx * inv, jy * inv, jz * inv
    uu = ux * ux + uy * uy + uz * uz
    out = np.empty_like(f)
    for q in range(27):
        cu = C[q, 0] * ux + C[q, 1] * uy + C[q, 2] * uz
        feq = W[q] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu)
        out[..., q] = (1 - OMEGA) * f[..., q] + OMEGA * feq
    return out


def dense_streaming(f):
    out = np.empty_like(f)
    for q in range(27):
        out[..., q] = np.roll(f[..., q], shift=(int(C[q, 0]), int(C[q, 1]), int(C[q, 2])), axis=(0, 1, 2))
    return out


# ---------- TT ユーティリティ ----------
def tt_svd(T, dims, tol_abs):
    d = len(dims); eps = tol_abs / np.sqrt(max(d - 1, 1))
    M = np.ascontiguousarray(T).reshape(dims[0], -1); r = 1; cores = []
    for k in range(d - 1):
        M = M.reshape(r * dims[k], -1)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        rn = len(s)
        while rn > 1 and tail[rn - 1] <= eps:
            rn -= 1
        cores.append(U[:, :rn].reshape(r, dims[k], rn)); M = s[:rn, None] * Vt[:rn]; r = rn
    cores.append(M.reshape(r, dims[-1], 1))
    return cores


def tt_full(cores):
    G = cores[0].reshape(cores[0].shape[1], -1)
    for c in cores[1:]:
        r, n, r2 = c.shape
        G = (G @ c.reshape(r, n * r2)).reshape(-1, r2)
    return G.reshape(-1)


def tt_round(cores, tol_abs):
    """左から QR で直交化し、右から SVD で切り捨てる標準的な丸め。"""
    d = len(cores); eps = tol_abs / np.sqrt(max(d - 1, 1))
    cores = [c.copy() for c in cores]
    for k in range(d - 1):
        r, n, r2 = cores[k].shape
        Q, R = np.linalg.qr(cores[k].reshape(r * n, r2))
        cores[k] = Q.reshape(r, n, -1)
        cores[k + 1] = np.einsum("ab,bnc->anc", R, cores[k + 1])
    for k in range(d - 1, 0, -1):
        r, n, r2 = cores[k].shape
        U, s, Vt = np.linalg.svd(cores[k].reshape(r, n * r2), full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        rn = len(s)
        while rn > 1 and tail[rn - 1] <= eps:
            rn -= 1
        cores[k] = Vt[:rn].reshape(rn, n, r2)
        cores[k - 1] = np.einsum("anb,bc->anc", cores[k - 1], U[:, :rn] * s[:rn])
    return cores


def tt_sum(c1, c2):
    out = []
    for k, (a, b) in enumerate(zip(c1, c2)):
        ra, n, ra2 = a.shape; rb, _, rb2 = b.shape
        if k == 0:
            out.append(np.concatenate([a, b], axis=2))
        elif k == len(c1) - 1:
            out.append(np.concatenate([a, b], axis=0))
        else:
            blk = np.zeros((ra + rb, n, ra2 + rb2)); blk[:ra, :, :ra2] = a; blk[ra:, :, ra2:] = b
            out.append(blk)
    return out


def tt_scale(cores, s):
    out = [c.copy() for c in cores]; out[0] = out[0] * s; return out


# ---------- paired TT からの dense moment ----------
def moments_from_paired(cores):
    """cores: paired [(x,qx),(z,qz),(y,qy)]. 戻り rho, jx, jy, jz を dense (Nx,Nz,Ny) で。"""
    G1, G2, G3 = cores
    r1 = G1.shape[2]; r2 = G2.shape[2]
    g1 = G1.reshape(Nx, 3, r1); g2 = G2.reshape(r1, Nz, 3, r2); g3 = G3.reshape(r2, Ny, 3)
    one = np.ones(3)
    outs = []
    for wx, wz, wy in [(one, one, one), (cax, one, one), (one, one, cax), (one, cax, one)]:  # rho, jx, jy, jz
        A1 = np.einsum("xqa,q->xa", g1, wx)            # (Nx, r1)
        A2 = np.einsum("azqb,q->azb", g2, wz)          # (r1, Nz, r2)
        A3 = np.einsum("byq,q->by", g3, wy)            # (r2, Ny)
        T12 = (A1 @ A2.reshape(r1, Nz * r2)).reshape(Nx * Nz, r2)
        outs.append((T12 @ A3).reshape(Nx, Nz, Ny))
    return outs


# ---------- 構造付き randomized TT-SVD: T = sum_i a_i(qx) d_i(qz) b_i(qy) m_i(x,z,y) ----------
def structured_tt_svd(mfields, Abas, Dbas, Bbas, tol_abs, k=40, power=1, rng=None):
    """T[(x,qx),(z,qz),(y,qy)] = sum_i a_i(qx) d_i(qz) b_i(qy) m_i(x,z,y) の randomized TT-SVD。
    実体化せず、10 本の (Nx x Nz*Ny) 行列との batched matmul で範囲探索する。"""
    rng = rng or np.random.default_rng(0)
    nI = mfields.shape[0]
    M = np.ascontiguousarray(mfields.reshape(nI, Nx, Nz * Ny))          # m_i(x, s), s=(z,y)
    Mt = np.ascontiguousarray(M.transpose(0, 2, 1))                       # m_i(s, x)
    DB = np.einsum("iq,ip->iqp", Dbas, Bbas).reshape(nI, 9)                # (i, (qz,qy))

    def apply_T(Om):            # Om: (Nz, 3, Ny, 3, k) -> Y: (Nx*3, k)
        Om_r = Om.transpose(0, 2, 1, 3, 4).reshape(Nz * Ny, 9, k)         # (s, (qz,qy), k)
        Omt = np.einsum("ir,srl->isl", DB, Om_r)                           # (i, s, k)
        Yi = M @ Omt                                                       # (i, Nx, k)
        return np.einsum("iq,ixl->xql", Abas, Yi).reshape(Nx * 3, k)

    def apply_Tt(Y):            # Y: (Nx*3, k) -> Z: (Nz,3,Ny,3,k)
        Yi = np.einsum("iq,xql->ixl", Abas, Y.reshape(Nx, 3, k))          # (i, Nx, k)
        Zi = Mt @ Yi                                                       # (i, s, k)
        Z = np.einsum("ir,isl->srl", DB, Zi).reshape(Nz, Ny, 3, 3, k)     # (z,y,qz,qy,k)
        return Z.transpose(0, 2, 1, 3, 4)

    Y = apply_T(rng.standard_normal((Nz, 3, Ny, 3, k)))
    for _ in range(power):
        Q, _ = np.linalg.qr(Y)
        Y = apply_T(apply_Tt(Q))
    Q, _ = np.linalg.qr(Y)                                                 # (Nx*3, k)
    Qi = np.einsum("iq,xql->ixl", Abas, Q.reshape(Nx, 3, k))
    Bi = (Mt @ Qi).reshape(nI, Nz, Ny, k)                                  # (i, z, y, l)
    Bmat = np.einsum("ir,izyl->lzyr", DB, Bi).reshape(k, Nz, Ny, 3, 3).transpose(0, 1, 3, 2, 4).reshape(k, Nz * 3 * Ny * 3)
    eps = tol_abs / np.sqrt(2)
    U, sv, Vt = np.linalg.svd(Bmat, full_matrices=False)
    tail = np.sqrt(np.cumsum(sv[::-1] ** 2))[::-1]
    r1 = len(sv)
    while r1 > 1 and tail[r1 - 1] <= eps:
        r1 -= 1
    core1 = (Q @ U[:, :r1]).reshape(1, Nx * 3, r1)
    Mrest = (sv[:r1, None] * Vt[:r1]).reshape(r1 * Nz * 3, Ny * 3)
    U2, s2, Vt2 = np.linalg.svd(Mrest, full_matrices=False)
    tail2 = np.sqrt(np.cumsum(s2[::-1] ** 2))[::-1]
    r2 = len(s2)
    while r2 > 1 and tail2[r2 - 1] <= eps:
        r2 -= 1
    core2 = U2[:, :r2].reshape(r1, Nz * 3, r2)
    core3 = (s2[:r2, None] * Vt2[:r2]).reshape(r2, Ny * 3, 1)
    return [core1, core2, core3]


# ---------- 固定 pivot の cross interpolation（oracle は dense moment から O(1)） ----------
def maxvol(A, tol=1.05, maxiter=200):
    """A: (n, r), n >= r. |det| を貪欲に最大化する r 行を返す。"""
    n, r = A.shape
    Ac = A.copy(); I = []
    mask = np.ones(n, bool)
    for kcol in range(r):
        col = np.where(mask, np.abs(Ac[:, kcol]), -1.0)
        i = int(np.argmax(col)); I.append(i); mask[i] = False
        piv = Ac[i, kcol]
        if abs(piv) > 0:
            Ac = Ac - np.outer(Ac[:, kcol] / piv, Ac[i, :])
            Ac[i, :] = 0.0
    I = np.array(I)
    for _ in range(maxiter):
        Bm = A @ np.linalg.inv(A[I])
        i, j = np.unravel_index(int(np.argmax(np.abs(Bm))), Bm.shape)
        if abs(Bm[i, j]) <= tol:
            break
        I[j] = i
    return I


def pivots_from_tt(cores):
    """paired 3 コア TT から入れ子の pivot 集合を取る（TT-cross の標準手順）。
    I1: (x,qx) の r1 個。 I2: (I1 の要素, (z,qz)) の r2 個。 J2: (y,qy) の r2 個。 J1: ((z,qz), J2 の要素) の r1 個。"""
    G1, G2, G3 = cores
    r1, r2 = G1.shape[2], G2.shape[2]
    n1, n2, n3 = G1.shape[1], G2.shape[1], G3.shape[1]
    # 左: core1 を直交化して maxvol
    Q1, _ = np.linalg.qr(G1.reshape(n1, r1))
    I1 = maxvol(Q1)
    # 左: 実際の pivot 行 G1[I1] と core2 の積で frame を作り maxvol
    G12 = np.einsum("ab,bnc->anc", G1.reshape(n1, r1)[I1], G2).reshape(r1 * n2, r2)
    Q12, _ = np.linalg.qr(G12)
    I2_pairs = np.stack(np.unravel_index(maxvol(Q12), (r1, n2)), axis=1)      # (r2, 2): (a in I1, n2)
    # 右: core3 を直交化して maxvol
    Q3, _ = np.linalg.qr(G3.reshape(r2, n3).T)
    J2 = maxvol(Q3)
    # 右: 実際の pivot 列 G3[:, J2] と core2 の積で frame を作り maxvol
    G23 = np.einsum("anb,bc->anc", G2, G3.reshape(r2, n3)[:, J2])           # (r1, n2, r2)
    Q23, _ = np.linalg.qr(G23.transpose(1, 2, 0).reshape(n2 * r2, r1))
    J1_pairs = np.stack(np.unravel_index(maxvol(Q23), (n2, r2)), axis=1)      # (r1, 2): (n2, c in J2)
    return dict(I1=I1, I2_pairs=I2_pairs, J2=J2, J1_pairs=J1_pairs)


def cross_feq(mf, Abas, Dbas, Bbas, piv):
    """固定 pivot の cross: T ~ C1 P1^-1 C2 P2^-1 C3。oracle は T = sum_i a_i d_i b_i m_i。"""
    I1 = piv["I1"]; I2p = piv["I2_pairs"]; J2 = piv["J2"]; J1p = piv["J1_pairs"]
    r1, r2 = len(I1), len(J2)
    # 添字分解
    x1, qx1 = np.unravel_index(I1, (Nx, 3))
    y2, qy2 = np.unravel_index(J2, (Ny, 3))
    n2_J1 = J1p[:, 0]; b_J1 = J1p[:, 1]                       # J1 = ((z,qz) index, pointer into J2)
    z1, qz1 = np.unravel_index(n2_J1, (Nz, 3)); yJ1, qyJ1 = y2[b_J1], qy2[b_J1]
    a_I2 = I2p[:, 0]; n2_I2 = I2p[:, 1]                       # I2 = (pointer into I1, (z,qz) index)
    xI2, qxI2 = x1[a_I2], qx1[a_I2]; z2, qz2 = np.unravel_index(n2_I2, (Nz, 3))
    # C1[(x,qx), j] = sum_i A[i,qx] D[i,qz_j] B[i,qy_j] m[i,x,z_j,y_j]
    coef1 = Dbas[:, qz1] * Bbas[:, qyJ1]                       # (i, r1)
    C1 = np.einsum("ij,iq,ixj->xqj", coef1, Abas, mf[:, :, z1, yJ1]).reshape(Nx * 3, r1)
    P1 = C1[I1]                                                # (r1, r1)
    # C2[a, (z,qz), b] = sum_i A[i,qx_a] m[i,x_a,z,y_b] D[i,qz] B[i,qy_b]
    C2 = np.einsum("ia,ib,iazb,iq->azqb", Abas[:, qx1], Bbas[:, qy2], mf[:, x1][:, :, :, y2], Dbas).reshape(r1, Nz * 3, r2)
    P2 = C2.reshape(r1 * Nz * 3, r2)[np.ravel_multi_index((a_I2, n2_I2), (r1, Nz * 3))]   # (r2, r2)
    # C3[c, (y,qy)] = sum_i A[i,qx_c] D[i,qz_c] m[i,x_c,z_c,y] B[i,qy]
    coef3 = Abas[:, qxI2] * Dbas[:, qz2]                       # (i, r2)
    C3 = np.einsum("ic,icy,ip->cyp", coef3, mf[:, xI2, z2, :], Bbas).reshape(r2, Ny * 3)
    core1 = np.linalg.solve(P1.T, C1.T).T.reshape(1, Nx * 3, r1)              # C1 P1^-1
    core2 = np.linalg.solve(P2.T, C2.reshape(-1, r2).T).T.reshape(r1, Nz * 3, r2)   # C2 P2^-1
    core3 = C3.reshape(r2, Ny * 3, 1)
    n_raw = C1.size + C2.size + C3.size
    return [core1, core2, core3], n_raw, (np.linalg.cond(P1), np.linalg.cond(P2))


def exact_feq_paired(mfields, Abas, Dbas, Bbas):
    """厳密 f_eq を paired 配置 (Nx*3, Nz*3, Ny*3) で実体化（検証用、dense 衝突相当の費用）。"""
    nI = mfields.shape[0]
    V = np.einsum("iq,ir,ip->iqrp", Abas, Dbas, Bbas)         # (nI, qx, qz, qy)
    T = np.einsum("ixzy,iqrp->xqzryp", mfields, V)
    return T.reshape(Nx * 3, Nz * 3, Ny * 3)


def to_paired(arr):   # (Nx,Ny,Nz,27) -> (Nx*3, Nz*3, Ny*3)   q=9(cx+1)+3(cy+1)+(cz+1) -> (qx,qy,qz)
    return arr.reshape(Nx, Ny, Nz, 3, 3, 3).transpose(0, 3, 2, 5, 1, 4).reshape(Nx * 3, Nz * 3, Ny * 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-9, help="f* の絶対予算 / ||f*||")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="docs/toy_moment_hybrid_results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rho, u = build_fields(args.eps)
    feq = equilibrium(rho, u)
    f = feq + nonequilibrium_CE(rho, u)
    fstar_ref = (1 - OMEGA) * f + OMEGA * feq
    nf = np.linalg.norm(fstar_ref)
    del feq

    # 状態 TT（前更新の受理 F に相当）: paired 配置, 1e-8 相対
    fp = to_paired(f)
    state = tt_svd(fp, (Nx * 3, Nz * 3, Ny * 3), 1e-8 * np.linalg.norm(fp))
    state_ranks = [c.shape[2] for c in state[:-1]]
    f_tt_err = np.linalg.norm(tt_full(state) - fp.reshape(-1)) / np.linalg.norm(fp)
    Abas, Dbas, Bbas = direction_basis()
    res = dict(config=dict(eps=args.eps, tol=args.tol, threads=os.environ["OMP_NUM_THREADS"], Nx=Nx, Ny=Ny, Nz=Nz, tau=TAU),
               state_tt=dict(ranks=state_ranks, rel_err=float(f_tt_err)))

    def timeit(fn, n=args.repeat):
        ts = []
        out = None
        for _ in range(n):
            t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), out

    # ---- dense 参照 ----
    t_dense_coll, fstar_dense = timeit(lambda: dense_collision(f))
    assert np.allclose(fstar_dense, fstar_ref, rtol=0, atol=1e-15 * nf)
    t_dense_stream, _ = timeit(lambda: dense_streaming(fstar_dense))
    res["dense"] = dict(collision_s=t_dense_coll, streaming_s=t_dense_stream)
    del fstar_dense

    # ---- hybrid ----
    t_mom, mom = timeit(lambda: moments_from_paired(state))
    rho_d, jx_d, jy_d, jz_d = mom
    mom_err = max(np.abs(rho_d - rho.transpose(0, 2, 1)).max() / rho.max(),
                  np.abs(jx_d - (rho * u[0]).transpose(0, 2, 1)).max() / np.abs(rho * u[0]).max())
    t_coef, mf = timeit(lambda: coefficient_fields(rho_d, (jx_d, jy_d, jz_d)))
    budget_eq = 0.5 * args.tol * nf / OMEGA
    t_svd, feq_tt_rand = timeit(lambda: structured_tt_svd(mf, Abas, Dbas, Bbas, budget_eq, k=40, power=1, rng=rng))
    Tex = exact_feq_paired(mf, Abas, Dbas, Bbas)
    nTex = np.linalg.norm(Tex)
    rand_err = np.linalg.norm(tt_full(feq_tt_rand) - Tex.reshape(-1))
    res["hybrid_randomized_svd"] = dict(seconds=t_svd, ranks=[c.shape[2] for c in feq_tt_rand[:-1]],
                                        abs_err=float(rand_err), err_over_fstar=float(rand_err / nf))
    # ---- 固定 pivot cross ----
    # pivot 源 (a): 現在の厳密 f_eq の TT-SVD（oracle-assisted、上限性能）
    ref_now = tt_svd(Tex, (Nx * 3, Nz * 3, Ny * 3), budget_eq)
    # pivot 源 (b): 近傍状態（位相を k_x U(y) * shift だけ進めた場）の f_eq TT-SVD。前更新の pivot 継承を模す
    cross_results = {}
    for label, src in [("pivots_from_current_feq", None), ("pivots_from_shifted_state_10", 10.0), ("pivots_from_shifted_state_50", 50.0)]:
        if src is None:
            ref = ref_now
        else:
            rho_s, u_s = build_fields(args.eps, shift_steps=src)
            mf_s = coefficient_fields(rho_s.transpose(0, 2, 1), tuple((rho_s * uu).transpose(0, 2, 1) for uu in u_s))
            Tex_s = exact_feq_paired(mf_s, Abas, Dbas, Bbas)
            ref = tt_svd(Tex_s, (Nx * 3, Nz * 3, Ny * 3), budget_eq)
            del Tex_s
        piv = pivots_from_tt(ref)
        t_cross, (feq_tt_c, n_raw, conds) = timeit(lambda: cross_feq(mf, Abas, Dbas, Bbas, piv))
        err_c = np.linalg.norm(tt_full(feq_tt_c) - Tex.reshape(-1))
        cross_results[label] = dict(seconds=t_cross, ranks=[len(piv["I1"]), len(piv["J2"])], n_raw=int(n_raw),
                                    cond_P1=float(conds[0]), cond_P2=float(conds[1]),
                                    abs_err=float(err_c), err_over_fstar=float(err_c / nf), err_over_feq=float(err_c / nTex),
                                    within_budget=bool(err_c <= budget_eq))
        if label == "pivots_from_current_feq":
            feq_tt = feq_tt_c; t_feq = t_cross
    res["hybrid_cross"] = cross_results
    feq_ranks = [c.shape[2] for c in feq_tt[:-1]]
    feq_abs_err = cross_results["pivots_from_current_feq"]["abs_err"]
    t_verify = None
    # 面検証（y=2 面、全 27 方向）: 厳密 f_eq と TT f_eq
    def verify_plane(yidx=2):
        G1, G2, G3 = feq_tt
        r1, r2 = G1.shape[2], G2.shape[2]
        g3 = G3.reshape(r2, Ny, 3)[:, yidx, :]
        T12 = (G1.reshape(Nx * 3, r1) @ G2.reshape(r1, Nz * 3 * r2)).reshape(Nx * 3 * Nz * 3, r2)
        plane_tt = (T12 @ g3).reshape(Nx, 3, Nz, 3, 3)
        V = np.einsum("iq,ir,ip->iqrp", Abas, Dbas, Bbas)
        plane_ex = np.einsum("ixz,iqrp->xqzrp", mf[:, :, :, yidx], V)
        return np.linalg.norm(plane_tt - plane_ex), np.linalg.norm(plane_ex)
    t_plane, (pl_err, pl_norm) = timeit(verify_plane)
    del Tex
    # f* = (1-omega) f + omega f_eq, 丸め
    def assemble():
        s = tt_sum(tt_scale(state, 1 - OMEGA), tt_scale(feq_tt, OMEGA))
        return tt_round(s, 0.5 * args.tol * nf)
    t_asm, fstar_tt = timeit(assemble)
    fstar_ranks = [c.shape[2] for c in fstar_tt[:-1]]
    fstar_full = tt_full(fstar_tt).reshape(Nx * 3, Nz * 3, Ny * 3)
    ref_p = to_paired(fstar_ref)
    err_full = np.linalg.norm(fstar_full - ref_p) / nf
    # 面相対誤差 (y=2, y=31) 全 27 方向
    def plane_rel(yidx):
        a = fstar_full.reshape(Nx, 3, Nz, 3, Ny, 3)[:, :, :, :, yidx, :]
        b = ref_p.reshape(Nx, 3, Nz, 3, Ny, 3)[:, :, :, :, yidx, :]
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))
    # 比較のため、現在 TT 状態 (1e-8 丸め済) を厳密に衝突させた参照との差も出す
    res["hybrid"] = dict(
        moments_s=t_mom, moments_maxrel_err=float(mom_err),
        coefficient_fields_s=t_coef,
        feq_construction_s=t_feq, feq_method="fixed-pivot cross (pivots from current f_eq)",
        feq_tt_ranks=feq_ranks, feq_budget_abs=float(budget_eq),
        feq_abs_err_full=float(feq_abs_err), feq_err_over_fstar=float(feq_abs_err / nf),
        verify_plane_s=t_plane, feq_plane_rel_err=float(pl_err / pl_norm),
        assemble_sum_round_s=t_asm, fstar_tt_ranks=fstar_ranks,
        fstar_err_full_over_ref=float(err_full), fstar_plane_rel_err={"y2": plane_rel(2), "y16": plane_rel(16), "y31": plane_rel(31)},
        construction_total_s=t_mom + t_coef + t_feq + t_asm,
        construction_plus_plane_verify_s=t_mom + t_coef + t_feq + t_asm + t_plane,
    )
    res["ratios"] = dict(
        construction_over_dense_collision=res["hybrid"]["construction_total_s"] / t_dense_coll,
        construction_plus_plane_verify_over_dense_collision=res["hybrid"]["construction_plus_plane_verify_s"] / t_dense_coll,
        construction_over_dense_collision_plus_streaming=res["hybrid"]["construction_total_s"] / (t_dense_coll + t_dense_stream),
    )
    print(json.dumps(res, indent=1))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
