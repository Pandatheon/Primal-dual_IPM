import numpy as np
# 小 OCP:N=3, n_x=2, n_u=1.  让 R<0 制造某 stage H_k 不定
n_x, n_u, N = 2, 1, 3
A = np.array([[1.05, 0.],[0., 0.9]])
B = np.array([[1.],[0.5]])
Q  = 0.1*np.eye(n_x)
QN = 0.1*np.eye(n_x)
R  = np.array([[-2.0]])
S  = np.zeros((n_u, n_x))

# --- 反向 Riccati(无正则化:为了纯净验证曲率关系) ---
P = [None]*(N+1); P[N] = QN.copy()
Ks = [None]*N; Hs = [None]*N
indef = []
for k in range(N-1,-1,-1):
    Hk = R + B.T @ P[k+1] @ B
    Gk = S + B.T @ P[k+1] @ A
    eigs, vecs = np.linalg.eigh(Hk)
    Hs[k] = Hk
    print(f'stage {k}:  H_k = {Hk[0,0]:.4f}   eigvals = {eigs}')
    if eigs.min() < 0:
        indef.append((k, float(eigs.min()), vecs[:, int(np.argmin(eigs))]))
    Kk = -np.linalg.solve(Hk, Gk)
    Ks[k] = Kk
    P[k] = Q + A.T @ P[k+1] @ A - Gk.T @ np.linalg.solve(Hk, Gk)

print()
print('不定 stage 及其 mu_k:', [(k, mu) for k, mu, _ in indef])

# --- 选最负的 stage,正向扫构造扰动 ---
indef.sort(key=lambda t: t[1])
k_star, mu_star, u_star = indef[0]
print(f'\\n选 k* = {k_star},  mu* = {mu_star:.4f},  u* = {u_star}')

dx = np.zeros(n_x)
xs = [dx.copy()]; us = []
for j in range(N):
    v = u_star if j == k_star else np.zeros(n_u)
    du = Ks[j] @ dx + v
    us.append(du.copy())
    dx = A @ dx + B @ du
    xs.append(dx.copy())

# 可行性:线性化动力学满足吗
feas = max(np.max(np.abs(xs[j+1] - A@xs[j] - B@us[j])) for j in range(N))
print(f'动力学可行性误差: {feas:.2e}')

# 目标的二次型扰动 = sum z^T C z + x_N^T Q_N x_N (不带 1/2)
C = np.block([[Q, S.T],[S, R]])
curv = sum(np.concatenate([xs[j],us[j]]) @ C @ np.concatenate([xs[j],us[j]]) for j in range(N))
curv += xs[N] @ QN @ xs[N]
print(f'轨迹扰动的总曲率: {curv:.6f}')
print(f'预测值 mu* :       {mu_star:.6f}')
print(f'差: {abs(curv - mu_star):.2e}  (应≈机器精度)')

# 顺便看一眼扰动长什么样
print('\\nδx 轨迹:')
for j,x in enumerate(xs): print(f'  x_{j} = {x}')
print('δu 序列:')
for j,u in enumerate(us): print(f'  u_{j} = {u}')
