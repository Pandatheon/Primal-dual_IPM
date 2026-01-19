import numpy as np

class QP_mats:
    def __init__(self):
        self.N = 5
        #dynamics
        self.A = np.array([[1.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0],
                           [0.0, 0.0, 2.0],])
        self.B = np.array([[1.0, 2.0],
                           [1.0, 1.0],
                           [0.0, 1.0],])
        self.b = np.array([0.0, 0.0, 0.0])
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        # cost
        self.Q = np.diag(np.array([2.0, 2.0, -1.0]))
        self.R = np.diag(np.array([1.0, 2.0]))
        self.S = np.zeros((self.nu, self.nx))
        self.q = np.zeros((self.nx,))
        self.r = np.zeros((self.nu,))
        #terminal cost
        self.Q_e = self.Q
        self.q_e = np.zeros((self.nx,))

        self.uh = 2.0
        self.lh = -2.0
        # inequality constraintsL: h(x,u) = Cx + Du <= d
        self.C = np.zeros((4, self.nx))
        self.D = np.array([[-1.0, 0.0],[0.0, -1.0], [1.0, 0.0],[0.0, 1.0]])
        self.d = np.array([-self.lh, -self.lh, self.uh, self.uh])

    def save_to_npz(self, filename="M_inde_P_inde_mats.npz"):
        """Save all matrices to an .npz file."""
        np.savez(filename,
                 N=self.N,
                 A=self.A,
                 B=self.B,
                 b=self.b,
                 nx=self.nx,
                 nu=self.nu,
                 Q=self.Q,
                 R=self.R,
                 S=self.S,
                 q=self.q,
                 r=self.r,
                 Q_e=self.Q_e,
                 q_e=self.q_e,
                 uh=self.uh,
                 lh=self.lh,
                 C=self.C,
                 D=self.D,
                 d=self.d)
        print(f"Matrices saved to {filename}")

    def export_to_markdown(self, mats, H, eigvals, reduced_list, bpb_list):
        md = []
        
        # 1. Title and Basic Parameters
        md.append(f"# QP Analysis Report\n")
        md.append(f"## Core Parameter Summary\n")
        md.append(f"| Parameter | Value |\n| :--- | :--- |")
        md.append(f"| Prediction Horizon (N) | {mats.N} |")
        md.append(f"| Hessian H Eigenvalues | `{np.array2string(eigvals, precision=3)}` |")

        # 2. Iteration Process Data Table (Core Analysis)
        md.append(f"## Discrete Riccati Iteration Eigenvalue Evolution\n")
        md.append(f"| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $B^T P B$ Contribution Eigenvalues |")
        md.append(f"| :--- | :--- | :--- |")
        for i in range(len(reduced_list)):
            step_num = mats.N - i
            m_eigs = np.array2string(reduced_list[i], precision=4, separator=', ')
            b_eigs = np.array2string(bpb_list[i], precision=4, separator=', ')
            md.append(f"| Step {step_num} | `{m_eigs}` | `{b_eigs}` |")

        # 3. Detail Blocks (Using HTML details tags for folding large matrices)
        md.append(f"\n## Matrix Details (System Definition)\n")
        
        matrix_info = [
            ("A (System Dynamics)", mats.A),
            ("B (Input Matrix)", mats.B),
            ("Q (State Cost)", mats.Q),
            ("R (Input Cost)", mats.R),
            ("H (Full Hessian)", H)
        ]

        for name, data in matrix_info:
            md.append(f"<details>\n<summary>Click to expand {name}</summary>\n\n")
            md.append(f"```python\n{np.array2string(data, precision=3, suppress_small=True)}\n```\n")
            md.append(f"</details>\n")

        # Write to file or print
        from pathlib import Path
        current_file = Path(__file__)
        output = "\n".join(md)
        with open(f"{current_file.stem}_report.md", "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Markdown report generated to: {current_file.stem}_report.md")

if __name__ == "__main__":
    mats = QP_mats()
    print("N:", mats.N)
    print("A:\n", mats.A)
    print("B:\n", mats.B)
    print("b:\n", mats.b)
    print("Q:\n", mats.Q)
    print("R:\n", mats.R)
    print("S:\n", mats.S)
    print("q:\n", mats.q)
    print("r:\n", mats.r)
    print("C:\n", mats.C)
    print("D:\n", mats.D)
    print("d:\n", mats.d)
    print("Q_e:\n", mats.Q_e)
    print("q_e:\n", mats.q_e)
    H = np.block([[mats.Q, mats.S.T],
                  [mats.S, mats.R]])
    print("Hessian matrix H:\n", H)
    eigvals = np.linalg.eigvals(H)
    print("Eigenvalues of H:\n", eigvals)
    # get R+B'PB for N horizon
    P = mats.Q_e
    reduced_eigenvals_list = []
    BPB_eigenvals_list = []
    for _ in range(mats.N):
        reduced_hessian = mats.R + mats.B.T @ P @ mats.B
        reduced_eigenvals_list.append(np.linalg.eigvals(reduced_hessian))
        BPB_eigenvals_list.append(np.linalg.eigvals(mats.B.T @ P @ mats.B))
        K = -np.linalg.solve(reduced_hessian, (mats.S + mats.B.T @ P @ mats.A))
        P = mats.Q + mats.A.T @ P @ mats.A - (mats.S.T + mats.A.T @ P @ mats.B) @ (-K)
    print("Eigenvalues of reduced Hessians M (R + B'PB) over horizon:")
    for i, eigs in enumerate(reduced_eigenvals_list):
        print(f" Step {mats.N - i}: {eigs}")
    print("Corresponding B'PB eigenvalues:")
    for i, bpbs in enumerate(BPB_eigenvals_list):
        print(f" Step {mats.N - i}: {bpbs}")
    
    # Save matrices to .npz file
    # mats.save_to_npz()
    mats.export_to_markdown(mats, H, eigvals, reduced_eigenvals_list, BPB_eigenvals_list)