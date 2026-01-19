import numpy as np

class abstract_QP_mats:
    def __init__(self):
        raise NotImplementedError("This is an abstract class. Please implement in a subclass.")

    def save_to_npz(self, filename="M_pos_P_pos_mats.npz"):
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

    def export_to_markdown(self, mats, H, eigvals, reduced_list, p_list):
        md = []
        
        # 1. Title and Basic Parameters
        md.append(f"# QP Analysis Report\n")
        md.append(f"## Core Parameter Summary\n")
        md.append(f"| Parameter | Value |\n| :--- | :--- |")
        md.append(f"| Prediction Horizon (N) | {mats.N} |")
        md.append(f"| Hessian H Eigenvalues | `{np.array2string(eigvals, precision=3)}` |")

        # 2. Iteration Process Data Table (Core Analysis)
        md.append(f"## Discrete Riccati Iteration Eigenvalue Evolution\n")
        md.append(f"| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $P$ Eigenvalues |")
        md.append(f"| :--- | :--- | :--- |")
        for i in range(len(reduced_list)):
            step_num = mats.N - i
            m_eigs = np.array2string(reduced_list[i], precision=4, separator=', ')
            b_eigs = np.array2string(p_list[i], precision=4, separator=', ')
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
        output = "\n".join(md)
        with open(f"{mats.name}_report.md", "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Markdown report generated to: {mats.name}_report.md")