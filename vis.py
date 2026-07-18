import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------
# 1. Mathematical Operations
# ---------------------------------------------------------
def eval_legendre(z: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Evaluates standard Legendre polynomials up to max_degree using Bonnet's recursion formula."""
    L = [torch.ones_like(z)]
    if max_degree >= 1:
        L.append(z)
    for k in range(1, max_degree):
        # (k+1) L_{k+1}(z) = (2k+1) z L_k(z) - k L_{k-1}(z)
        L_next = ((2 * k + 1) * z * L[k] - k * L[k - 1]) / (k + 1)
        L.append(L_next)
    return torch.stack(L, dim=-1)

def project_path_to_coeffs(W: torch.Tensor, t: torch.Tensor, K: int) -> torch.Tensor:
    """Projects an SDE path onto K Legendre coefficients (Itô integral approximation)."""
    dW = W[1:] - W[:-1]
    t_left = t[:-1]
    s, t_end = t[0], t[-1]

    z_left = 2.0 * (t_left - s) / (t_end - s) - 1.0
    L_z = eval_legendre(z_left, K - 1)
    coeffs = torch.einsum('nk,n->k', L_z, dW)
    return coeffs

def reconstruct_path_from_coeffs(coeffs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Reconstructs the deterministic continuous path analytically from the coefficients."""
    K = coeffs.shape[0]
    s, t_end = t[0], t[-1]
    z = 2.0 * (t - s) / (t_end - s) - 1.0

    L_z = eval_legendre(z, K)
    W_recon = torch.zeros_like(t)

    W_recon += coeffs[0] * (z + 1.0) / 2.0
    for k in range(1, K):
        W_recon += coeffs[k] * (L_z[:, k+1] - L_z[:, k-1]) / 2.0

    return W_recon

# ---------------------------------------------------------
# 2. Path Generation and Animation Setup
# ---------------------------------------------------------
if __name__ == '__main__':
    torch.manual_seed(42)
    N = 10000
    s, t_end = 0.0, 10.0
    t = torch.linspace(s, t_end, N)
    dt = t[1] - t[0]
    dW = torch.randn(N - 1) * torch.sqrt(dt)
    W_true = torch.cat([torch.zeros(1), torch.cumsum(dW, dim=0)])

    K_frames = list(range(1, 20, 1)) + list(range(20, 100, 5)) + list(range(100, 301, 10))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Use subplots_adjust instead of tight_layout to prevent the title from getting cut off
    fig.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.95)

    ax.plot(t.numpy(), W_true.numpy(), color='black', alpha=0.3, label="True Brownian Motion", linewidth=1.5)
    approx_line, = ax.plot([], [], color='#d62728', linewidth=1.5, label=r"Legendre Approx $W^{(K)}_t$")

    ax.set_xlim(s, t_end)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("W(t)")
    ax.legend(loc='upper left')
    ax.grid(True)

    # ---------------------------------------------------------
    # 3. Animation Rendering Loop
    # ---------------------------------------------------------
    def update(frame_idx):
        K = K_frames[frame_idx]

        c_k = project_path_to_coeffs(W_true, t, K)
        W_approx = reconstruct_path_from_coeffs(c_k, t)

        approx_line.set_data(t.numpy(), W_approx.numpy())
        ax.set_title(f"Legendre Polynomial Expansion (Degree K = {K})", fontsize=14)

        # [FIX] Dynamically adapt the Y-axis to prevent the curve from getting cut off
        current_min = min(W_true.min().item(), W_approx.min().item())
        current_max = max(W_true.max().item(), W_approx.max().item())

        # Add a 1.5 margin to the top and bottom dynamically
        ax.set_ylim(current_min-0.3, current_max+0.3)

        return approx_line,

    ani = animation.FuncAnimation(
        fig, update, frames=len(K_frames), interval=1, blit=False
    )

    print("Generating and saving animation to GIF... (This may take a moment)")
    ani.save('legendre_brownian_approximation.gif', writer='pillow', fps=12)
    print("Successfully saved 'legendre_brownian_approximation.gif'.")
