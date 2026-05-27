"""Differentiable pendulum swing-up.

Physics: a simple damped pendulum, theta = 0 hanging straight down.

    theta_ddot = -(g / L) * sin(theta) - c * theta_dot + u / (m * L^2)

We unroll K Euler steps in pure micro-jax, parameterize a torque schedule
u in R^K, and minimize a cost that wants theta(T) at the top (pi) with
small terminal angular velocity. `grad` differentiates through the entire
unrolled simulation; `jit` compiles the whole forward + reverse pass into
a single flat program.

Trick: micro-jax has no integer indexing, so "u[k]" is implemented as
`sum(u * one_hot_k)`. The one-hots are closure-captured constants — jit
folds them away, and the resulting program is just numpy arithmetic.

Output: a single PNG showing
  1. angle over time (before training vs. after training),
  2. the optimized torque schedule,
  3. a stroboscopic view of the pendulum swinging up.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import micro_jax as mjx
from micro_jax import Array
from micro_jax.jit import CompiledProgram


# --- physics ---------------------------------------------------------------
G = 9.81
L = 1.0
M = 1.0
DAMPING = 0.1
DT = 0.1
K = 40                              # 4 seconds of simulation
TARGET = np.pi                      # upright

# --- cost weights ----------------------------------------------------------
W_ANGLE = 10.0                       # terminal angle penalty
W_VEL = 0.5                          # terminal angular velocity penalty
W_CTRL = 0.001                       # control-effort regularizer


def print_program(prog: CompiledProgram, title: str, max_lines: int = 80) -> None:
    """Pretty-print a compiled jit program with renamed local ids."""
    rename: dict[int, str] = {}

    def n(i: int) -> str:
        if i not in rename:
            rename[i] = f"v{len(rename)}"
        return rename[i]

    for i in prog.input_ids:
        n(i)
    for i in prog.constants:
        n(i)

    counts = Counter(e.prim.name for e in prog.entries)
    op_histogram = ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    print(f"\n=== {title} ===")
    print(f"  {len(prog.entries)} ops, {len(prog.constants)} constants, "
          f"{len(prog.input_ids)} inputs")
    print(f"  op histogram: {op_histogram}")

    lines = []
    for e in prog.entries:
        out = n(e.output_id)
        ins = ", ".join(n(i) for i in e.input_ids)
        ps = ", ".join(f"{k}={v}" for k, v in e.params.items())
        suffix = f" [{ps}]" if ps else ""
        lines.append(f"  {out:>5} = {e.prim.name:<12}({ins}){suffix}")

    if len(lines) <= max_lines:
        print("\n".join(lines))
    else:
        head = max_lines // 2
        tail = max_lines - head
        print("\n".join(lines[:head]))
        print(f"  ... ({len(lines) - max_lines} ops elided) ...")
        print("\n".join(lines[-tail:]))
    print(f"  output:    {n(prog.output_id)}")


# One-hot basis for indexing into the torque schedule. Each `basis[k]`
# is an Array of shape (K,) with a 1 at position k. Because these are
# captured from a closure, jit treats them as compile-time constants.
BASIS = [Array(np.eye(K, dtype=np.float32)[k]) for k in range(K)]


def _scalar(x: float) -> Array:
    return Array(np.float32(x))


def get_uk(u: Array, k: int) -> Array:
    """Extract u[k] without indexing — just a masked sum."""
    return mjx.sum(u * BASIS[k])


def simulate_and_loss(u: Array) -> Array:
    """Roll out K Euler steps from rest and return the trajectory cost."""
    theta = _scalar(0.0)
    theta_dot = _scalar(0.0)

    ctrl_cost = _scalar(0.0)
    dt = _scalar(DT)
    inertia = _scalar(M * L * L)
    grav = _scalar(G / L)
    damp = _scalar(DAMPING)
    w_ctrl = _scalar(W_CTRL)

    for k in range(K):
        uk = get_uk(u, k)
        theta_ddot = -grav * mjx.sin(theta) - damp * theta_dot + uk / inertia
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * theta_ddot
        ctrl_cost = ctrl_cost + w_ctrl * uk * uk

    # 1 + cos(theta) is a smooth surrogate for (theta - pi)^2 that doesn't
    # have spurious minima at theta = pi + 2*pi*n.
    angle_cost = _scalar(W_ANGLE) * (_scalar(1.0) + mjx.cos(theta))
    vel_cost = _scalar(W_VEL) * theta_dot * theta_dot
    return angle_cost + vel_cost + ctrl_cost


# --- a pure-numpy rollout for plotting/visualization -----------------------
def rollout_numpy(u_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.zeros(K + 1, dtype=np.float32)
    theta_dots = np.zeros(K + 1, dtype=np.float32)
    for k in range(K):
        ddot = (
            -(G / L) * np.sin(thetas[k])
            - DAMPING * theta_dots[k]
            + u_np[k] / (M * L * L)
        )
        thetas[k + 1] = thetas[k] + DT * theta_dots[k]
        theta_dots[k + 1] = theta_dots[k] + DT * ddot
    return thetas, theta_dots


def main() -> None:
    out_path = Path(__file__).parent / "pendulum_control.png"

    # --- optimize the torque schedule --------------------------------------
    # IMPORTANT: u = 0 is an exact saddle point. With the pendulum at rest
    # at theta = 0, sin(theta_K) is exactly zero and the gradient of the
    # terminal angle cost vanishes. We start from small random torques so
    # the first gradient has signal.
    rng = np.random.RandomState(0)
    u = Array((rng.randn(K) * 0.6).astype(np.float32))
    u_initial_np = np.copy(u.data)

    grad_loss = mjx.jit(mjx.grad(simulate_and_loss))

    velocity = np.zeros(K, dtype=np.float32)
    lr = 0.05
    mom = 0.92
    steps = 1500

    t0 = time.perf_counter()
    for step in range(steps):
        g = grad_loss(u).data
        velocity = mom * velocity - lr * g
        u = Array(u.data + velocity)
        if step % 100 == 0 or step == steps - 1:
            cost = float(simulate_and_loss(u))
            gn = float(np.linalg.norm(g))
            print(f"step {step:>4}: cost = {cost:8.4f}   |grad| = {gn:.3e}")
    print(f"optimization took {time.perf_counter() - t0:.2f}s")

    # --- show the compiled bytecode of grad_loss ---------------------------
    compiled = next(iter(grad_loss._cache.values()))
    print_program(compiled, "jit'd grad(simulate_and_loss) bytecode", max_lines=60)

    u_final_np = np.copy(u.data)

    # --- run both schedules through the numpy simulator --------------------
    thetas_before, _ = rollout_numpy(u_initial_np)
    thetas_after, theta_dots_after = rollout_numpy(u_final_np)
    t_axis = np.arange(K + 1) * DT

    print(f"\nfinal theta:     {thetas_after[-1]:+.4f}  (target {TARGET:+.4f})")
    print(f"final theta_dot: {theta_dots_after[-1]:+.4f}")
    print(f"peak |torque|:   {np.max(np.abs(u_final_np)):.3f}")

    # --- plotting ----------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.35, wspace=0.25)

    ax_angle = fig.add_subplot(gs[0, 0])
    ax_angle.plot(t_axis, thetas_before, "--", color="gray", label="initial (u=0)")
    ax_angle.plot(t_axis, thetas_after, "-", color="C0", label="optimized")
    ax_angle.axhline(TARGET, color="C3", linestyle=":", label="target (pi)")
    ax_angle.set_xlabel("time (s)")
    ax_angle.set_ylabel("theta (rad)")
    ax_angle.set_title("Pendulum angle")
    ax_angle.legend(loc="lower right", fontsize=9)
    ax_angle.grid(alpha=0.3)

    ax_torque = fig.add_subplot(gs[0, 1])
    ax_torque.step(t_axis[:-1], u_final_np, where="post", color="C2")
    ax_torque.axhline(0, color="black", linewidth=0.6)
    ax_torque.set_xlabel("time (s)")
    ax_torque.set_ylabel("torque u(t)")
    ax_torque.set_title("Optimized torque schedule")
    ax_torque.grid(alpha=0.3)

    # Stroboscopic view: draw the pendulum at evenly-spaced snapshots.
    ax_strobe = fig.add_subplot(gs[1, :])
    n_frames = 14
    sample_idx = np.linspace(0, K, n_frames).astype(int)
    spacing = 2.4
    for i, k in enumerate(sample_idx):
        x_pivot = i * spacing
        th = thetas_after[k]
        x_end = x_pivot + np.sin(th) * L
        y_end = -np.cos(th) * L
        color_t = plt.cm.viridis(i / max(1, n_frames - 1))
        ax_strobe.plot([x_pivot, x_end], [0, y_end], "-", color=color_t, linewidth=2.2)
        ax_strobe.plot(x_pivot, 0, "o", color="black", markersize=4)
        ax_strobe.plot(x_end, y_end, "o", color=color_t, markersize=9)
        ax_strobe.text(
            x_pivot, -1.35, f"t={k * DT:.1f}s",
            ha="center", fontsize=8, color="gray",
        )
    ax_strobe.set_xlim(-1, n_frames * spacing - 1)
    ax_strobe.set_ylim(-1.55, 1.35)
    ax_strobe.set_aspect("equal")
    ax_strobe.set_title("Stroboscopic view of the swing-up (left to right)")
    ax_strobe.set_xticks([])
    ax_strobe.set_yticks([])
    for spine in ax_strobe.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "micro-jax: gradient-based pendulum swing-up "
        f"(K={K} steps, jit'd grad through full simulation)",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot to {out_path}")


if __name__ == "__main__":
    main()
