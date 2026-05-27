"""Karpathy-style char-rnn, ported to micro-jax.

Trains a vanilla RNN character-by-character on Karpathy's tinyshakespeare
corpus (~1.1M characters, 65 unique symbols) with backprop-through-time
over T=25 unrolled steps. We `jit` one gradient function per parameter
(5 total); each compiled program contains the entire forward + reverse
pass through the unrolled RNN.

Limitations imposed by micro-jax:
* No integer indexing -> inputs/targets are one-hot vectors; the "row of X
  at time t" is extracted via a masked-sum trick. Captured masks become
  jit constants.
* No `max` primitive -> log-sum-exp runs without the max-subtraction
  stability trick. Logits stay bounded for the sizes we use here.
* No `argmax` -> sampling/decoding happens in plain numpy with the trained
  parameter buffers; we don't differentiate through it.

Outputs:
* generated text samples printed at several training stages,
* `char_rnn_loss.png` — loss curve,
* `char_rnn_transitions.png` — P(next_char | current_char), h=0 heatmap.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import micro_jax as mjx
from micro_jax import Array


# --- corpus and vocabulary --------------------------------------------------
HERE = Path(__file__).parent
CORPUS_PATH = HERE / "tinyshakespeare.txt"
CORPUS = CORPUS_PATH.read_text(encoding="utf-8")

VOCAB = sorted(set(CORPUS))
V = len(VOCAB)
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
ONEHOT = np.eye(V, dtype=np.float32)

# --- hyperparameters --------------------------------------------------------
H = 128                      # hidden size
T = 25                       # BPTT length
STEPS = 30000                # training steps
LR =0.02                    # learning rate
GRAD_CLIP = 5.0              # max L2 norm per gradient
SAMPLE_AT = (0, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 12000, 14000, 160000, 20000, 25000, 26000, STEPS - 1)
SAMPLE_LEN = 50             # chars to generate per sample
SAMPLE_SEED = "t"     # priming text — initializes the hidden state


# --- "indexing" via one-hot sums -------------------------------------------
# ROW_MASKS[t] selects row t of an (T, V) Array when broadcast-multiplied
# and summed along axis 0. Captured from closure -> jit treats as constant.
ROW_MASKS = [
    Array(np.eye(T, dtype=np.float32)[t].reshape(T, 1)) for t in range(T)
]


def row_t(X: Array, t: int) -> Array:
    """Pull row t out of an (T, V) tensor as a (1, V) tensor."""
    return mjx.sum(X * ROW_MASKS[t], axis=0, keepdims=True)


# --- RNN primitives --------------------------------------------------------
def tanh(x: Array) -> Array:
    ep = mjx.exp(x)
    en = mjx.exp(-x)
    return (ep - en) / (ep + en)


def cross_entropy(logits: Array, target_onehot: Array) -> Array:
    """Cross-entropy of a single timestep. logits, target: (1, V)."""
    log_z = mjx.log(mjx.sum(mjx.exp(logits), axis=1, keepdims=True))  # (1,1)
    log_softmax = logits - log_z                                       # (1,V)
    return -mjx.sum(target_onehot * log_softmax)


def rnn_step(
    W_xh: Array, W_hh: Array, W_hy: Array, b_h: Array, b_y: Array,
    x: Array, h: Array,
) -> tuple[Array, Array]:
    h_new = tanh(mjx.matmul(x, W_xh) + mjx.matmul(h, W_hh) + b_h)
    y = mjx.matmul(h_new, W_hy) + b_y
    return h_new, y


def loss_fn(
    W_xh: Array, W_hh: Array, W_hy: Array, b_h: Array, b_y: Array,
    X: Array, Y: Array,
) -> Array:
    """BPTT loss over T unrolled timesteps."""
    h = Array(np.zeros((1, H), dtype=np.float32))
    total = Array(np.float32(0.0))
    for t in range(T):
        x_t = row_t(X, t)
        h, y = rnn_step(W_xh, W_hh, W_hy, b_h, b_y, x_t, h)
        total = total + cross_entropy(y, row_t(Y, t))
    return total


# --- pure-numpy sampling ---------------------------------------------------
def sample(
    W_xh: np.ndarray, W_hh: np.ndarray, W_hy: np.ndarray,
    b_h: np.ndarray, b_y: np.ndarray,
    seed: str, n: int, temperature: float, rng: np.random.RandomState,
) -> str:
    h = np.zeros((1, H), dtype=np.float32)
    out = [seed[0]]
    x = ONEHOT[CHAR_TO_IDX[seed[0]]:CHAR_TO_IDX[seed[0]] + 1]
    # Prime the hidden state with any extra seed characters.
    for c in seed[1:]:
        h = np.tanh(x @ W_xh + h @ W_hh + b_h)
        out.append(c)
        x = ONEHOT[CHAR_TO_IDX[c]:CHAR_TO_IDX[c] + 1]
    for _ in range(n):
        h = np.tanh(x @ W_xh + h @ W_hh + b_h)
        logits = (h @ W_hy + b_y)[0] / max(1e-6, temperature)
        logits = logits - logits.max()                      # numpy stability
        p = np.exp(logits)
        p = p / p.sum()
        idx = int(rng.choice(V, p=p))
        out.append(IDX_TO_CHAR[idx])
        x = ONEHOT[idx:idx + 1]
    return "".join(out)


def transition_matrix(
    W_xh: np.ndarray, W_hh: np.ndarray, W_hy: np.ndarray,
    b_h: np.ndarray, b_y: np.ndarray,
) -> np.ndarray:
    """P(next | current) with hidden state initialized to zero — i.e. what
    the model thinks follows each character with no history."""
    P = np.zeros((V, V), dtype=np.float32)
    h0 = np.zeros((1, H), dtype=np.float32)
    for i in range(V):
        x = ONEHOT[i:i + 1]
        h = np.tanh(x @ W_xh + h0 @ W_hh + b_h)
        logits = (h @ W_hy + b_y)[0]
        logits = logits - logits.max()
        p = np.exp(logits)
        P[i] = p / p.sum()
    return P


# --- training loop ---------------------------------------------------------
def clip(g: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(g))
    return g * (max_norm / n) if n > max_norm else g


def main() -> None:
    here = Path(__file__).parent
    print(f"corpus: {len(CORPUS)} chars, vocab: {V} symbols")
    print(f"vocab: {''.join(VOCAB)!r}")
    print(f"network: V={V}, H={H}, T={T}\n")

    rng = np.random.RandomState(42)
    W_xh = Array((rng.randn(V, H) * 0.1).astype(np.float32))
    W_hh = Array((rng.randn(H, H) * 0.1).astype(np.float32))
    W_hy = Array((rng.randn(H, V) * 0.1).astype(np.float32))
    b_h = Array(np.zeros((H,), dtype=np.float32))
    b_y = Array(np.zeros((V,), dtype=np.float32))

    grad_fns = [mjx.jit(mjx.grad(loss_fn, argnum=i)) for i in range(5)]
    loss_jit = mjx.jit(loss_fn)  # for cheap progress reporting

    loss_history: list[tuple[int, float]] = []
    sample_log: list[tuple[int, str]] = []

    sample_rng = np.random.RandomState(0)
    lr_arr = Array(np.float32(LR))

    t_start = time.perf_counter()
    for step in range(STEPS):
        offset = rng.randint(0, len(CORPUS) - T - 1)
        seq = CORPUS[offset:offset + T + 1]
        X_np = np.stack([ONEHOT[CHAR_TO_IDX[c]] for c in seq[:T]])
        Y_np = np.stack([ONEHOT[CHAR_TO_IDX[c]] for c in seq[1:T + 1]])
        X = Array(X_np)
        Y = Array(Y_np)

        if step % 50 == 0:
            loss_val = float(loss_jit(W_xh, W_hh, W_hy, b_h, b_y, X, Y))
            loss_history.append((step, loss_val))
            if step % 200 == 0:
                print(f"step {step:>4}: loss = {loss_val:.4f}")

        # Compute and clip gradients.
        gs = [fn(W_xh, W_hh, W_hy, b_h, b_y, X, Y).data for fn in grad_fns]
        gs = [clip(g, GRAD_CLIP) for g in gs]

        W_xh = Array(W_xh.data - LR * gs[0])
        W_hh = Array(W_hh.data - LR * gs[1])
        W_hy = Array(W_hy.data - LR * gs[2])
        b_h = Array(b_h.data - LR * gs[3])
        b_y = Array(b_y.data - LR * gs[4])

        if step in SAMPLE_AT:
            text = sample(
                W_xh.data, W_hh.data, W_hy.data, b_h.data, b_y.data,
                seed=SAMPLE_SEED, n=SAMPLE_LEN, temperature=0.8,
                rng=sample_rng,
            )
            sample_log.append((step, text))
            print(f"--- sample @ step {step} ---")
            print(text)
            print()

    print(f"\ntotal training time: {time.perf_counter() - t_start:.2f}s")
    print(f"final loss (last bucket): {loss_history[-1][1]:.4f}")

    # Show one compiled program for context.
    prog = next(iter(grad_fns[0]._cache.values()))
    print(
        f"compiled grad(W_xh) program: {len(prog.entries)} ops, "
        f"{len(prog.constants)} constants"
    )

    # --- plot 1: loss curve -----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    xs, ys = zip(*loss_history)
    ax.plot(xs, ys, color="C0")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (sum CE over T=25 steps)")
    ax.set_title(f"char-rnn training loss  (H={H}, T={T}, V={V})")
    ax.grid(alpha=0.3)
    loss_path = here / "char_rnn_loss.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved loss curve to {loss_path}")

    # --- plot 2: transition heatmap ---------------------------------------
    P = transition_matrix(W_xh.data, W_hh.data, W_hy.data, b_h.data, b_y.data)

    def pretty(c: str) -> str:
        if c == "\n":
            return "\\n"
        if c == " ":
            return "_"
        return c

    labels = [pretty(c) for c in VOCAB]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(P, aspect="auto", cmap="magma", vmin=0.0, vmax=P.max())
    ax.set_xticks(range(V))
    ax.set_yticks(range(V))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("next character")
    ax.set_ylabel("current character")
    ax.set_title("P(next | current),  hidden state = 0")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    heatmap_path = here / "char_rnn_transitions.png"
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved transition heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()
