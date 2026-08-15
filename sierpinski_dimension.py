"""Estimate the fractal (box-counting) dimension of the Sierpinski triangle.

Box-counting works by covering the fractal with a grid of boxes of side eps
and counting how many boxes N(eps) contain at least one point of the set.
For a self-similar fractal, N(eps) scales like (1/eps)**D, so

        D = lim_{eps -> 0}  log N(eps) / log (1/eps)

and D is recovered as the slope of log N(eps) against log(1/eps).

Everything here stays vectorized and GPU-aware: the fractal is built with the
bitwise method, and each box count is done with a reshape + any() over blocks
rather than a Python loop over boxes.
"""

import torch
import matplotlib.pyplot as plt


def sierpinski_bitwise(order, device):
    """Boolean (2**order, 2**order) grid; True where (x & y) == 0."""
    size = 2 ** order
    coords = torch.arange(size, device=device)
    X, Y = torch.meshgrid(coords, coords, indexing="xy")
    return torch.bitwise_and(X, Y) == 0


def box_count(grid, box):
    """Count boxes of side `box` pixels that contain at least one True pixel.

    The grid is reshaped into (n, box, n, box) blocks and we ask whether each
    block contains any set pixel. This is fully vectorized -- no loop over
    the individual boxes.
    """
    size = grid.shape[0]
    n = size // box                      # boxes per side
    blocks = grid[:n * box, :n * box].reshape(n, box, n, box)
    occupied = blocks.any(dim=(1, 3))    # (n, n): True if block is non-empty
    return int(occupied.sum().item())


def estimate_dimension(order=11, device=None):
    """Return (box_sizes, counts, estimated_dimension)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grid = sierpinski_bitwise(order, device)

    # Use box sides that are powers of two so they align with the fractal's
    # binary self-similarity. Skip the very largest/smallest to avoid the
    # trivial single-box and single-pixel regimes.
    box_sizes = [2 ** k for k in range(order - 1, 0, -1)]  # e.g. 1024 ... 2
    counts = [box_count(grid, b) for b in box_sizes]

    # Fit log N(eps) vs log(1/eps); the slope is the box-counting dimension.
    eps = torch.tensor(box_sizes, dtype=torch.float64)
    n_eps = torch.tensor(counts, dtype=torch.float64)
    x = torch.log(1.0 / eps)
    y = torch.log(n_eps)

    # Least-squares slope (closed form) done with tensor ops.
    x_mean, y_mean = x.mean(), y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()

    return box_sizes, counts, float(slope)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    box_sizes, counts, dim = estimate_dimension(order=11, device=device)

    theoretical = torch.log(torch.tensor(3.0)) / torch.log(torch.tensor(2.0))
    print(f"\n{'box side':>9} {'N(eps)':>10}")
    for b, c in zip(box_sizes, counts):
        print(f"{b:>9} {c:>10}")
    print(f"\nEstimated dimension:        {dim:.4f}")
    print(f"Theoretical log(3)/log(2):  {theoretical.item():.4f}")

    # Visualize the log-log fit whose slope IS the dimension.
    import math
    xs = [math.log(1.0 / b) for b in box_sizes]
    ys = [math.log(c) for c in counts]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xs, ys, "o", label="box counts")
    # Line with the fitted slope through the data's mean point.
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_line = [min(xs), max(xs)]
    y_line = [y_mean + dim * (xv - x_mean) for xv in x_line]
    ax.plot(x_line, y_line, "-", label=f"slope = {dim:.4f}")

    ax.set_xlabel("log(1 / box size)")
    ax.set_ylabel("log N(box size)")
    ax.set_title("Box-counting dimension of the Sierpinski triangle")
    ax.legend()

    plt.tight_layout()
    plt.savefig("sierpinski_dimension.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
