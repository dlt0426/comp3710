"""Generate the Sierpinski triangle using the bitwise/arithmetic method in PyTorch.

The whole fractal comes from a single number-theoretic fact: a pixel at
integer coordinates (x, y) belongs to the Sierpinski triangle if and only if
the bitwise AND of x and y is zero, i.e. (x & y) == 0. The binary digits of
the coordinates encode the self-similar structure directly.

Because each pixel depends only on its own coordinates -- never on a neighbor
or on any previous step -- the entire image is computed in one elementwise
tensor operation, with no Python loop over pixels. This makes it a perfect
fit for the GPU.
"""

import torch
import matplotlib.pyplot as plt


def sierpinski_bitwise(order=9, device=None):
    """Compute a Sierpinski triangle image with the bitwise method.

    Parameters
    ----------
    order : int
        Controls resolution. The image is a square of side 2**order, which
        keeps the fractal's binary self-similarity exact (powers of two line
        up perfectly with the bit pattern).
    device : torch.device or None
        Device to run on. Defaults to CUDA if available, else CPU.

    Returns
    -------
    torch.Tensor
        A (size, size) boolean tensor; True where the pixel is in the set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = 2 ** order

    # 1D ranges of integer coordinates [0, 1, ..., size-1], on the device.
    coords = torch.arange(size, device=device)

    # Build 2D coordinate grids so every pixel knows its own (x, y).
    #   X[i, j] = j  (column index / x coordinate)
    #   Y[i, j] = i  (row index    / y coordinate)
    X, Y = torch.meshgrid(coords, coords, indexing="xy")

    # The entire fractal in one parallel elementwise operation:
    # a pixel is "on" exactly when x AND y has no shared set bits.
    # bitwise_and works on integer tensors across the whole grid at once.
    in_set = torch.bitwise_and(X, Y) == 0

    return in_set


def main():
    # 2) Device configuration: use the GPU when one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Compute the fractal (size = 2**order per side).
    image = sierpinski_bitwise(order=9, device=device)

    # 3) Move back to CPU and convert to NumPy for Matplotlib.
    #    .cpu() is a no-op if we were already on the CPU.
    image_np = image.cpu().numpy()

    # Plot. origin="upper" puts the right angle of the triangle at the
    # top-left, which is the orientation the (x & y) pattern produces.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np, cmap="binary", origin="upper", interpolation="nearest")
    ax.set_title("Sierpinski Triangle (bitwise method, PyTorch GPU)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("sierpinski_bitwise.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
