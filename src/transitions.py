import numpy as np


def linear_transition(f1, f2, total_steps, step):
    morph = (f1 * ((total_steps - step) / total_steps)) + (f2 * (step / total_steps))

    return morph


def gaussian_transition(f1, f2, total_steps, step):
    m, n = f1.shape

    if step == 0:
        D0 = 0.001
    else:
        D0 = (step / total_steps) * ((m / 2) ** 2 + (n / 2) ** 2) ** 0.5

    # if step == 0:
    #     D0 = 0.001
    # else:
    #     D0 = 30 * step

    H = gaussian_low_pass_filter(m, n, D0)

    return (f1 - f2) * H + f2


def gaussian_transition2(f1, f2, total_steps, step):
    m, n = f1.shape

    D0 = [0.0001, 5, 7, 10, 25, 40, 60, 200][step]

    H = gaussian_low_pass_filter(m, n, D0)

    return (f1 - f2) * H + f2


def gaussian_low_pass_filter(M, N, d):
    """
    Create a Gaussian Low Pass Filter with the origin at the top-left corner.

    Parameters:
    M (int): Number of rows in the filter (typically image height).
    N (int): Number of columns in the filter (typically image width).
    d (float): Radius of the Gaussian low pass filter.

    Returns:
    numpy.ndarray: A 2D Gaussian Low Pass Filter.
    """
    # Create a 2D grid of coordinates
    u, v = np.meshgrid(np.arange(N), np.arange(M))

    # Compute the Gaussian Low Pass Filter
    D = np.sqrt(u**2 + v**2)
    GLPF = np.exp(-(D**2) / (2 * (d**2)))

    return GLPF


def phase_magnitude_interpolation(f1, f2, total_steps, step):
    if f1.shape != f2.shape:
        raise ValueError("Images must be the same size for morphing")

    # Compute magnitude and phase
    magnitude1, phase1 = np.abs(f1), np.angle(f1)
    magnitude2, phase2 = np.abs(f2), np.angle(f2)

    # alpha = step / total_steps # Linear interpolation
    alpha = 0.5 * (1 - np.cos(np.pi * step / total_steps))  # Cosine interpolation

    # Interpolate magnitude and phase separately

    # Linear Interpolation
    # interpolated_magnitude = (1 - alpha) * magnitude1 + alpha * magnitude2
    # interpolated_phase = (1 - alpha) * phase1 + alpha * phase2

    # Polar Interpolation
    interpolated_magnitude = np.abs(
        (1 - alpha) * (magnitude1 * np.exp(1j * phase1))
        + alpha * (magnitude2 * np.exp(1j * phase2))
    )
    interpolated_phase = np.angle(
        (1 - alpha) * (magnitude1 * np.exp(1j * phase1))
        + alpha * (magnitude2 * np.exp(1j * phase2))
    )

    # Reconstruct DFT with interpolated magnitude and phase
    interpolated_dft = interpolated_magnitude * np.exp(1j * interpolated_phase)

    return interpolated_dft
