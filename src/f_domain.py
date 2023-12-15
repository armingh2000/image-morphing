import scipy.fftpack as fftpack
from transitions import *


def perform_dct(image):
    # Perform the 2D DCT
    dct = fftpack.dct(fftpack.dct(image.T, norm="ortho").T, norm="ortho")
    return dct


def perform_idct(dct_image):
    # Perform the 2D inverse DCT
    idct = fftpack.idct(fftpack.idct(dct_image.T, norm="ortho").T, norm="ortho")
    return idct


def dct_morph(img1, img2, total_steps=7):
    morphs = []

    # Perform DCT
    dct1 = perform_dct(img1)
    dct2 = perform_dct(img2)
    # morphs.append(img1)

    # Combine the DCT
    for i in range(total_steps + 1):
        dct_image = gaussian_transition(dct1, dct2, total_steps, i)
        # Perform IDCT
        idct_image = perform_idct(dct_image)
        idct_image = np.clip(idct_image, 0, 255)
        morphs.append(idct_image)

    # morphs.append(img2)

    return morphs


def perform_dft(image):
    # Perform the 2D DFT
    dft = fftpack.fft2(image)
    return dft


def perform_idft(dft_image):
    # Perform the 2D inverse DFT
    idft = fftpack.ifft2(dft_image)
    return idft


def dft_morph(img1, img2, total_steps=7):
    morphs = []

    # Perform DFT
    dft1 = perform_dft(img1)
    dft2 = perform_dft(img2)

    # Combine the DFT
    for i in range(total_steps + 1):
        dft_image = phase_magnitude_interpolation(dft1, dft2, total_steps, i)

        # Perform IDFT
        idft_image = np.abs(perform_idft(dft_image))

        # Normalize or scale pixel values if necessary
        idft_image = np.clip(idft_image, 0, 255)  # Clip values to the range [0, 255]
        # dft_image = np.real(dft_image).astype(np.uint8)  # Convert to uint8 type if needed

        morphs.append(idft_image)

    return morphs


def morph(img1, img2, total_steps=7, morph_type="dft"):
    if morph_type == "dft":
        return dft_morph(img1, img2, total_steps)
    elif morph_type == "dct":
        return dct_morph(img1, img2, total_steps)
