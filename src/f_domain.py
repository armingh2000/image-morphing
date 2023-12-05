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


def dct_morph(img1, img2, steps=7):
    morphs = []

    # Perform DCT
    dct1 = perform_dct(img1)
    dct2 = perform_dct(img2)
    # morphs.append(img1)

    # Combine the DCT
    for i in range(steps + 1):
        dct_image = gaussian_transition2(dct1, dct2, steps, i)
        # Perform IDCT
        idct_image = perform_idct(dct_image)
        morphs.append(idct_image)

    # morphs.append(img2)

    return morphs
