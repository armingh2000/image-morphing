import scipy.fftpack as fftpack


def perform_dct(image):
    # Perform the 2D DCT
    dct = fftpack.dct(fftpack.dct(image.T, norm="ortho").T, norm="ortho")
    return dct


def perform_idct(dct_image):
    # Perform the 2D inverse DCT
    idct = fftpack.idct(fftpack.idct(dct_image.T, norm="ortho").T, norm="ortho")
    return idct


def linear_transition(img1, img2, steps, step):
    morph = (img1 * ((steps - step) / steps)) + (img2 * (step / steps))

    return morph


def dct_morph(img1, img2, steps):
    morphs = []

    # Perform DCT
    dct1 = perform_dct(img1)
    dct2 = perform_dct(img2)

    # Combine the DCT
    for i in range(steps + 1):
        dct_image = linear_transition(dct1, dct2, steps, i)
        # Perform IDCT
        idct_image = perform_idct(dct_image)
        morphs.append(idct_image)

    return morphs
