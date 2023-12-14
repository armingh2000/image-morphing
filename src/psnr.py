from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr_image(image_true, image_test):
    return psnr(image_true, image_test, data_range=255)


def calculate_psnr_images(images, source, target, weights):
    psnr_values_source = [calculate_psnr_image(source, image) for image in images]
    psnr_values_target = [calculate_psnr_image(target, image) for image in images]

    assert len(psnr_values_source) == len(psnr_values_target) == len(weights["source"])

    psnr_values_weighted_average = [
        psnr_values_source[i] * weights["source"][i]
        + psnr_values_target[i] * weights["target"][i]
        for i in range(len(psnr_values_source))
    ]

    return psnr_values_weighted_average
