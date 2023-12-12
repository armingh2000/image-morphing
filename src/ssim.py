from skimage.metrics import structural_similarity as ssim


def calculate_ssim_image(image1, image2):
    return ssim(image1, image2, data_range=255, multichannel=True)


def calculate_ssim_images(images, source, target, weights):
    ssim_values_source = [calculate_ssim_image(image, source) for image in images]
    ssim_values_target = [calculate_ssim_image(image, target) for image in images]

    assert len(ssim_values_source) == len(ssim_values_target) == len(weights["source"])

    ssim_values_weighted_average = [
        ssim_values_source[i] * weights["source"][i]
        + ssim_values_target[i] * weights["target"][i]
        for i in range(len(ssim_values_source))
    ]

    return ssim_values_weighted_average
