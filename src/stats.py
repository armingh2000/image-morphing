from f_domain import *
from utils import *
from ssim import calculate_ssim_images
from aligner import *
import random
from tqdm import tqdm
from configs import *


def generate_ssim_morphs(path1, path2, morph_type):
    # Load images
    img1, img2 = load_image(path1), load_image(path2)
    if img1 is None:
        raise (ValueError(f"img1 is none; path: {path1}"))
    if img2 is None:
        raise (ValueError(f"img2 is none; path: {path2}"))

    # Detect Landmarks
    landmarks1 = detect_landmarks_points(img1)
    landmarks2 = detect_landmarks_points(img2)

    # Level Face
    face1 = level_face(img1, landmarks1)
    face2 = level_face(img2, landmarks2)

    # Detect Landmarks
    landmarks1 = detect_landmarks_points(face1)
    landmarks2 = detect_landmarks_points(face2)

    # Align eyes of two faces
    face2 = align_eyes(face1, face2, landmarks1, landmarks2)

    # Detect Landmarks
    landmarks2 = detect_landmarks_points(face2)

    # Crop Face
    face1, coordinates = crop_face(face1, landmarks1)
    face2, _ = crop_face(face2, landmarks2, coordinates=coordinates)

    # Resize faces to 300x300
    face1 = resize_image(face1)
    face2 = resize_image(face2)

    # Perform morph
    morphs = morph(face1, face2, morph_type=morph_type)

    ssims = calculate_ssim_images(morphs, face1, face2, weights[morph_type])

    return ssims


def generate_ssim_stats(number_of_examples):
    combs = get_all_file_combinations(
        configs.neutral_image_path, configs.neutral_image_path
    )

    files = random.sample(combs, number_of_examples)
    dump_data(files, "stats/files.pkl")

    for morph_type in morph_types:
        stats = []

        for path1, path2 in tqdm(files):
            stats.append(generate_ssim_morphs(path1, path2, morph_type))

        stats = np.array(stats)
        dump_data(stats, f"stats/ssim_{morph_type}.pkl")
        final_stats = np.average(stats, axis=0)
        print(final_stats)


def generate_stats():
    generate_ssim_stats(number_of_examples)


def polar_weight(step, total_steps):
    return 0.5 * (1 - np.cos(np.pi * step / total_steps))


def gaussian_weight(step, total_steps):
    # Normalize the current step
    x = float(step) / total_steps

    # Calculate the Gaussian weight
    return np.exp(-((x - 0) ** 2) / (2 * 0.15**2))


total_steps = 7
polar_weights = {
    "source": [1 - polar_weight(i, total_steps) for i in range(total_steps + 1)],
    "target": [polar_weight(i, total_steps) for i in range(total_steps + 1)],
}
gaussian_weights = {
    "source": [gaussian_weight(i, total_steps) for i in range(total_steps + 1)],
    "target": [1 - gaussian_weight(i, total_steps) for i in range(total_steps + 1)],
}
morph_types = ["dft", "dct"]
weights = {"dft": polar_weights, "dct": gaussian_weights}


if __name__ == "__main__":
    generate_stats()
    # print(np.average(np.array(load_data("stats/dft.pkl")), axis=0))
    # print(np.average(np.array(load_data("stats/dft.pkl")), axis=0))
