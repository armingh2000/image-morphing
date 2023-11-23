import matplotlib.pyplot as plt
import cv2
import os
import random


def load_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def random_file(directory):
    # List all files in the given directory
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    # Randomly select a file
    if files:
        return random.choice(files)
    else:
        return None


def combine_faces(*faces, **paths):
    # Combine faces side by side
    combined = []

    for face in faces:
        if face is not None:
            combined.append(face)

        else:
            path1, path2 = paths["path1"], paths["path2"]
            raise (ValueError(f"face is none; path1: {path1}, path2: {path2}"))

    return combined


def show_images(*faces, **paths):
    images = combine_faces(*faces, **paths)
    titles = []
    assert len(faces) >= 2

    for i in range(len(faces)):
        titles.append(f"Step {i}")

    titles[0] = "Source"
    titles[-1] = "Target"

    # Display images in a row
    plt.figure(figsize=(12, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()
