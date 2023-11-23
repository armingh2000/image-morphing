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
    print(images)
    assert len(images) >= 2

    for i in range(len(images)):
        titles.append(f"Step {i}")

    titles[0] = "Source"
    titles[-1] = "Target"

    # Validate the number of columns
    cols = 3
    if cols < 1:
        cols = 1
    elif cols > len(images):
        cols = len(images)

    # Calculate rows needed
    rows = (len(images) + cols - 1) // cols

    # Display images
    plt.figure(
        figsize=(6 * cols, 5 * rows)
    )  # Adjust figure size based on rows and cols
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.show()
