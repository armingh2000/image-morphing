import matplotlib.pyplot as plt
import cv2
import os
import random
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import configs


def load_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def resize_image(image):
    return cv2.resize(image, (configs.img_height, configs.img_width))


def random_files(*dirs):
    results = []
    i = 0

    while i < len(dirs):
        dir = dirs[i]

        # List all files in the given directory
        files = [
            f
            for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f))
            if "jpg" in f
        ]

        # Randomly select a file
        if files:
            file = random.choice(files)
            if file not in results:
                results.append(str(dir / file))
                i += 1
            else:
                continue
        else:
            raise (ValueError(f"No files found in directory: {dir}"))

    return results


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


def show_images_slider(*faces, **paths):
    # Combine faces and do preprocessing steps
    images = combine_faces(*faces, **paths)
    titles = []
    assert len(images) >= 2

    for i in range(len(images)):
        titles.append(f"Step {i}")

    titles[0] = "Source"
    titles[-1] = "Target"

    # Set the figure size
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.3)
    # plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(images[0], cmap="gray", aspect="auto")
    ax.set_title(titles[0])
    ax.axis("off")

    # Slider setup
    ax_slider = plt.axes([0.20, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Step", 1, len(images), valinit=1, valfmt="%0.0f")

    # Update function for the slider
    def update(val):
        img_index = int(slider.val) - 1
        img_display.set_data(images[img_index])
        ax.set_title(titles[img_index])
        fig.canvas.draw_idle()

    # Animation function for bouncing effect
    def animate(i):
        # Calculate the next image index in a bouncing manner
        if i // (len(images) - 1) % 2 == 0:
            img_index = i % (len(images) - 1)
        else:
            img_index = len(images) - 1 - (i % (len(images) - 1))
        slider.set_val(img_index + 1)

    # Call update function when slider value is changed
    slider.on_changed(update)

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=2 * (len(images) - 1), interval=500, repeat=True
    )

    plt.show()


def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        cv2.circle(image, landmark, 2, (0, 255, 0), -1)

    return image
