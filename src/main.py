from f_domain import *
from utils import *
import configs
from aligner import *


if __name__ == "__main__":
    # Set emotions for images
    emotion1, emotion2 = configs.neutral_image_path, configs.neutral_image_path

    # Set paths
    path1, path2 = str(emotion1 / random_file(emotion1)), str(
        emotion2 / random_file(emotion2)
    )

    # Load images
    img1, img2 = load_image(path1), load_image(path2)

    # Detect, align, and crop faces
    aligned_face1, _ = detect_and_align_face(img1, predictor)
    aligned_face2, _ = detect_and_align_face(img2, predictor)

    # Combine the faces if both are detected
    show_images(aligned_face1, aligned_face2, path1=path1, path2=path2)

    # # Perform DCT
    # dct1 = perform_dct(img1)
    # dct2 = perform_dct(img2)

    # # Perform IDCT
    # idct_image = perform_idct(dct_image)

    # # Show original, DCT, and IDCT images
    # show_images(
    #     [image, dct_image, idct_image], ["Original Image", "DCT Image", "IDCT Image"]
    # )
