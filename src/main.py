from f_domain import *
from utils import *
import configs
from aligner import *


if __name__ == "__main__":
    # Set emotions for images
    emotion1, emotion2 = configs.neutral_image_path, configs.neutral_image_path

    # Set paths
    # paths = random_files(emotion1, emotion2)
    paths = [
        "/home/subroot/Personal/university/dip/image-morphing/data/front/neutral_front/125_03.jpg",
        "/home/subroot/Personal/university/dip/image-morphing/data/front/neutral_front/129_03.jpg",
    ]

    # Load images
    img1, img2 = load_image(paths[0]), load_image(paths[1])

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

    # face1 = draw_landmarks(face1, landmarks1)
    # face2 = draw_landmarks(face2, landmarks2)

    # show_images(face1, face2, path1=paths[0], path2=paths[1])

    # Perform morph
    morphs = dct_morph(face1, face2)

    # Combine the faces if both are detected
    # # show_images(*morphs, path1=paths[0], path2=paths[1])
    show_images_slider(*morphs, path1=paths[0], path2=paths[1])
