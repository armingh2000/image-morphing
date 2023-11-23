import cv2
import dlib
import numpy as np
import configs


def detect_and_align_face(gray, predictor):
    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1)

    if len(faces) > 0:
        face = faces[0]  # Assuming there is only one face
        landmarks = predictor(gray, face)

        # Extracting the landmarks
        landmarks_points = [
            (landmarks.part(n).x, landmarks.part(n).y) for n in range(68)
        ]

        # Crop and align face
        aligned_face, crop_coordinates = align_and_crop_face(gray, landmarks_points)

        return aligned_face, crop_coordinates
    else:
        return None, None


def align_and_crop_face(gray, landmarks_points):
    # Calculating rotation based on eye positions
    left_eye_corner = landmarks_points[37]
    right_eye_corner = landmarks_points[43]
    dY = right_eye_corner[1] - left_eye_corner[1]
    dX = right_eye_corner[0] - left_eye_corner[0]
    angle = np.degrees(np.arctan2(dY, dX))
    eyes_center = (
        (left_eye_corner[0] + right_eye_corner[0]) // 2,
        (left_eye_corner[1] + right_eye_corner[1]) // 2,
    )

    # Affine transformation for alignment
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_face = cv2.warpAffine(
        gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC
    )

    # Crop the face
    x, y, w, h = cv2.boundingRect(np.array(landmarks_points))
    cropped_face = aligned_face[y : y + h, x : x + w]

    return cropped_face, (x, y, w, h)


def resize_images(image1, image2):
    # Determine the minimum dimensions
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])
    output_size = (min_width, min_height)

    # Resize images
    resized_image1 = cv2.resize(image1, output_size)
    resized_image2 = cv2.resize(image2, output_size)

    return resized_image1, resized_image2


# Load the shape predictor
predictor = dlib.shape_predictor(
    str(configs.project_root / "src/shape_predictor_68_face_landmarks.dat")
)
