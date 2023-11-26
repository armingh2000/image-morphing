import cv2
import dlib
import numpy as np
import configs


def detect_landmarks_points(gray):
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

    return landmarks_points


def level_face(gray, landmarks_points):
    # Correct landmarks for the corners of the eyes
    left_eye_corner = landmarks_points[36]  # Outer corner of the right eye
    right_eye_corner = landmarks_points[45]  # Outer corner of the left eye

    # Calculate the angle between the eye corners
    dY = right_eye_corner[1] - left_eye_corner[1]
    dX = right_eye_corner[0] - left_eye_corner[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Center between the eyes
    eyes_center = (
        (left_eye_corner[0] + right_eye_corner[0]) // 2,
        (left_eye_corner[1] + right_eye_corner[1]) // 2,
    )

    # Affine transformation for alignment
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned_face = cv2.warpAffine(
        gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_CUBIC
    )

    return aligned_face


def crop_face(gray, landmarks_points, coordinates=None):
    # Crop the face

    if coordinates is None:
        x, y, w, h = cv2.boundingRect(np.array(landmarks_points))
    else:
        x, y, w, h = coordinates

    cropped_face = gray[y : y + h, x : x + w]

    return cropped_face, (x, y, w, h)


def align_eyes(gray1, gray2, landmarks1, landmarks2):
    points1 = np.float32([landmarks1[36], landmarks1[45], landmarks1[33]])
    points2 = np.float32([landmarks2[36], landmarks2[45], landmarks2[33]])

    # Compute the affine transform matrix
    M = cv2.getAffineTransform(points2, points1)

    # Apply the affine transformation
    aligned_face2 = cv2.warpAffine(
        gray2, M, (gray2.shape[1], gray2.shape[0]), flags=cv2.INTER_CUBIC
    )

    return aligned_face2  # Returning only the aligned image


# Load the shape predictor
predictor = dlib.shape_predictor(
    str(configs.project_root / "src/shape_predictor_68_face_landmarks.dat")
)
