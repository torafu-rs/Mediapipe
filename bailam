"""Track hand landmarks and motion using a webcam."""

import argparse
import importlib.machinery
import importlib.util
import sys
from collections import deque
from pathlib import Path

import cv2

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
OVERLAY_WIDTH = 250
OVERLAY_HEIGHT = 500
TWO_FINGER_OVERLAY_WIDTH = 640
TWO_FINGER_OVERLAY_HEIGHT = 360
TWO_HANDS_OVERLAY_HEIGHT = 700


def load_mediapipe():
    """Load the installed package without importing this script as the package."""
    script_directory = Path(__file__).resolve().parent
    package_paths = []

    for entry in sys.path:
        try:
            entry_path = Path(entry or ".").resolve()
        except OSError:
            continue
        if entry_path != script_directory:
            package_paths.append(entry)

    spec = importlib.machinery.PathFinder.find_spec("mediapipe", package_paths)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(
            "MediaPipe was not found. Install it with: "
            "python -m pip install mediapipe opencv-python"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules["mediapipe"] = module
    spec.loader.exec_module(module)
    return module


def distance_squared(first, second):
    return (
        (first.x - second.x) ** 2
        + (first.y - second.y) ** 2
        + (first.z - second.z) ** 2
    )


def resize_to_fit(image, max_width, max_height):
    image_height, image_width = image.shape[:2]
    scale = min(max_width / image_width, max_height / image_height)
    size = (
        max(1, int(image_width * scale)),
        max(1, int(image_height * scale)),
    )
    return cv2.resize(image, size)


def crop_to_aspect_ratio(frame, target_ratio):
    height, width = frame.shape[:2]
    frame_ratio = width / height
    if frame_ratio > target_ratio:
        cropped_width = int(height * target_ratio)
        left = (width - cropped_width) // 2
        return frame[:, left : left + cropped_width]

    cropped_height = int(width / target_ratio)
    top = (height - cropped_height) // 2
    return frame[top : top + cropped_height, :]


def is_finger_extended(landmarks, pip, tip):
    wrist = landmarks[0]
    return distance_squared(landmarks[tip], wrist) > (
        distance_squared(landmarks[pip], wrist) * 0.92
    )


def is_finger_straight(landmarks, mcp, pip, tip):
    joint = landmarks[pip]
    first = (landmarks[mcp].x - joint.x, landmarks[mcp].y - joint.y)
    second = (landmarks[tip].x - joint.x, landmarks[tip].y - joint.y)
    first_length = (first[0] ** 2 + first[1] ** 2) ** 0.5
    second_length = (second[0] ** 2 + second[1] ** 2) ** 0.5
    if first_length == 0 or second_length == 0:
        return False

    cosine = (
        first[0] * second[0] + first[1] * second[1]
    ) / (first_length * second_length)
    return cosine < -0.65


def is_finger_folded(landmarks, pip, tip):
    wrist = landmarks[0]
    return distance_squared(landmarks[tip], wrist) < (
        distance_squared(landmarks[pip], wrist) * 0.95
    )


def is_thumb_folded(landmarks):
    return distance_squared(landmarks[4], landmarks[9]) < distance_squared(
        landmarks[3], landmarks[9]
    )


def is_two_finger_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    return (
        is_finger_extended(landmarks, 6, 8)
        and is_finger_extended(landmarks, 10, 12)
        and is_finger_folded(landmarks, 14, 16)
        and is_finger_folded(landmarks, 18, 20)
    )


def is_index_pinky_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    return (
        is_finger_extended(landmarks, 6, 8)
        and is_finger_straight(landmarks, 5, 6, 8)
        and is_finger_extended(landmarks, 18, 20)
        and is_finger_straight(landmarks, 17, 18, 20)
        and is_finger_folded(landmarks, 10, 12)
        and is_finger_folded(landmarks, 14, 16)
        and is_thumb_folded(landmarks)
    )


def is_three_finger_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_closed = is_finger_folded(landmarks, 6, 8)
    middle_up = is_finger_extended(landmarks, 10, 12)
    ring_up = is_finger_extended(landmarks, 14, 16)
    pinky_up = is_finger_extended(landmarks, 18, 20)
    return index_closed and middle_up and ring_up and pinky_up


def is_open_hand(hand_landmarks):
    landmarks = hand_landmarks.landmark
    thumb_open = distance_squared(landmarks[4], landmarks[9]) > distance_squared(
        landmarks[3], landmarks[9]
    )
    fingers_open = [
        is_finger_extended(landmarks, pip, tip)
        for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20))
    ]

    return thumb_open and all(fingers_open)


def is_closed_hand(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers_closed = [
        is_finger_folded(landmarks, pip, tip)
        for pip, tip in ((6, 8), (10, 12), (14, 16), (18, 20))
    ]

    return is_thumb_folded(landmarks) and all(fingers_closed)


def main():
    parser = argparse.ArgumentParser(
        description="Show hand landmarks and movement from a webcam."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="webcam index (default: 0)",
    )
    args = parser.parse_args()

    images_directory = Path(__file__).with_name("images")
    domain_image_path = images_directory / "domain_expansion.png"
    domain_image = cv2.imread(str(domain_image_path))
    if domain_image is None:
        raise FileNotFoundError(f"Could not load gesture image: {domain_image_path}")

    two_finger_image_path = images_directory / "gesture_image.png"
    two_finger_image = cv2.imread(str(two_finger_image_path))
    if two_finger_image is None:
        raise FileNotFoundError(
            f"Could not load two-finger gesture image: {two_finger_image_path}"
        )
    two_finger_image = resize_to_fit(
        two_finger_image,
        TWO_FINGER_OVERLAY_WIDTH,
        TWO_FINGER_OVERLAY_HEIGHT,
    )

    two_hands_image_path = images_directory / "two_hands.png"
    two_hands_image = cv2.imread(str(two_hands_image_path))
    if two_hands_image is None:
        raise FileNotFoundError(
            f"Could not load two-hands gesture image: {two_hands_image_path}"
        )

    three_finger_image_path = images_directory / "domain_expansion_three_fingers.png"
    three_finger_image = cv2.imread(str(three_finger_image_path))
    if three_finger_image is None:
        raise FileNotFoundError(
            f"Could not load three-finger gesture image: "
            f"{three_finger_image_path}"
        )

    mp = load_mediapipe()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            f"Could not open camera {args.camera}. Check that it is connected "
            "or try another index with --camera."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    window_name = "MediaPipe - Hand Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    window_sized_to_camera = False
    gesture_history = {
        "left_open": deque(maxlen=3),
        "right_closed": deque(maxlen=3),
        "right_two_fingers": deque(maxlen=3),
        "right_three_fingers": deque(maxlen=3),
        "both_index_and_pinky": deque(maxlen=3),
    }
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                frame = cv2.flip(frame, 1)
                frame = crop_to_aspect_ratio(
                    frame, WINDOW_WIDTH / WINDOW_HEIGHT
                )
                height, width = frame.shape[:2]
                if not window_sized_to_camera:
                    camera_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    camera_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(
                        f"Camera resolution: {camera_width}x{camera_height}; "
                        f"display: {WINDOW_WIDTH}x{WINDOW_HEIGHT}"
                    )
                    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
                    window_sized_to_camera = True
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detected_gestures = {
                    "left_open": False,
                    "right_closed": False,
                    "right_two_fingers": False,
                    "right_three_fingers": False,
                    "both_index_and_pinky": False,
                }
                index_pinky_hands = set()

                if results.multi_hand_landmarks:
                    handedness_list = results.multi_handedness or []
                    for index, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        label = f"Hand {index + 1}"
                        if index < len(handedness_list):
                            label = handedness_list[index].classification[0].label

                        if label == "Left" and is_open_hand(hand_landmarks):
                            detected_gestures["left_open"] = True
                        if is_index_pinky_gesture(hand_landmarks):
                            index_pinky_hands.add(label)
                        if label == "Right":
                            if is_closed_hand(hand_landmarks):
                                detected_gestures["right_closed"] = True
                            if is_two_finger_gesture(hand_landmarks):
                                detected_gestures["right_two_fingers"] = True
                            if is_three_finger_gesture(hand_landmarks):
                                detected_gestures["right_three_fingers"] = True

                detected_gestures["both_index_and_pinky"] = {
                    "Left",
                    "Right",
                }.issubset(index_pinky_hands)

                for gesture, detected in detected_gestures.items():
                    gesture_history[gesture].append(detected)
                active_gestures = {
                    gesture: sum(history) >= 2
                    for gesture, history in gesture_history.items()
                }

                frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                height, width = WINDOW_HEIGHT, WINDOW_WIDTH
                if active_gestures["left_open"] and active_gestures["right_closed"]:
                    overlay = resize_to_fit(
                        domain_image,
                        min(OVERLAY_WIDTH, width // 2),
                        min(OVERLAY_HEIGHT, height),
                    )
                    overlay_height, overlay_width = overlay.shape[:2]
                    frame[
                        :overlay_height, width - overlay_width : width
                    ] = overlay
                if active_gestures["right_two_fingers"]:
                    overlay = resize_to_fit(
                        two_finger_image,
                        min(TWO_FINGER_OVERLAY_WIDTH, width // 2),
                        min(TWO_FINGER_OVERLAY_HEIGHT, height),
                    )
                    frame[: overlay.shape[0], : overlay.shape[1]] = overlay
                if active_gestures["both_index_and_pinky"]:
                    overlay = resize_to_fit(
                        two_hands_image,
                        min(width // 2, width),
                        min(TWO_HANDS_OVERLAY_HEIGHT, height),
                    )
                    overlay_height, overlay_width = overlay.shape[:2]
                    frame[
                        :overlay_height, width - overlay_width : width
                    ] = overlay
                if (
                    active_gestures["left_open"]
                    and active_gestures["right_three_fingers"]
                ):
                    overlay = resize_to_fit(
                        three_finger_image,
                        min(width // 2, width),
                        min(height // 2, height),
                    )
                    overlay_height, overlay_width = overlay.shape[:2]
                    frame[
                        height - overlay_height : height, :overlay_width
                    ] = overlay

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
