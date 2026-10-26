"""Reconstructed initial hand-tracking demo from the conversation."""

import argparse
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import cv2

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
OVERLAY_WIDTH = 320
OVERLAY_HEIGHT = 180


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


def is_two_finger_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    return index_up and middle_up and ring_down and pinky_down


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="movement threshold in pixels per frame (default: 8)",
    )
    args = parser.parse_args()

    image_path = (
        Path(__file__).resolve().parent.parent
        / "images"
        / "gesture_image.png"
    )
    gesture_image = cv2.imread(str(image_path))
    if gesture_image is None:
        raise FileNotFoundError(f"Could not load gesture image: {image_path}")
    gesture_image = cv2.resize(gesture_image, (OVERLAY_WIDTH, OVERLAY_HEIGHT))

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

    previous_centers = {}
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera.")

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                height, width = WINDOW_HEIGHT, WINDOW_WIDTH
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_centers = {}
                show_gesture_image = False

                if results.multi_hand_landmarks:
                    handedness_list = results.multi_handedness or []
                    for index, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp.solutions.hands.HAND_CONNECTIONS,
                        )

                        label = f"Hand {index + 1}"
                        if index < len(handedness_list):
                            label = handedness_list[index].classification[0].label

                        if label == "Right" and is_two_finger_gesture(
                            hand_landmarks
                        ):
                            show_gesture_image = True

                        center_x = sum(
                            landmark.x for landmark in hand_landmarks.landmark
                        ) / len(hand_landmarks.landmark)
                        center_y = sum(
                            landmark.y for landmark in hand_landmarks.landmark
                        ) / len(hand_landmarks.landmark)
                        center = (center_x * width, center_y * height)
                        current_centers[label] = center

                        previous = previous_centers.get(label)
                        movement = 0.0
                        if previous is not None:
                            dx = center[0] - previous[0]
                            dy = center[1] - previous[1]
                            movement = (dx * dx + dy * dy) ** 0.5

                        status = (
                            "Moving"
                            if previous is not None and movement >= args.threshold
                            else "Still"
                        )
                        text = f"{label}: {status} ({movement:.1f} px/frame)"
                        cv2.putText(
                            frame,
                            text,
                            (10, 35 + index * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            frame,
                            (int(center[0]), int(center[1])),
                            6,
                            (0, 0, 255),
                            -1,
                        )

                previous_centers = current_centers
                if show_gesture_image:
                    frame[:OVERLAY_HEIGHT, :OVERLAY_WIDTH] = gesture_image

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
