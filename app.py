import time

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

#from plotting import plot_landmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

PLOT3D = False


def visualize(image, results, selfie=False):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    # Flip the image horizontally for a selfie-view display.
    if selfie:
        image = cv2.flip(image, 1)

    # cv2.namedWindow(
    #    "Pose", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
    # )

    cv2.imshow("Pose", image)
    #if PLOT3D:
    #    plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


def process_image(pose, image, selfie):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    width = 1280
    height = int(image.shape[0] / image.shape[1] * width)
    image = cv2.resize(image, (width, height))
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if not results.pose_landmarks:
        return None, None
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results, image


def info(t0, nframes, results):
    t1 = time.time()
    fps = nframes / (t1 - t0)
    print(f"fps {fps:.0f}")
    # print("Nose coordinates")
    # print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    # print("Nose world landmark:")
    # print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    # plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    return t1


def main():
    plt.ion()
    # video_path = "videos/yoga1.mp4"  # 0 for webcam
    # video_path = "videos/mov1.mp4"  # 0 for webcam
    #video_path = "videos/dance.mkv"  # 0 for webcam
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    selfie = False
    if video_path == 0:
        selfie = True
    t0 = time.time()
    count = 0
    nframes = 15
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
    ) as pose:
        while cap.isOpened():
            if cv2.waitKey(5) & 0xFF == ord("e"):
                print("Exit")
                break

            success, image = cap.read()
            if not success:
                break
            results, image = process_image(pose, image, selfie)
            if results:
                visualize(image, results, selfie)
                if count % nframes == 0:
                    t0 = info(t0, nframes, results)
                count += 1

        cap.release()


if __name__ == "__main__":
    main()
