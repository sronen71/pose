import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import holistic, pose

#from plotting import plot_landmarks

PLOT3D = False


def remove_landmarks(results):
    unwanted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22]
    if results:
        if results.pose_landmarks:
            for i in unwanted:
                results.pose_landmarks.landmark[i].visibility = 0


def to_array(results):
    body = [[r.x, r.y, r.z] for r in results.pose_world_landmarks.landmark]
    body_hands = [17, 18, 19, 20, 21, 22]
    body = [body[i] for i in range(len(body)) if i not in body_hands]
    body = np.array(body)
    if results.right_hand_landmarks:
        right_hand = [[r.x, r.y, r.z] for r in results.right_hand_world_landmarks.landmark]
        right_hand = np.array(right_hand)
        right_hand += body[16, :] - right_hand[0, :]
    else:
        right_hand = [body[16]] * 21
    if results.left_hand_landmarks:
        left_hand = [[r.x, r.y, r.z] for r in results.left_hand_world_landmarks.landmark]
        left_hand = np.array(left_hand)
        left_hand += body[15, :] - left_hand[0, :]

    else:
        left_hand = [body[15]] * 21
    return np.concatenate([body, right_hand, left_hand], axis=0)


def visualize(image, results, selfie=False):

    remove_landmarks(results)  # remove unwanted landmarks
    # draw connection of elbows to wrists
    # landmark_list: landmark_pb2.NormalizedLandmarkList,
    if results.pose_landmarks and results.left_hand_landmarks:
        results.pose_landmarks.landmark[15].CopyFrom(results.left_hand_landmarks.landmark[0])
    if results.pose_landmarks and results.right_hand_landmarks:
        results.pose_landmarks.landmark[16].CopyFrom(results.right_hand_landmarks.landmark[0])

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
    )
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            holistic.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style(),
        )
    # Flip the image horizontally for a selfie-view display.
    if selfie:
        image = cv2.flip(image, 1)

    # cv2.namedWindow(
    #    "Pose", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL
    # )

    cv2.imshow("Pose", image)
    #if PLOT3D:
    #    plot_landmarks(results.pose_world_landmarks, pose.POSE_CONNECTIONS)


def process_image(holistic, image, selfie):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    width = 1280
    height = int(image.shape[0] / image.shape[1] * width)
    image = cv2.resize(image, (width, height))
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    if not results.pose_landmarks:
        return None, None
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results, image


def info(t0, nframes, results):
    t1 = time.time()
    fps = nframes / (t1 - t0)
    print(f"fps {fps:.1f}")
    # print("Nose coordinates")
    # print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    # print("Nose world landmark:")
    # print(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
    # plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    return t1


def get_joints():

    body_pose_names = [el.name for el in holistic.PoseLandmark]
    hand_pose_names = [el.name for el in holistic.HandLandmark]
    left_hand_names = ["left_" + name for name in hand_pose_names]
    right_hand_names = ["right_" + name for name in hand_pose_names]
    body_hands = [17, 18, 19, 20, 21, 22]
    body_pose_names = [
        body_pose_names[i] for i in range(len(body_pose_names)) if i not in body_hands
    ]
    joint_names = body_pose_names + right_hand_names + left_hand_names
    hand_edges = list(holistic.HAND_CONNECTIONS)
    body_edges = list(holistic.POSE_CONNECTIONS)
    body_edges = [list(edge) for edge in body_edges]

    body_edges = [
        edge for edge in body_edges if edge[0] not in body_hands and edge[1] not in body_hands
    ]
    for edge in body_edges:
        for i in [0, 1]:
            if edge[i] > 16:
                edge[i] -= 6

    nbody = len(body_pose_names)
    nhand = len(hand_pose_names)
    right_hand_edges = [[h[0] + nbody, h[1] + nbody] for h in hand_edges]
    left_hand_edges = [[h[0] + nbody + nhand, h[1] + nbody + nhand] for h in hand_edges]

    joint_edges = np.array(body_edges + right_hand_edges + left_hand_edges)
    return joint_names, joint_edges


def main():
    import poseviz

    joint_names, joint_edges = get_joints()
    viz = poseviz.PoseViz(joint_names, joint_edges)
    plt.ion()
    # video_path = "videos/yoga1.mp4"  # 0 for webcam
    # video_path = "videos/mov1.mp4"  # 0 for webcam
    # video_path = "videos/dance.mkv"  # 0 for webcam
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    selfie = False
    if video_path == 0:
        selfie = True
    t0 = time.time()
    count = 0
    nframes = 60
    with holistic.Holistic(
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

                width = image.shape[1]
                r2 = to_array(results) * width
                r2[:, 2] = r2[:, 2] + 5000
                viz.update(
                    frame=image,
                    boxes=[],
                    poses=[r2],
                    camera=poseviz.Camera.from_fov(55, image.shape[:2]),
                )
                if count % nframes == 0:
                    t0 = info(t0, nframes, results)
                count += 1

        cap.release()


if __name__ == "__main__":
    main()
