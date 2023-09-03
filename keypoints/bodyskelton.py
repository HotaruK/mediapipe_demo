import cv2
import mediapipe as mp

if __name__ == '__main__':
    image_name = 'karina-carvalho-fKTKVrNqXQQ-unsplash.jpg'
    output_name = 'body_skelton_no.jpg'

    image = cv2.imread(image_name)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    mp_drawing = mp.solutions.drawing_utils
    annotated_image = image.copy()
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
        cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imwrite(output_name, annotated_image)
