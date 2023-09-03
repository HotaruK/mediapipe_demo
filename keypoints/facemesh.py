import cv2
import mediapipe as mp

if __name__ == '__main__':
    image_name = 'ilya-mondryk-pH8bJytQMZc-unsplash.jpg'
    output_name = 'face_mesh_no.jpg'

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True
    )
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_name)
    height, width, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255)),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0))
            )
            for i, landmark in enumerate(face_landmarks.landmark):
                x, y = int(landmark.x * width), int(landmark.y * height)
                cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imwrite(output_name, image)
