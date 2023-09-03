import cv2
import mediapipe as mp
import csv

# settings
input_video_name = '../samplevideo.mp4'
output_video_name = 'facemesh.avi'
output_landmark_file_name = 'face_landmarks.csv'

if __name__ == '__main__':
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(input_video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    with open(output_landmark_file_name, mode='w') as csv_file:
        fieldnames = ['frame_num', 'landmark_x', 'landmark_y']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        frame_num = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for id, lm in enumerate(face_landmarks.landmark):
                        writer.writerow({'frame_num': frame_num, 'landmark_x': lm.x, 'landmark_y': lm.y})

                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            result.write(image)
            frame_num += 1

    cap.release()
    result.release()
