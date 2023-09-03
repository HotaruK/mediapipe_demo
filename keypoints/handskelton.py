import cv2
import mediapipe as mp

if __name__ == '__main__':
    image_name = 'pexels-viktoria-slowikowska-5871627.jpg'
    output_name = 'hand_skelton_no.jpg'

    image = cv2.imread(image_name)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    mp_drawing = mp.solutions.drawing_utils
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * annotated_image.shape[1]), int(landmark.y * annotated_image.shape[0])
            cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
            x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * annotated_image.shape[1]), int(
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * annotated_image.shape[0])
            cv2.putText(annotated_image, 'Left hand', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * annotated_image.shape[1]), int(
                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * annotated_image.shape[0])
            cv2.putText(annotated_image, 'Right hand', (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(output_name, annotated_image)
