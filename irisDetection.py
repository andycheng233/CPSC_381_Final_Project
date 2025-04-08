import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            '''try: 
                mp_drawing.draw_landmarks(
                    image=frame, 
                    landmark_list=face_landmarks, 
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

            except ValueError as e:
                print("Error drawing iris landmarks")
                continue'''
            left_eye_points = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_points = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            #mesh_points=np.array([np.multiply([p.x, p.y], [frame.shape[1], frame.shape[0]]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            #cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            #cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)

            for point in left_eye_points + right_eye_points:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x,y), 1, (255,0,0), -1)

            #left_eye_center = (int(left_eye_points[0].x * frame.shape[1]), int(left_eye_points[0].y * frame.shape[0]))
            #right_eye_center = (int(right_eye_points[0].x * frame.shape[1]), int(right_eye_points[0].y * frame.shape[0]))

            #cv2.circle(frame, left_eye_center, 5, (0,255,0), -1)
            #cv2.circle(frame, right_eye_center, 5, (0,255,0), -1)


    cv2.imshow("Iris Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()