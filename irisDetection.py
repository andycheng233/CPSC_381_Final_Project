import cv2
import mediapipe as mp
import numpy as np
import math

TOTAL_BLINKS = 0
COUNTER = 0
BLINK_PROCESSED = False

FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=0.6, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)

LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
MOUTH = [13, 14, 15, 17, 0, 11, 12, 61, 291]
# UPPER LIP = [13, 14, 15, 17], LOWER LIP = [0 11, 12], LEFT/RIGHT CORNERS = [61, 291]
FACE_CENTER = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
# 10, 152 --> top and bottom
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def landmarksDetection(image, results, draw=False):
    image_height, image_width= image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
    return mesh_coordinates


def blinkRatio(landmarks, right_indices, left_indices):

    right_eye_landmark1 = landmarks[right_indices[0]]
    right_eye_landmark2 = landmarks[right_indices[8]]

    right_eye_landmark3 = landmarks[right_indices[12]]
    right_eye_landmark4 = landmarks[right_indices[4]]

    left_eye_landmark1 = landmarks[left_indices[0]]
    left_eye_landmark2 = landmarks[left_indices[8]]

    left_eye_landmark3 = landmarks[left_indices[12]]
    left_eye_landmark4 = landmarks[left_indices[4]]

    right_eye_horizontal_distance = euclaideanDistance(right_eye_landmark1, right_eye_landmark2)
    right_eye_vertical_distance = euclaideanDistance(right_eye_landmark3, right_eye_landmark4)

    left_eye_vertical_distance = euclaideanDistance(left_eye_landmark3, left_eye_landmark4)
    left_eye_horizobtal_distance = euclaideanDistance(left_eye_landmark1, left_eye_landmark2)

    right_eye_ratio = right_eye_horizontal_distance/right_eye_vertical_distance
    left_eye_ratio = left_eye_horizobtal_distance/left_eye_vertical_distance

    eyes_ratio = (right_eye_ratio+left_eye_ratio)/2

    return eyes_ratio

def mouthRatio(landmarks, mouth_indices):
    mouth_top = landmarks[mouth_indices[0]]
    mouth_bottom = landmarks[mouth_indices[1]]
    mouth_left = landmarks[mouth_indices[7]]
    mouth_right = landmarks[mouth_indices[8]]

    mouth_vertical_ratio = euclaideanDistance(mouth_top, mouth_bottom)
    mouth_horizontal_ratio = euclaideanDistance(mouth_left, mouth_right)

    return [int(mouth_horizontal_ratio), int(mouth_vertical_ratio)]

def faceCenter(landmarks, face_indices):
    x = [landmarks[i][0] for i in face_indices]
    y = [landmarks[i][1] for i in face_indices]

    return [int(sum(x)/len(x)), int(sum(y)/len(y))]

def faceAngle(landmarks, face_indices):
    dx = landmarks[face_indices[0]][0] - landmarks[face_indices[len(face_indices)-1]][0]
    dy = landmarks[face_indices[0]][1] - landmarks[face_indices[len(face_indices)-1]][1]
    angle_rad = math.atan2(dy,dx)
    return abs(int(math.degrees(angle_rad)))

def varianceCalculate(old, new):
    if old is None or new is None:
        return 0
    
    delta = 0
    
    if isinstance(old, list):
        if len(old) != len(new):
            print("Length of the two inputs do not match!")
            return 0
        
        for i in range(len(old)):
            delta += math.sqrt((new[i]-old[i])**2)
    
    if isinstance(old, int):
        delta += math.sqrt((new - old)**2)

    return delta

mouth_ratio_old = None
face_center_old = None
face_angle_old = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_coordinates = landmarksDetection(frame, results, True)

        eyes_ratio = blinkRatio(mesh_coordinates, RIGHT_EYE, LEFT_EYE)

        mouth_ratio = mouthRatio(mesh_coordinates, MOUTH)
        mouth_variance = varianceCalculate(mouth_ratio_old, mouth_ratio)
        if mouth_ratio is not None:
            mouth_ratio_old = mouth_ratio.copy()

        face_center = faceCenter(mesh_coordinates, FACE_CENTER)
        face_center_variance = varianceCalculate(face_center_old, face_center)
        if face_center is not None:
            face_center_old = face_center.copy()

        face_angle = faceAngle(mesh_coordinates, FACE_CENTER)
        face_angle_variance = varianceCalculate(face_angle_old, face_angle)
        if face_angle is not None:
            face_angle_old = face_angle

        h, w, _ = frame.shape

        if eyes_ratio > 4.5:
            if BLINK_PROCESSED == False:
                TOTAL_BLINKS += 1
            BLINK_PROCESSED = True

        else:
            BLINK_PROCESSED = False
            if COUNTER > 4:
                TOTAL_BLINKS += 1
                COUNTER = 0

        cv2.rectangle(frame, (20, 100), (500, 420), (0,0,0), -1)
        cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}',(30, 150), FONT, 1, (0, 255, 0), 3)
        cv2.putText(frame, f'Mouth Ratio: {mouth_ratio[0]}x, {mouth_ratio[1]}y', (30,200), FONT, 1, (0,255,0), 3)   
        cv2.putText(frame, f'Mouth Variance: {mouth_variance}', (30,250), FONT, 1, (0,255,0), 3)   
        cv2.putText(frame, f'Face Center: {face_center[0]}x, {face_center[1]}y', (30,300), FONT, 1, (0,255,0), 3) 
        cv2.putText(frame, f'Face Center Variance: {face_center_variance}', (30,350), FONT, 1, (0,255,0), 3)   
        cv2.putText(frame, f'Face Angle Variance: {face_angle_variance}', (30,400), FONT, 1, (0,255,0), 3)   


        for idx in MOUTH:
            x = int(results.multi_face_landmarks[0].landmark[idx].x * w)
            y = int(results.multi_face_landmarks[0].landmark[idx].y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots on mouth
        
        for idx in FACE_CENTER:
            x = int(results.multi_face_landmarks[0].landmark[idx].x * w)
            y = int(results.multi_face_landmarks[0].landmark[idx].y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)  # Yellow dots on mouth

    cv2.imshow("Iris Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()