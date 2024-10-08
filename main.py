import cv2
import mediapipe as mp
import math
import time
import mouse
import pyautogui

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

prev_thumb_tip = (0, 0)
prev_move_x = 0
prev_move_y = 0
damping_factor = 0.8

SPEED_SCALING_FACTOR = 5

last = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]  
        lmList = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in handLms.landmark]

        thumb_tip = lmList[4]
        middle_tip = lmList[12]  
        index_tip = lmList[8]
        distance_thumb_middle = math.sqrt((thumb_tip[0] - middle_tip[0]) ** 2 + (thumb_tip[1] - middle_tip[1]) ** 2)
        distance_index_thumb = math.sqrt((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2)

        if distance_index_thumb < 30:  
            move_x = thumb_tip[0] - prev_thumb_tip[0]
            move_y = thumb_tip[1] - prev_thumb_tip[1]
            move_x *= -SPEED_SCALING_FACTOR
            move_y *= SPEED_SCALING_FACTOR
            move_x = (prev_move_x * damping_factor) + (move_x * (1 - damping_factor))
            move_y = (prev_move_y * damping_factor) + (move_y * (1 - damping_factor))
            mouse.move(move_x, move_y, absolute=False)
            prev_move_x = move_x
            prev_move_y = move_y
        else:
            if distance_thumb_middle < 30 and time.time() - last > 0.2:  
                last = time.time()
                pyautogui.click()

        prev_thumb_tip = thumb_tip

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
