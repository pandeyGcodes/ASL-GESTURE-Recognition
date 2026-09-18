import cv2
import joblib
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os

# Load trained model
model = joblib.load(os.path.join(os.path.dirname(__file__), "asl_knn_model.pkl"))

# Webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        offset = 20
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = x + w + offset, y + h + offset

        hand_img = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        flat = resized.flatten().reshape(1, -1)

        prediction = model.predict(flat)[0]
        cv2.putText(img, f"Detected: {prediction}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
