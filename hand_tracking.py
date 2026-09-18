import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hand and extract landmarks
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmarks
        bbox = hand["bbox"]      # Bounding box info x,y,w,h
        center = hand["center"]  # Center of the hand

        # Display center point
        cv2.circle(img, center, 10, (255, 0, 0), cv2.FILLED)

    # Display the webcam feed
    cv2.imshow("ASL Hand Tracker", img)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
