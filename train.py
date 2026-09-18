import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import time
rest = ''

while rest !='n':

    # Set the label you want to collect (e.g., 'A', 'B', etc.)
    label = input("Enter Data Name:")
    gesture_label = label

    # Set up webcam
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    # Create folder if it doesn't exist
    data_folder = f'Data/{gesture_label}'
    os.makedirs(data_folder, exist_ok=True)

    print(f"Capturing data for gesture: {gesture_label}")
    print("Press 's' to start saving frames. Press 'q' to start next.")

    img_count = 0
    start_capture = False

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Add padding to the bounding box
            offset = 20
            x1, y1 = max(0, x - offset), max(0, y - offset)
            x2, y2 = x + w + offset, y + h + offset
            hand_crop = img[y1:y2, x1:x2]

            # Save frame if start_capture is True
            if start_capture:
                img_count += 1
                cv2.imwrite(f"{data_folder}/img_{img_count}.jpg", hand_crop)
                cv2.putText(img, f"Saved: {img_count}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show cropped hand image on screen
            cv2.imshow("Cropped Hand", hand_crop)

        # Display webcam feed
        cv2.imshow("Data Collection - ASL", img)

        key = cv2.waitKey(1)

        # Start saving frames
        if key == ord('s'):
            print("Started capturing...")
            start_capture = True

        # Stop
        elif key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    rest = input("Proceed to next sign(y/n):")
