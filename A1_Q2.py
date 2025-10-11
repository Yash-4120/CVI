import cv2
import numpy as np
import time

def apply_invisible_cloak(frame, background):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for green cloak
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask1 = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological transformation to remove noise
    # cv2.morphologyEx helps refine the mask by removing small white dots or holes
    kernel = np.ones((3, 3), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, kernel, iterations=1)

    # Inverse mask to get parts not covered by the cloak
    mask2 = cv2.bitwise_not(mask1)

    # Segment out the cloak from the frame and background
    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(frame, frame, mask=mask2)

    # Combine background + current frame
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    return final_output


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    print("Capturing background... stay still for 2 seconds.")
    for i in range(60):
        ret, background = cap.read()
    background = np.flip(background, axis=1)

    print("Background captured. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.flip(frame, axis=1)
        output = apply_invisible_cloak(frame, background)

        cv2.imshow("Cloak Effect", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
