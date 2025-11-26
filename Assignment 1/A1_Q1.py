import cv2
import numpy as np

# Create a blank black image
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw Blue Ellipse
cv2.ellipse(img, (250, 150), (100, 50), 0, 0, 360, (255, 0, 0), -1)

# Draw Green Ellipse
cv2.ellipse(img, (170, 300), (100, 50), 0, 0, 360, (0, 255, 0), -1)

# Draw Red Ellipse
cv2.ellipse(img, (330, 300), (100, 50), 0, 0, 360, (0, 0, 255), -1)

# Add the text "OpenCV"
cv2.putText(img, "OpenCV", (160, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

# Show and save the image
cv2.imshow("OpenCV Logo", img)
cv2.imwrite("OpenCV_logo.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


