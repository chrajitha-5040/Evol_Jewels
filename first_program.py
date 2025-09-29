import cv2

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the input image
image = cv2.imread("lady.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    # Draw rectangle around face
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Approximate neck region (just below the face)
    neck_y_start = y + h
    neck_height = int(h * 0.6)   # you can tune this factor
    neck_y_end = neck_y_start + neck_height

    # Define neck bounding box
    neck_x_start = x + int(w * 0.25)   # narrower than face
    neck_x_end = x + int(w * 0.75)

    cv2.rectangle(image, (neck_x_start, neck_y_start), (neck_x_end, neck_y_end), (0, 255, 0), 2)

# Show result
cv2.imshow("Neck Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
