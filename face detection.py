import cv2

# Step 2: Load the Pre-trained Haar Cascade Model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Step 4: Capture Video from Webcam (Optional)
cap = cv2.VideoCapture(0)

# Step 5: Process the Video Stream or Load an Image
while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 6: Perform Face Detection
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Step 7: Draw Rectangles around Detected Faces and Display in a Window
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Crop the face region and display it in a separate window
        face_roi = img[y:y + h, x:x + w]
        cv2.imshow('Detected Face', face_roi)

    cv2.imshow('Face Detection', img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()