import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose an image to detect faces in
img = cv2.imread("rdj.jpg")

# To capture video from webcam
# webcam = cv2.VideoCapture(0)

### Iterate forever over frames
while True:

    ### Read the current frame
    #succesful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print(face_coordinates)

    # Display the image with the faces
    cv2.imshow("YM Face Detector", img)
    key = cv2.waitKey(1)

    #### Stop if Q key is pressed
    if key == 81 or key == 113:
        break

### Release the VideoCapture object
#webcam.release()

# print("Hello World")
