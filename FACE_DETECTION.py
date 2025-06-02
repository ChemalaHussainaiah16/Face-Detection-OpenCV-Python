import cv2

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a video capture object
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide a file path for a video

# List of names of people to be tracked
names = ["Person 1", "Person 2", "Person 3"]  # Add more names as needed

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame from video source.")
        break  # Exit the loop if frame reading fails
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each detected face and display the name
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if i < len(names):
            cv2.putText(frame, names[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()