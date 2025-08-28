import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os

print("Initializing...")

# --- Helper Function to Calculate Eye Aspect Ratio (EAR) ---
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the EAR
    ear = (A + B) / (2.0 * C)
    return ear

# --- Initialize Dlib's Face Detector and Landmark Predictor ---
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Check if the model file exists
if not os.path.exists(predictor_path):
    print(f"Error: Model file not found at '{predictor_path}'")
    print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# --- Define Constants for Blink Detection ---
EYE_AR_THRESH = 0.22  # Threshold to determine if an eye is closed
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames the eye must be closed for a "blink"

# --- State Variables for the Liveness Challenge ---
challenge_passed = False
blinks_required = 2
blink_counter = 0
eye_closed_for_frames = 0

# Get the indices of the facial landmarks for the left and right eye
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # --- LIVENESS CHALLENGE LOGIC ---
    if not challenge_passed:
        instruction_text = f"Blink {blinks_required} times ({blink_counter}/{blinks_required})"
        cv2.putText(frame, instruction_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        # Only process if at least one face is detected
        if rects:
            # Assume the first detected face is the user
            shape = predictor(gray, rects[0])
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the EAR for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Check if the EAR is below the blink threshold
            if ear < EYE_AR_THRESH:
                eye_closed_for_frames += 1
            else:
                # If eyes were closed for a sufficient number of frames, count it as a blink
                if eye_closed_for_frames >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                # Reset the counter
                eye_closed_for_frames = 0
        
        # Check if the user has completed the challenge
        if blink_counter >= blinks_required:
            challenge_passed = True

    # --- POST-CHALLENGE DISPLAY ---
    else:
        success_text = "Liveness Check Passed!"
        cv2.putText(frame, success_text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw a box around the detected face(s) after passing
        for rect in rects:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Active Liveness Test - Press 'q' to Quit", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()