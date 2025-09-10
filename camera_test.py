# camera_test.py
import cv2

def test_camera(index, backend):
    """Tests a camera with a specific backend."""
    backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
    print(f"--- Testing Camera Index {index} with {backend_name} ---")
    
    # Use the specified backend
    cap = cv2.VideoCapture(index, backend)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {index} and backend {backend_name}.")
        return

    print("Camera opened successfully. A window should appear. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow(f'Camera Test (Index {index}, Backend {backend_name}) - Press Q to quit', frame)

        # Wait for 'q' key to be pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"--- Test for Camera Index {index} with {backend_name} finished ---")


if __name__ == "__main__":
    print("Starting camera diagnostics...")
    print("This script will test different camera backends.")
    print("A window should appear with your webcam feed.")
    print("If a window appears, press 'q' with the window selected to close it and continue the test.\n")

    # Test with DSHOW - often the most compatible on Windows
    test_camera(0, cv2.CAP_DSHOW)
    
    # Test with MSMF - the other common Windows backend
    test_camera(0, cv2.CAP_MSMF)

    # You can add more indices if your camera is not at index 0
    # test_camera(1, cv2.CAP_DSHOW)

    print("Diagnostics finished.")
