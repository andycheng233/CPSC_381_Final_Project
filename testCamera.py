import cv2
import time

def run_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press enter to quit.")

    try:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to read frame.")
                break

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Program ended by user.")

    except:
        print("Error!")

    finally:
        cap.release()
        cv2.destroyAllWindows()

def find_available_cameras(max_index=10):
    print("Scanning for available cameras...\n")
    available_cameras = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            print(f"Camera found at index {index}")
            available_cameras.append(index)
        else:
            print(f"No camera at index {index}")
        cap.release()

    if not available_cameras:
        print("\nNo cameras found.")
    else:
        print(f"\nAvailable camera indexes: {available_cameras}")
    return available_cameras    

if __name__ == "__main__":
    run_camera()