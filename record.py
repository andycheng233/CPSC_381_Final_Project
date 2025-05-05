import cv2
from datetime import datetime


# Open a video capture (use 0 for webcam or a video file path)
cap = cv2.VideoCapture(0)  # Use 0 for webcam or 'video_file.mp4' for a video file

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get video properties (frame width, frame height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

# Define the codec and create a VideoWriter object to save the video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20230505_142530'
#output_filename = f'truth/output_video_{timestamp}.mp4'
output_filename = f'lie/output_video_{timestamp}.mp4'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()  # Read a frame from the video capture
    if not ret:
        break  # If no frame is returned, exit the loop

    # Optionally, process the frame (e.g., apply filters, transformations)

    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Recording', frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and output objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
