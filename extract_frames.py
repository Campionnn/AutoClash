import cv2
import os

def video_to_frames(video_path, output_folder, start_time_sec=0):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps
    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    print(f"Video duration: {duration_sec:.2f} seconds")

    # Set the start time in milliseconds
    start_frame = int(start_time_sec * fps)
    if start_time_sec > duration_sec:
        print("Error: Start time exceeds video duration.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Starting from frame: {start_frame}")

    frame_number = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:  # Break if there are no frames left
            break

        # Save frame as an image
        frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")

        frame_number += 1

    # Release video capture
    cap.release()
    print("Done! All frames from the specified time have been extracted.")

# Example usage
video_path = "clash01.mp4"  # Replace with your video path
output_folder = "clash01"    # Folder to save frames
start_time_sec = 45          # Start extracting frames from 10 seconds
video_to_frames(video_path, output_folder, start_time_sec)