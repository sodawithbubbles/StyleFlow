# First step: Turn videos into frames
import cv2
import os

video_path = "examples/videos/lynx.mp4"
output_dir = "examples/origin_frames/lynx"
os.makedirs(output_dir, exist_ok=True)

interval = 1  # extract every frame
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = os.path.join(output_dir, f"lynx{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! Save {saved_count} frames to {output_dir}")
