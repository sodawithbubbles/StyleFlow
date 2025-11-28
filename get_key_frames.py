# 第一步 生成关键帧
import cv2
import os

video_path = "videos/cat.mp4"
output_dir = "videos/allframes_cat"
os.makedirs(output_dir, exist_ok=True)

interval = 1  # 每 5 帧取一帧（根据视频 FPS 调整）
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % interval == 0:
        filename = os.path.join(output_dir, f"cat{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! 保存了 {saved_count} 帧到 {output_dir}")
