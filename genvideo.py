# 第四步 由帧生成视频
import cv2
import os

frame_dir = "output_57_efraft"
frames = sorted(os.listdir(frame_dir))  # 确保顺序！

# 读取第一帧，确定分辨率
first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w, _ = first_frame.shape

out = cv2.VideoWriter(
    "output2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    24,   # fps
    (w, h)
)

for f in frames:
    img = cv2.imread(os.path.join(frame_dir, f))
    out.write(img)

out.release()
