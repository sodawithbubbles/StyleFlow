# Generate the video by the stylized frames
import cv2
import os
# frame_dir = "videos/allframes_wave"
frame_dir = "examples/stylized_frames/output_lynx"
frames = sorted(os.listdir(frame_dir)) # sorted frames

# resolution
first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
h, w, _ = first_frame.shape

out = cv2.VideoWriter(
    "examples/stylized_videos/lynx1.mp4", # video name
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,   # fps
    (w, h)
)

for f in frames:
    img = cv2.imread(os.path.join(frame_dir, f))
    out.write(img)

out.release()

print("Done!")
