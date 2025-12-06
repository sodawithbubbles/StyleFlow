import os
import cv2
import re
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from lpips import LPIPS  # pip install lpips
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import clip
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# 参数配置
# -------------------------------
ori_dir = "examples/output_frames"      # 原视频帧
sty_dir = "examples/output_57_efraft_lynx"      # 风格化视频帧
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 工具函数
# -------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)

def sorted_frames(folder):
    frames = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".png")]
    frames = sorted(frames, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return [os.path.join(folder, f) for f in frames]

# -------------------------------
# LPIPS (Visual Quality)
# -------------------------------
lpips_model = LPIPS(net='vgg').to(device)
def compute_lpips(img1, img2):
    t1 = image_to_tensor(img1)
    t2 = image_to_tensor(img2)
    return lpips_model(t1, t2).item()

# -------------------------------
# CSD-Score (风格一致性)
# -------------------------------
def compute_csd(img1, img2):
    img1_hsv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2HSV)
    img2_hsv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2HSV)
    hist1 = cv2.calcHist([img1_hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist2 = cv2.calcHist([img2_hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# -------------------------------
# Motion Smooth (光流平滑度)
# -------------------------------
def calc_flow(img1, img2):
    g1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow

def motion_smooth(frames_imgs):
    flows = []
    for i in range(len(frames_imgs)-1):
        f1 = frames_imgs[i]
        f2 = frames_imgs[i+1]
        flow = calc_flow(f1, f2)
        flows.append(flow)
    diffs = []
    for i in range(len(flows)-1):
        diff = np.mean((flows[i+1]-flows[i])**2)
        diffs.append(diff)
    return np.mean(diffs)

# -------------------------------
# CLIP Content Similarity
# -------------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
def compute_clip_content(orig_img, styl_img):
    orig_input = clip_preprocess(orig_img).unsqueeze(0).to(device)
    styl_input = clip_preprocess(styl_img).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_feat = clip_model.encode_image(orig_input)
        styl_feat = clip_model.encode_image(styl_input)
    orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
    styl_feat = styl_feat / styl_feat.norm(dim=-1, keepdim=True)
    return (orig_feat @ styl_feat.T).item()

# -------------------------------
# 主循环
# -------------------------------
ori_frames = sorted_frames(ori_dir)
sty_frames = sorted_frames(sty_dir)

lpips_scores = []
csd_scores = []
clip_scores = []

for i, (ori_path, sty_path) in enumerate(zip(ori_frames, sty_frames)):
    ori_img = load_image(ori_path)
    sty_img = load_image(sty_path)
    
    # Visual Quality
    lpips_scores.append(compute_lpips(ori_img, sty_img))
    
    # 风格一致性
    if i > 0:
        csd_scores.append(compute_csd(prev_sty, sty_img))
    
    # CLIP Content
    clip_scores.append(compute_clip_content(ori_img, sty_img))
    
    prev_sty = sty_img

# Motion Smooth
motion_score = motion_smooth([load_image(f) for f in sty_frames])

print("====== Evaluation Result ======")
print(f"Visual Quality (LPIPS): {np.mean(lpips_scores):.4f}  (lower = better)")
print(f"CSD-Score:             {np.mean(csd_scores):.4f}  (higher = better)")
print(f"Motion Smooth:         {motion_score:.4f}  (lower = smoother)")
print(f"Content Similarity (CLIP): {np.mean(clip_scores):.4f}  (higher = better)")

