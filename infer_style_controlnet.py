import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

import os

import cv2
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

output_dir = "examples/stylized_frames/lynx"
os.makedirs(output_dir, exist_ok=True)

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)


# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

# style image
image = "./styles/1.jpg"
image = Image.open(image)
image.resize((512, 512))

# control image
input_image = cv2.imread("./examples/origin_frames/lynx/lynx00058.jpg")
h, w = input_image.shape[:2] # should be the mutiple of 8, otherwise sdxl will change it
print("width =", w, "height =", h)

# canny image
detected_map = cv2.Canny(input_image, 50, 100) # change two threshold to ensure canny map may catch the key structure
canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
canny_map.save("./examples/canny.png")

control_image = canny_map
# generate image
images = ip_model.generate(pil_image=image,
                           prompt="a lynx watching the far in ukiyoe style, best quality, high quality",
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           neg_content_scale=0.5,
                           scale=1,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=20, 
                           seed=42,
                           image=control_image,
                           controlnet_conditioning_scale=0.8,
                           height=h,
                           width=w,
                          )

images[0].save(output_dir + "/lynx058.jpg")