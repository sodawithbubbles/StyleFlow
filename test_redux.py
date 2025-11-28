# 第三步 光流生成所有对应帧
import gc
import os
import sys
import time

import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ezsynth.sequences import EasySequence
from ezsynth.aux_classes import RunConfig
from ezsynth.aux_utils import save_seq
from ezsynth.main_ez import Ezsynth

st = time.time()

style_paths = [
    #"D:/Ezsynth/examples/styles/cat00000.jpg",
    "D:/Ezsynth/examples/styles/cat00026.jpg",
    #"D:/Ezsynth/examples/styles/style003.jpg",
    #"D:/Ezsynth/examples/styles/style006.png",
    #"D:/Ezsynth/examples/styles/style010.png",
    #"D:/Ezsynth/examples/styles/style014.png",
    #"D:/Ezsynth/examples/styles/style019.png",
    #"D:/Ezsynth/examples/styles/style099.jpg",
]

image_folder = "D:/Ezsynth/videos/allframes_cat"
mask_folder = "D:/Ezsynth/examples/mask/mask_feather"
output_folder = "D:/Ezsynth/output"

# edge_method="Classic"
edge_method = "PAGE"
# edge_method="PST"

# flow_arch = "RAFT"
# flow_model = "sintel"

flow_arch = "EF_RAFT"
flow_model = "25000_ours-sintel"

# flow_arch = "FLOW_DIFF"
# flow_model = "FlowDiffuser-things"


ezrunner = Ezsynth(
    style_paths=style_paths,
    image_folder=image_folder,
    cfg=RunConfig(pre_mask=False, feather=5),
    edge_method=edge_method,
    raft_flow_model_name=flow_model,
    mask_folder=mask_folder,
    # do_mask=True,
    do_mask=False,
    flow_arch=flow_arch
)


# only_mode = EasySequence.MODE_FWD
# only_mode = EasySequence.MODE_REV
only_mode = None

stylized_frames, err_frames, flow_frames = ezrunner.run_sequences_full(
    only_mode, return_flow=True
)
# stylized_frames, err_frames = ezrunner.run_sequences(only_mode)

save_seq(stylized_frames, "D:/Ezsynth/output_57_efraft_cat")
save_seq(flow_frames, "D:/Ezsynth/flow_output_57_efraft_cat")
# save_seq(err_frames, "D:/Ezsynth/output_51err")

gc.collect()
torch.cuda.empty_cache()

print(f"Time taken: {time.time() - st:.4f} s")
