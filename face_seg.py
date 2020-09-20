import os

from skimage.transform import resize

import sys
import torch
import torch.nn.functional as F
import matplotlib.patches as mpatches
import imageio
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/content/motion-co-seg/')
sys.path.append('/content/DeepFaceLab/')
sys.path.append('/content/motion-co-seg/face_parsing/cp/')

from part_swap import load_face_parser

from PIL import Image
import cv2
from DFLIMG import DFLIMG
from pathlib import Path
import sys
import settings


def visualize_segmentation(image, network, supervised=True, hard=True, colormap='gist_rainbow'):
    with torch.no_grad():
        inp = torch.tensor(image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        if supervised:
            inp = F.interpolate(inp, size=(512, 512))
            inp = (inp - network.mean) / network.std
            mask = torch.softmax(network(inp)[0], dim=1)
            mask = F.interpolate(mask, size=image.shape[:2])
        else:
            mask = network(inp)['segmentation']
            mask = F.interpolate(mask, size=image.shape[:2], mode='bilinear')
    
    if hard:
        mask = (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()
    
    colormap = plt.get_cmap(colormap)
    num_segments = mask.shape[1]
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    color_mask = 0
    patches = []
    for i in range(num_segments):
        if i != 0:
            color = np.array(colormap((i - 1) / (num_segments - 1)))[:3]
        else:
            color = np.array((0, 0, 0))
        patches.append(mpatches.Patch(color=color, label=str(i)))
        color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)


    full_mask = mask

    only_face_area = 1 - mask[:,:,0] - mask[:,:,16] - mask[:,:,17]- mask[:,:,14] # excluding neck , dress, and hair
    full_head_area = 1 - mask[:,:,0] - mask[:,:,16] - mask[:,:,14] # face + hair
    
    return 0.3 * image + 0.7 * color_mask, color_mask, full_mask, only_face_area, full_head_area
    
    

face_parser = load_face_parser(cpu=False)

for face_frames_path in ["workspace/data_src/aligned/", "workspace/data_dst/aligned/"]:

    extracted_face_paths = [os.path.join(face_frames_path, i) for i in os.listdir(face_frames_path)]

    n_total_files = len(extracted_face_paths)

    mode = settings.Face_Type

    for img_path in extracted_face_paths: 

      img = resize(imageio.imread(img_path), (256, 256))[..., :3]

      vis_mask, color_mask, full_mask, only_face_area, full_head_area = visualize_segmentation(img, face_parser)


      dflimg = DFLIMG.load(Path(img_path))

      if mode == 'head':

        dflimg.set_xseg_mask(full_head_area)

      elif mode == 'wf':

        dflimg.set_xseg_mask(only_face_area)
        
      else:
      
        pass
      
      dflimg.save()
