import glob 
import numpy as np
from shutil import copyfile
import os

if os.path.isdir('/content/workspace/preview'): os.system('rm -r /content/workspace/preview')

import os

if not os.path.isdir('/content/workspace/preview'): os.mkdir('/content/workspace/preview')
if not os.path.isdir('/content/workspace/preview/aligned'): os.mkdir('/content/workspace/preview/aligned')
if not os.path.isdir('/content/workspace/preview/merged'): os.mkdir('/content/workspace/preview/merged')
if not os.path.isdir('/content/workspace/preview/merged_mask'): os.mkdir('/content/workspace/preview/merged_mask')

f = glob.glob('/content/workspace/data_dst/aligned/*')
if len(f)>50:
  h = np.arange(0,50,2)
  f = np.array(sorted(f)[:50])[h]

else:
  h = np.arange(0,len(f),2)
  f = np.array(sorted(f)[:len(f)])[h]

for i in f:
  copyfile(i, os.path.join('/content/workspace/preview/aligned/', i.split('/')[-1]))
f = glob.glob('/content/workspace/data_dst/*g')

if len(f)>50:
  h = np.arange(0,50,2)
  f = np.array(sorted(f)[:50])[h]

else:
  h = np.arange(0,len(f),2)
  f = np.array(sorted(f)[:len(f)])[h]

for i in f:
  copyfile(i, os.path.join('/content/workspace/preview/', i.split('/')[-1]))