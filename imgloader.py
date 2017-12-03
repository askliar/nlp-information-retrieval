import h5py
import json
import numpy as np
import os

img_data = './data/img_data'
text_data = './data/text_data'

img_features = np.asarray(h5py.File(os.path.join(img_data, 'IR_image_features.h5'), 'r')['img_features'])

with open(os.path.join(img_data, 'IR_img_features2id.json'), 'r') as f:
     visual_feat_mapping = json.load(f)['IR_imgid2id']