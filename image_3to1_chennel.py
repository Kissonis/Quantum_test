from PIL import Image
import os
path_in = 'data/img_train/masks_3chennel/'
path_out = 'data/img_train/masks/'
os.makedirs(path_out, exist_ok=True)
masks_list = os.listdir(path_in)
thresh = 200
fn = lambda x : 255 if x > thresh else 0

for img_mask in masks_list:
    img_current = Image.open(path_in + img_mask)
    img_1chemmel = img_current.convert('L').point(fn, mode='1')
    img_1chemmel.save(path_out + img_mask)
