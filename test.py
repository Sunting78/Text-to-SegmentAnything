from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np

import time
import matplotlib.pyplot as plt
start = time.time()
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
image = cv2.imread('./notebooks/images/dog.jpg')
print(image.shape)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


model = build_sam(checkpoint="./model/sam_vit_h_4b8939.pth").to('cuda')

predictor = SamPredictor(model)
predictor.set_image(image)
input_point = np.array([[30, 500]])
input_label = np.array([1])
#box = np.array([[500, 20], [600, 20]])
masks, iou_predictions, low_res_masks = predictor.predict(point_coords=input_point,
    point_labels=input_label,
   # box = box,
    multimask_output=True,)

plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
plt.axis('off')
plt.savefig('show.png')


print(masks)

print(iou_predictions)




'''
from segment_anything import build_sam, SamAutomaticMaskGenerator
mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="</path/to/model.pth>"))
masks = mask_generator_generate(<your_image>)
'''
