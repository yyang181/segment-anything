from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from torchvision.models import resnet50
from thop import profile

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


sam_checkpoint = "/home/yyang181/github/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)
# mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint="checkpoints/sam_vit_h_4b8939.pth"))

image = cv2.imread('/home/yyang181/github/segment-anything/1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




# total = sum([param.nelement() for param in sam.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# input = torch.randn(1, 3, 224, 224)

# import pickle
# # Open the file in binary mode
# with open('file.pkl', 'rb') as file:
#     # Call load method to deserialze
#     encoder_input, decoder_input = pickle.load(file)

# flops, params = profile(sam.prompt_encoder, inputs=encoder_input)  # inputs=(input, True)
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

# flops, params = profile(sam.mask_decoder, inputs=decoder_input)  # inputs=(input, True)
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

'''
input = torch.randn(1, 3, 224, 224)
point_coords = torch.randn(1, 2, 2)
a = {'image':input}
b = {'original_size':torch.randn(1, 2)}
c = {'point_coords':point_coords}
d = {'point_labels':None}
e = {'boxes':None}
f = {'mask_inputs':None}

dict_input = [a,b,c,d,e,f]
flops, params = profile(sam, inputs=[dict_input, True])  # inputs=(input, True)
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')
assert 1==0
'''

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 