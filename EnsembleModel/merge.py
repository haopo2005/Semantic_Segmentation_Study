#-*- coding: UTF-8 -*- 
import os
import sys
import cv2
import numpy as np


if __name__ == "__main__":
    tuseng_path = sys.argv[1]
    bonnet_path = sys.argv[2]
    maskrcnn_path = sys.argv[3]
    
    tuseng_obj = open(tuseng_path,'r')
    bonnet_obj = open(bonnet_path,'r')
    maskrcnn_obj = open(maskrcnn_path,'r')
    
    tuseng_content = tuseng_obj.read()
    bonnet_content = bonnet_obj.read()
    maskrcnn_content = maskrcnn_obj.read()
    
    tuseng_imglists = tuseng_content.strip().split('\n')
    bonnet_imglists = bonnet_content.strip().split('\n')
    maskrcnn_imglists = maskrcnn_content.strip().split('\n')
    
    for idx in range(0,len(tuseng_imglists)):
        img_tuseng = cv2.imread(tuseng_imglists[idx], cv2.IMREAD_GRAYSCALE)
        img_bonnet = cv2.imread(bonnet_imglists[idx], cv2.IMREAD_GRAYSCALE)
        img_maskrcnn = cv2.imread(maskrcnn_imglists[idx], cv2.IMREAD_GRAYSCALE)
        
        img_h = img_tuseng.shape[0]
        img_w = img_tuseng.shape[1]
        final_mask = np.zeros([img_h,img_w], dtype=np.uint8)
        temp = tuseng_imglists[idx].split('.')[0].split('/')
        final_name = '/home/jst/share/test/'+ temp[len(temp)-1] + '.png'
        for h in range(0,img_h):
            for w in range(0,img_w):
                if img_tuseng[h,w] != img_bonnet[h,w] or img_tuseng[h,w] != img_maskrcnn[h,w]:
                    if img_bonnet[h,w] == img_maskrcnn[h,w]:
                        final_mask[h,w] = img_bonnet[h,w]
                    else:
                        final_mask[h,w] = img_tuseng[h,w]
                else:
                        final_mask[h,w] = img_tuseng[h,w]
        cv2.imwrite(final_name, final_mask)