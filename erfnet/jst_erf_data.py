#-*- coding: UTF-8 -*- 
import os
import sys

if __name__ == "__main__":
    pngdir = sys.argv[1]
    os.system('cd '+pngdir)
    #rgb_img_tail = "_leftImg8bit.png"
    annotation_tail = "_gtFine_labelTrainIds.png"
    
    list = os.listdir(pngdir) #get the image or json file name
    for i in range(0,len(list)):
        img_path = os.path.join(pngdir,list[i])
        new_path = img_path.replace(annotation_tail, ".png")
        os.system('mv '+img_path +' '+ new_path)
        #print(new_path)