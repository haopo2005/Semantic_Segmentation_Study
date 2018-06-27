#-*- coding: UTF-8 -*- 
import os
import sys
import json
import numpy as np
import pprint

def jst_parse(json_path):
    annotations = json.load(open(json_path))
    height =  annotations['imgHeight']
    width =  annotations['imgWidth']
    instances = {}
    objects = annotations['objects']
    count = 0
    for a in objects:
        label = a['label']
        polygon = a['polygon']
        all_x_points = []
        all_y_points = []
        for x,y in polygon:
            if x >= width:
                x = width - 1
            if y >= height:
                y = height - 1
            all_x_points.append(x)
            all_y_points.append(y)
        instance = {'label':label,'shape_attributes':{'all_points_x':all_x_points,'all_points_y':all_y_points}}
        instances[count] = instance
        count = count + 1
    return height,width,instances

if __name__ == "__main__":
    jsondir = sys.argv[1]
    pngdir = sys.argv[2]
    os.system('cd '+jsondir)
    list = os.listdir(jsondir) #get the image or json file name
    final_json = {}
    new_json = {}
    f = open('test.json','w+')

    for i in range(0,len(list)):
        json_path = os.path.join(jsondir,list[i])
        temp = json_path.split('.')[0].split('/')
        json_name = temp[len(temp)-1]
        png_name = json_name.split('_')
        #print(png_name)
        png_path = pngdir + '/' + png_name[0]+'_'+png_name[1]+'_'+png_name[2]+'_leftImg8bit' + '.png'
        #print json_path 
        #print(png_path)
        h,w,instances = jst_parse(json_path)
        tjson = {'path':png_path,'w':w,'h':h,'instances':instances}
        #pprint.pprint(tjson)
        new_json[json_name] = tjson
        final_json['cityspace'] = new_json
    #print json.dumps(final_json)
    f.write(json.dumps(final_json))
    f.close()
