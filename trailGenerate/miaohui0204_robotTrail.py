#-*- coding: <encoding name> -*-
# -*- coding: utf-8 -*-
 
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
 

from draw_tool import Draw_tool_dlntest
 

class MiaoHui1116:
    def __init__(self,):
        pass
        
        
    def trail_generate(self, stick_img_path, parsing_path, trail_save_path, trial_img_path, scancodeid, writeName, max_thresh=130):
        version = '轨迹算法'
        robot_circle_path = os.path.join('./robot_circle.png')
        
        print(version+"_"+scancodeid)
       
 
        Draw_Tool = Draw_tool_dlntest(stick_img_path, parsing_path, trail_save_path,  max_thresh=200)
        
        
        finalpath = Draw_Tool.main(trial_img_path,scancodeid, writeName, robot_circle_path)
       
        return finalpath, version
      
 


if __name__ == '__main__':
    main()













