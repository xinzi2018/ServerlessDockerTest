# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:26:09 2020
@author: mhxyy
"""

import numpy as np
import cv2
import time
import skimage.filters.rank as sfr
from skimage.morphology import disk
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from .Trail_Generation import maintrail_generation,detailtrail_generation
from .Zipper4b import Zipper4b
from .Z_scan import Z_scan
from .Optimal_Path import OptimalPath,OptimalPathUtils
from .calligraphy import Calligraphy
from skimage.morphology import skeletonize
from sknw import build_sknw
#0ï¼šèƒŒæ™?1ï¼šè„¸ 2ï¼šçœ‰ 3ï¼šçœ¼ 4ï¼šçœ¼é•?5ï¼šè€?6ï¼?7ï¼šé¼» 8ï¼šå£è…?9ï¼šä¸Šå˜´å”‡


class Draw_tool:
    def __init__(self,stick_img_path, parsing_path, trail_save_path, model_zoo, max_thresh=150):
        print("stick_img_path", stick_img_path,parsing_path)
        self.model_zoo = model_zoo
#        #zhujingjie
        self.scancodeid_shuangren = ["1024"]#"8ha8aztvnq",
        max_thresh = 63 #63
        self.maintrail_generation = maintrail_generation()
        self.detailtrail_generation = detailtrail_generation()
        self.Zipper4b = Zipper4b()
        self.optimalPathUtils = OptimalPathUtils()
        self.optimalPath = OptimalPath(self.optimalPathUtils)
        self.Z_scan = Z_scan()
#        img_temp_pil = Image.open(stick_img_path)
#        img_out_pil = self.model_zoo.generate_simplify(img_temp_pil)
#        img_out_pil.save("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/extra-u2s_out/"+stick_img_path.split("/")[-1])
#        _,self.img = cv2.threshold(cv2.imread("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/extra-u2s_out/"+stick_img_path.split("/")[-1],0),144,255,cv2.THRESH_BINARY)
        
        self.stick_img_path = stick_img_path
        # _,self.img = cv2.threshold(cv2.imread(stick_img_path,0),144,255,cv2.THRESH_BINARY)# ori
        self.img =  cv2.imread(stick_img_path,0)
        
        self.vis_parsing_anno = np.load(parsing_path)
#        print("self.img self.img self.img ", self.img.shape, self.vis_parsing_anno.shape)
        
        
#        temp = self.vis_parsing_anno[:,:,17].astype(np.uint8)
#        cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/etemp/177hair.png", temp*255)
#        cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/etemp/177person.png", self.img)
        
        
#        for i in range(19):
##            bb = cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2GRAY)
#            bb = self.img
#            temp = self.vis_parsing_anno[:,:,i].astype(np.uint8)
#            new = cv2.addWeighted(bb,0.8, temp*255, 0.2,0)
#            cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/etemp/"+str(i)+"hair.png", new)

        ##11
#        yuan_h, yuan_w = self.img.shape[0], self.img.shape[1]
#        index_img_black = np.where(self.img<127)
#        self.img = self.img[min(index_img_black[0]):max(index_img_black[0]), min(index_img_black[1]):max(index_img_black[1])]
#        xian_h, xian_w = self.img.shape[0], self.img.shape[1]
#        pad_top, pad_left = (yuan_h - xian_h)//2, (yuan_w - xian_w)//2#top, bottom, left, right
#        pad_bottom, pad_right = yuan_h - pad_top - xian_h, yuan_w - pad_left - xian_w
#        self.img = cv2.copyMakeBorder(self.img, pad_top + 50, pad_bottom + 50, pad_left + 50, pad_right + 50, cv2.BORDER_CONSTANT, value = [255,255,255])
        ##22

        
        self.img = cv2.resize(self.img,(1550,1550), interpolation=cv2.INTER_NEAREST)
        self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
        
        
#        ##11
#        self.vis_parsing_anno = self.vis_parsing_anno[min(index_img_black[0]):max(index_img_black[0]), min(index_img_black[1]):max(index_img_black[1]),:]
#        self.vis_parsing_anno = cv2.copyMakeBorder(self.vis_parsing_anno, pad_top + 50, pad_bottom + 50, pad_left + 50, pad_right + 50, cv2.BORDER_CONSTANT, value = [0,0,0])
        ##22
        # print(self.vis_parsing_anno.shape)
        # print(np.unique(self.vis_parsing_anno.astype(np.float32)))
        # print('==================')
        

        self.vis_parsing_anno = cv2.resize(self.vis_parsing_anno, (1550,1550))
        self.vis_parsing_anno = cv2.copyMakeBorder(self.vis_parsing_anno, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
        


        # print(self.vis_parsing_anno.shape)
        # print(np.unique(self.vis_parsing_anno))
        
        
        
#        for i in range(19):
#            bb = self.img
#            temp = self.vis_parsing_anno[:,:,i].astype(np.uint8)
#            print(bb.shape,temp.shape)
#            new = cv2.addWeighted(bb,0.8, temp*255, 0.2,0)
#            cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/etemp/"+str(i)+"hair.png", new)
#            bb = sfr.minimum(bb, disk(2))
#            neww = cv2.addWeighted(bb,0.8, temp*255, 0.2,0)
#            cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/etemp/"+str(i)+"hairrr.png", neww)

#        maskimg = cv2.imread("/mnt/hdd3t/home/yuzeyuan/AISketcher/collected_model/111.bmp", 0)
#        index = np.where(maskimg>127)
#        self.img[index[0], index[1]] = 255
        if len(self.img.shape)==3:
            self.img = self.img[:,:,0]
        h,w = self.img.shape[:2]
        self.factor = max_thresh/max((h,w))
        
        #åˆå§‹åŒ–ä¸€äº›åŸºæœ¬å˜é‡?14æ¡¶æˆ–19é€?
        ##14
#        self.eyebrow_combine = [2-1]
#        self.eyeballs_combine = [3-1, 4-1]
#        self.nose_mouse_combine = [7-1, 8-1, 9-1]
#        self.face_combine = [1-1, 5-1, 6-1, 13-1, 14-1]
#        self.neck_dress_combine = [10-1, 11-1, 12-1]
#        self.glasses_combine = [4-1]
        ##19
        self.eyebrow_combine = [2, 3]
        self.eyeballs_combine = [4, 5, 6]
        self.nose_mouse_combine = [10, 11, 12, 13]
        self.hair_combine = [17]
        self.face_combine = [1, 7, 8, 9, 18]
        self.neck_dress_combine = [14, 15, 16]
        self.glasses_combine = [6]
        self.others_combine = [0]
#        print('self.vis_parsing_anno.shape=',self.vis_parsing_anno.shape)
        if self.vis_parsing_anno.shape[-1]==21 or self.vis_parsing_anno.shape[-1]==19:
            # 为了在有眼镜的时候能够将双眼皮以及下眼睑画出来 尝试将眼珠子的范围往外扩
            # way2: 尝试将眼球外部的眼睑以及双眼皮位置抠出来，然后合并到skin层
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            # 1.先将眼睛和眼镜部分合并A
            left_eye_ori = self.vis_parsing_anno[:,:,5].copy()
            right_eye = self.vis_parsing_anno[:,:,4].copy()    
             
            left_eye_dilate = cv2.dilate(self.vis_parsing_anno[:,:,5].copy(), kernel, iterations=3)
            right_eye_dilate = cv2.dilate(self.vis_parsing_anno[:,:,4].copy()   , kernel, iterations=3)
            eye_glass = right_eye_dilate + left_eye_dilate + self.vis_parsing_anno[:,:,6]
            # cv2.imwrite('eye_glass.jpg',eye_glass*255)
            # 2.将A腐蚀往里缩一部分，避免后面对眼睛部分操作的时候对镜框厚度有影响
            
            # eye_glass_erode = cv2.dilate(eye_glass, kernel, iterations=5)
            eye_glass_erode = cv2.erode(eye_glass, kernel, iterations=3)
            
            # cv2.imwrite('eye_glass_erode.jpg',eye_glass_erode*255)
            # print(np.unique(eye_glass_erode))
            # eye_glass_erode = cv2.threshold(eye_glass_erode, 0.5, 1, cv2.THRESH_BINARY)[1]


            # 3.一轮一轮将眼睛parsing往外膨胀

            left_eye = self.vis_parsing_anno[:,:,5].copy()
            # print(np.unique(left_eye))
            # cv2.imwrite('left_eye_before.jpg',self.vis_parsing_anno[:,:,5]*255)
            left_flag = 4
            while left_flag!=0:
                # print(left_flag)
                index = np.where(left_eye==1)
                # print((eye_glass_erode[index[0],index[1]]).sum(),left_eye.sum())
               

                if (eye_glass_erode[index[0],index[1]]).sum() >= left_eye.sum():
                    left_eye = cv2.dilate(left_eye, kernel, iterations=3)
                    left_flag=left_flag-1
                else:
                    left_flag=0
            # cv2.imwrite('left_eye_after.jpg',left_eye*255)



            right_eye = self.vis_parsing_anno[:,:,4].copy()
            # cv2.imwrite('right_eye_before.jpg',self.vis_parsing_anno[:,:,4]*255)
            right_flag = 4
            while right_flag != 0:
                # print(right_flag)
                index = np.where(right_eye==1)
                if (eye_glass_erode[index[0],index[1]]).sum() >= right_eye.sum():
                    right_eye = cv2.dilate(right_eye, kernel, iterations=3)
                    right_flag=right_flag-1
                else:
                    right_flag=0

            # cv2.imwrite('right_eye_after.jpg',right_eye*255)
            # 4.将膨胀得到的左右眼和原始的左右眼parsing求差，然后赋值给皮肤层
            left_skin = cv2.bitwise_xor(left_eye, self.vis_parsing_anno[:,:,5])
            right_skin = cv2.bitwise_xor(right_eye, self.vis_parsing_anno[:,:,4])
            # cv2.imwrite('left_skin.jpg',left_skin*255)
            # cv2.imwrite('right_skin.jpg',right_skin*255)

            self.vis_parsing_anno[:,:,1] = self.vis_parsing_anno[:,:,1] + left_skin + right_skin# 后处理皮肤parsing ,将处理出来的眼皮部分parsing和到皮肤层

            # index = np.where(left_skin==1)
            # self.vis_parsing_anno[:,:,6][index[0],index[1]]=0
            # index = np.where(right_skin==1)
            # self.vis_parsing_anno[:,:,6][index[0],index[1]]=0

            # cv2.imwrite('self.vis_parsing_anno[:,:,1].jpg',self.vis_parsing_anno[:,:,1]*255)

            ####way2结束
        

        #åˆå§‹åŒ–ä¸€ä¸ªæœ€ç»ˆç»“æž?
        final_img = np.zeros(self.img.shape)+255
        self.final_img = final_img.astype(np.uint8)
        
        #åˆå§‹åŒ–ä¸€ä¸ªç»˜ç”»æœ€ç»ˆç»“æž?
        final_draw = np.zeros(self.img.shape)+255
        self.final_draw = final_draw.astype(np.uint8)
        self.trail_save_path11 = trail_save_path
#        print("self.trail_save_path11self.trail_save_path11self.trail_save_path11", self.trail_save_path11)
        self.trail_file = open(self.trail_save_path11,'w')
        self.robot_path = "/home/zhujingjie/projects/sketch_wood_v3/datas/robot/"
        self.hair_save_path = self.trail_save_path11.replace("test.txt", "hair_test.txt")
        self.face_save_path = self.trail_save_path11.replace("test.txt", "face_test.txt")
        self.eyebrow_save_path = self.trail_save_path11.replace("test.txt", "eyebrow_test.txt")
        self.noparsing_save_path = self.trail_save_path11.replace("test.txt", "noparsing_test.txt")
        self.hair_file = open(self.hair_save_path,'w')
        self.face_file = open(self.face_save_path,'w')
        self.eyebrow_file = open(self.eyebrow_save_path,'w')
        self.noparsing_file = open(self.noparsing_save_path,'w')
        
        self.hairoptimflag = True
        self.faceoptimflag = True
        self.noparoptimflag = True
        self.eyebrowoptimflag = True
        
        self.calligraphy = Calligraphy()
        self.textjson = self.calligraphy.words.keys()
    ## 判读是否在字库里
    def RareWordJudgment(self, writeName):
        print("len(writeName),",len(writeName))
        if len(writeName)>4:
            return False
        for i in range(len(writeName)):
            if writeName[i] not in self.textjson:
                print("RareWordJudgmentRareWordJudgmentRareWordJudgment=")
                return False
        return True
            
    def hair_trail_write(self, final_cont, optimize_lists):
        for c in final_cont:
            cur_node = [0,0]
            write_list = []
            for p in c[1:-1]:
                dic_check=np.sqrt(np.sum(np.square(cur_node[0]-round(self.factor*p[0],2))+np.square(cur_node[1]-round(self.factor*p[1],2))))
                if dic_check>0.5:
                   cur_node = [round(self.factor*p[0],2),round(self.factor*p[1],2)]
                   write_list.append(cur_node)
            if len(write_list)>1: ##20220411
                optimize_list = []
                optimize_list.append([round(self.factor*c[0][0],2)])
                optimize_list.append([round(self.factor*c[0][1],2)])
                optimize_list.append([0])
                optimize_list.append([round(self.factor*c[0][0],2)])
                optimize_list.append([round(self.factor*c[0][1],2)])
                optimize_list.append([-33])
                for j in write_list:
                    optimize_list.append([j[0]])
                    optimize_list.append([j[1]])
                    optimize_list.append([0])
                optimize_list.append([round(self.factor*c[-1][0],2)])
                optimize_list.append([round(self.factor*c[-1][1],2)])
                optimize_list.append([33])
                optimize_lists.append(np.array(optimize_list))
        return optimize_lists
        
            
    def return_write_list(self, c):
        cur_node = [0,0]
        write_list = []
        for p in c[1:-1]:
            dic_check=np.sqrt(np.sum(np.square(cur_node[0]-round(self.factor*p[0],2))+np.square(cur_node[1]-round(self.factor*p[1],2))))
            if dic_check>0.5: ##20220411
                cur_node = [round(self.factor*p[0],2),round(self.factor*p[1],2)]
                write_list.append(cur_node)
        return write_list
 
    def trail_write_other(self, final_cont):
        for c in final_cont:
            cur_node = [0,0]
            write_list = []
            for p in c[1:-1]:
                dic_check=np.sqrt(np.sum(np.square(cur_node[0]-round(self.factor*p[0],2))+np.square(cur_node[1]-round(self.factor*p[1],2))))
                if dic_check>0.5:
                    cur_node = [round(self.factor*p[0],2),round(self.factor*p[1],2)]
                    write_list.append(cur_node)
            self.trail_file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'0'+'\n')
            self.trail_file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'-33'+'\n')
            for j in write_list:
                self.trail_file.write(str(j[0])+ ' '+str(j[1])+' '+'0'+'\n')
            self.trail_file.write(str(round(self.factor*c[-1][0],2))+ ' '+str(round(self.factor*c[-1][1],2))+' '+'33'+'\n')

    def trail_write(self, final_cont, file, flag=0):
        for c in final_cont:
            write_list = self.return_write_list(c)
            if len(write_list)>1:
                file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'0'+'\n')
                file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'-33'+'\n')
                for j in write_list:
                    file.write(str(j[0])+ ' '+str(j[1])+' '+'0'+'\n')

                if flag == 1:
                    file.write(str(round(self.factor*c[-1][0],2))+ ' '+str(round(self.factor*c[-1][1],2))+' '+'0'+'\n')
                    file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'0'+'\n')
                    file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'33'+'\n')
                else:
                    file.write(str(round(self.factor*c[-1][0],2))+ ' '+str(round(self.factor*c[-1][1],2))+' '+'33'+'\n')



    def Optimization(self, save_path):
        f2 = open(save_path, "r")
        lines = f2.readlines()
        BeginPoint, EndPoint, points = self.optimalPathUtils.readBeginEndPoint(lines)
        distance_numpy = np.array(self.optimalPathUtils.countPointDistance(BeginPoint,EndPoint))
        visited = np.zeros((distance_numpy.shape[0],1)).astype(np.uint8)
        wayIndexList = self.optimalPath.findMinDis(distance_numpy)
        self.optimalPathUtils.changeTxT(points, wayIndexList, save_path)



    def hair_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        vis_parsing_anno_color = self.getParsingBasedCombine(self.hair_combine, element_kernel=5, iterationsNum=9)
            
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]
      
        self.final_img = cv2.bitwise_and(self.final_img, img0)
        
        
        contours,result = self.maintrail_generation.main_part(img0)
        
        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)

            self.trail_write(final_cont, self.hair_file, flag=0)
        self.hair_file.close()

 
        if self.hairoptimflag:
            self.Optimization(self.hair_save_path)
        
        f2 = open(self.hair_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)

    def notes_EndToEnd(self,graph):

        nodes_list = []
        for (s, e) in graph.edges():
            nodes_list.append([s, e])
        # print('nodes_list',len(nodes_list))


        # # 将每个线上的点保存成轨迹需要的数据格式
        # final_cont = []
        # for (s, e) in graph.edges():
        #     ps = graph[s][e]['pts']
        #     count = []
        #     # if len(ps[:, 1])<=10:continue
        #     for i in range(len(ps[:, 1])):
        #         count.append([ps[:, 1][i], ps[:, 0][i]])
        #     final_cont.append(count)


        #   # 将点和线可视化的代码
        # print(self.img.shape)
        # drawed_line = np.ones(self.img.shape[:2])*255
        # # print(final_cont)
        # for i in range(len(final_cont)):
        #     for j in range(len(final_cont[i])-1):
        #         print(final_cont[i][j])
        #         cv2.line(drawed_line, tuple(final_cont[i][j]), tuple(final_cont[i][j+1]), 0, 1)


        # cv2.imwrite('mouth111.jpg',drawed_line)

        node_res = []
        final_cont = []
        for i in range(len(nodes_list)):
            if len(graph[nodes_list[i][0]][nodes_list[i][1]]['pts'][:,::-1].tolist())<10:
                continue
            if i == 0:
                node_res.append(nodes_list[0])
                final_cont.append(graph[nodes_list[0][0]][nodes_list[0][1]]['pts'][:,::-1].tolist())
            else:
                node = nodes_list[i]
                # 和node_res已有的结点每个都进行对比
                flag2 = 0
                for j in range(len(node_res)):
                    if node_res[j][1] == node[0]:
                        node_res[j][1] = node[1]
                        final_cont[j] = final_cont[j]+graph[node[0]][node[1]]['pts'][:,::-1].tolist()
                        flag2 = 1
                        break # 只要找到能合并的直接退出
                    elif node_res[j][0] == node[0]:# 当前结点的头与node_res[j]点的头部相连
                        node_res[j][0] = node[1]
                        final_cont[j] = graph[node[0]][node[1]]['pts'][:,::-1].tolist()[::-1] + final_cont[j]
                        flag2 = 1
                        break # 只要找到能合并的直接退出
                    elif node_res[j][1] == node[1]:# 当前结点的尾部与node_res[j]点的尾部相连
                        node_res[j][1] = node[0]
                        final_cont[j] = final_cont[j]+graph[node[0]][node[1]]['pts'][:,::-1].tolist()[::-1]  
                        flag2 = 1
                        break # 只要找到能合并的直接退出
                    elif node_res[j][0] == node[1]:# 当前结点的尾部与node_res[j]点的头部相连
                        node_res[j][0] = node[0]
                        final_cont[j] = graph[node[0]][node[1]]['pts'][:,::-1].tolist()  + final_cont[j]
                        flag2 = 1
                        break # 只要找到能合并的直接退出
                    
                        
                if flag2 == 0:
                    node_res.append(node)
                    final_cont.append(graph[node[0]][node[1]]['pts'][:,::-1].tolist())
        # print('final_cont=',len(final_cont))
        return final_cont



    def getParsingBasedCombine(self, parsing_combine, element_kernel=-1, iterationsNum=-1, IsErode=True, IsDilate=False):
        vis_parsing_anno_color = (np.zeros((self.vis_parsing_anno.shape[0], self.vis_parsing_anno.shape[1]))).astype(np.uint8) + 255
        for pi in parsing_combine:
            # index = np.where(self.vis_parsing_anno == pi)
            index = np.where(self.vis_parsing_anno[:, :, pi] > 0)
            vis_parsing_anno_color[index[0], index[1]] = 0
            
            if IsErode == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color, element, iterations=iterationsNum)

            if IsDilate == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.dilate(vis_parsing_anno_color, element, iterations=iterationsNum)

        return vis_parsing_anno_color

        

    def hair_trail_dln(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        
    
        vis_parsing_anno_color = self.getParsingBasedCombine(self.hair_combine, element_kernel=5, iterationsNum=9)

        index = np.where(vis_parsing_anno_color == 0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]  # 此时得到的img0表示的是头发部分的全部线条
        # self.final_img = cv2.bitwise_and(self.final_img, img0)  ## 表示最终已经画了的图

        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true

        cv2.imwrite('hair_trail_dln.jpg', img*255)
        temp = time.time()
        ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
        graph =  build_sknw(ske)
        print('实际上头发轨迹生成的时间==',time.time()-temp)
        final_cont = self.notes_EndToEnd(graph)


        # self.hair_trail_write(final_cont)
        self.trail_write(final_cont, self.hair_file, flag=0)

        self.hair_file.close() 
       
        if self.hairoptimflag:
            self.Optimization(self.hair_save_path)
        # print('-------optim hair =', time.time() - temp)

        temp = time.time()
        f2 = open(self.hair_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
        # print('-------save hair =', time.time() - temp)

    # def hair_trail_dln_temptest(self):
    #     # img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
    #     # vis_parsing_anno_color = self.getParsingBasedCombine(self.hair_combine, element_kernel=5, iterationsNum=9)

    #     # index = np.where(vis_parsing_anno_color == 0)
    #     # img0[index[0], index[1]] = self.img[index[0], index[1]]  # 此时得到的img0表示的是头发部分的全部线条
    #     # # self.final_img = cv2.bitwise_and(self.final_img, img0)  ## 表示最终已经画了的图

    #     # ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     # img[img == 255] = 1
    #     # img = (img == 1)  # 0-1 转换成false-true


    #     # img = img/255
    #     # cv2.imwrite('hair_trail_dln_temptest.png',img*255)
    #     img = cv2.imread('hair_trail_dln_temptest.png')/255
    #     img = (img == 1)
    #     temp = time.time()
    #     ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
    #     graph =  build_sknw(ske)
    #     print('实际上头发轨迹生成的时间==',time.time()-temp)
    #     final_cont = self.notes_EndToEnd(graph)


    #     # self.hair_trail_write(final_cont)
    #     self.trail_write(final_cont, self.hair_file, flag=0)

    #     self.hair_file.close() 
       
    #     if self.hairoptimflag:
    #         self.Optimization(self.hair_save_path)
    #     # print('-------optim hair =', time.time() - temp)

    #     temp = time.time()
    #     f2 = open(self.hair_save_path, "r")
    #     lines = f2.readlines()
    #     for line in lines:
    #         self.trail_file.write(line)
    #     # print('-------save hair =', time.time() - temp)

            
    def noparsing_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        vis_parsing_anno_color = (np.zeros((self.vis_parsing_anno.shape[0], self.vis_parsing_anno.shape[1]))).astype(np.uint8) 

         
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color,element, iterations = 1)
        img0 = self.img
        self.final_img = cv2.bitwise_and(self.final_img,img0)
#        cv2.imwrite("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/extra-u2s_in/test.jpg", self.final_img)
        
        contours,result = self.maintrail_generation.main_part(img0)
        
        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)
            

            # self.noparsing_trail_write(final_cont)
            self.trail_write(final_cont, self.noparsing_file, flag=0)
        self.noparsing_file.close()
        ##
        if self.noparoptimflag:
            self.Optimization(self.noparsing_save_path)
            
        f2 = open(self.noparsing_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
            
    #è„¸éƒ¨
    def face_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255

        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1)

        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]

        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=2)

        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255

        vis_parsing_anno_color = self.getParsingBasedCombine(self.eyebrow_combine, IsErode=False)
        
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255

        self.final_img = cv2.bitwise_and(self.final_img,img0)

        cv2.imwrite('face_trail.png',img0)
        
        temp = time.time()
        contours,result = self.maintrail_generation.main_part(img0)
        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)
            # self.face_trail_write(final_cont)
            self.trail_write(final_cont, self.face_file, flag=0)

        self.face_file.close()
        print('人脸轨迹的时间==',time.time()-temp)
        # cv2.imwrite('self.final_draw.jpg',self.final_draw)
        
        if self.faceoptimflag:
            self.Optimization(self.face_save_path)# 路径的优化

        f2 = open(self.face_save_path, "r")
        lines = f2.readlines()# 将优化后的路径合并到总路径中
        for line in lines:
            self.trail_file.write(line)

    def face_trail_dln(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
       

        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]

        
        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=2)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255
       
        vis_parsing_anno_color = self.getParsingBasedCombine(self.eyeballs_combine, IsErode=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255
     

        vis_parsing_anno_color = self.getParsingBasedCombine(self.eyebrow_combine, IsErode=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255


        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true


        ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
        graph =  build_sknw(ske)
        final_cont = self.notes_EndToEnd(graph)

       
        
        # self.face_trail_write(final_cont)
        self.trail_write(final_cont, self.face_file, flag=0)
        self.face_file.close()
        
        
        if self.faceoptimflag:
           
            self.Optimization(self.face_save_path)

        f2 = open(self.face_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
            
    def eyeballs_trail1(self):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##zhujingjie
        for i in self.eyeballs_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            eyeballs_cont, eyeballs_draw = self.detailtrail_generation.main_part(img0,1)

            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)

            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)

            eyeballs_cont, eyeballs_draw, final_draw  = self.detailtrail_generation.line_part(img0, self.final_draw)
            self.final_draw = final_draw
            self.final_draw = cv2.bitwise_and(eyeballs_draw[:,:,0], self.final_draw)
            for i in range(len(eyeballs_cont)):
                if eyeballs_cont[i][2] == -33:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'-33'+'\n')
                elif eyeballs_cont[i][2] == 0:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                else:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'33'+'\n')
                    
    def eyebrow_trail1(self, scancodeid):
        ##zhujingjie
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        for i in self.eyebrow_combine:
            img0 = np.zeros(self.img.shape) + 255
            img0 = img0.astype(np.uint8)
            self.vis_parsing_anno[:,:,i] = cv2.dilate(self.vis_parsing_anno[:,:,i], element, iterations = 2)
            index = np.where(self.vis_parsing_anno[:, :, i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            eyeballs_cont,eyeballs_draw = self.detailtrail_generation.main_part(img0,1)
            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)
            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)
            eyebrow_cont, eyebrow_draw, final_draw = self.detailtrail_generation.line_part(img0, self.final_draw)
            self.final_draw = final_draw
            self.final_draw = cv2.bitwise_and(eyebrow_draw[:,:,0], self.final_draw)

            for i in range(len(eyebrow_cont)):
                if eyebrow_cont[i][2] == -33:
                    self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2)) + ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2)) + ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'-33'+'\n')
                elif eyebrow_cont[i][2] == 0:
                    self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
                else:
                    self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'33'+'\n')
                    
            
    def eyeballs_trail_Z(self):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##zhujingjie
        for i in self.eyeballs_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = 0#self.img[index[0], index[1]]
            eyeballs_cont,eyeballs_draw = self.detailtrail_generation.main_part(img0,1)
            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)
            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)
            time_z_first = time.time()
            eyeballs_cont = self.Z_scan.ZLine(img0*255)
#            print("time_zeyeball_done==", time.time()-time_z_first)
#            self.final_draw = final_draw
#            self.final_draw = cv2.bitwise_and(eyeballs_draw[:,:,0], self.final_draw)
            for i in range(len(eyeballs_cont)):
                if eyeballs_cont[i][2] == -33:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'-33'+'\n')
                elif eyeballs_cont[i][2] == 0:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                else:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'33'+'\n')
    def eyebrow_trail_Z(self,scancodeid):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##zhujingjie
        for i in self.eyebrow_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            eyeballs_cont,eyeballs_draw = self.detailtrail_generation.main_part(img0,1)
            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)

            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)
            time_z_first = time.time()
#            cv2.imwrite("./"+str(i)+"_eyebrow.png", img0)
            img0 = img0.astype(np.uint8)
#            print(np.unique(img0))
#            img0 = cv2.imread("./img0.png",0)
            eyeballs_cont = self.Z_scan.ZLine(img0)
#            print("time_zeyebrow_done==", time.time()-time_z_first)
#            print("1",len(eyeballs_cont))
#            self.final_draw = final_draw
#            self.final_draw = cv2.bitwise_and(eyeballs_draw[:,:,0], self.final_draw)
            for i in range(len(eyeballs_cont)):
                if eyeballs_cont[i][2] == -33:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'-33'+'\n')
                elif eyeballs_cont[i][2] == 0:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                else:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'33'+'\n')
                    
    def nose_mouse_trail_Z(self):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##zhujingjie
        for i in self.nose_mouse_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            eyeballs_cont,eyeballs_draw = self.detailtrail_generation.main_part(img0,1)
            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)
            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)
            time_z_first = time.time()
            eyeballs_cont = self.Z_scan.ZLine(img0*255)
#            self.final_draw = final_draw
            self.final_draw = cv2.bitwise_and(eyeballs_draw[:,:,0], self.final_draw)
            for i in range(len(eyeballs_cont)):
                if eyeballs_cont[i][2] == -33:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'-33'+'\n')
                elif eyeballs_cont[i][2] == 0:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                else:
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'33'+'\n')
    #çœ¼ç›
    def eyeballs_trail(self):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##yuzeyuan
        for i in self.eyeballs_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            self.vis_parsing_anno[:,:,i] = cv2.dilate(self.vis_parsing_anno[:,:,i], element, iterations = 5)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            
        
            cv2.imwrite('eyeballs_trail.png', img0)
            temp = time.time()
            eyeballs_cont, eyeballs_draw = self.detailtrail_generation.main_part(img0,8)
            print('眼球轨迹的处理时间==', time.time()-temp)
            
            # self.canny_trail_write(eyeballs_cont)
            self.trail_write(eyeballs_cont, self.trail_file, flag=1)
            
            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
            self.final_img = cv2.bitwise_and(self.final_img,img0)
#        ##zhujingjie
#        for i in self.eyeballs_combine:
#            img0 = np.zeros(self.img.shape)+255
#            img0 = img0.astype(np.uint8)
#            index = np.where(self.vis_parsing_anno[:,:,i]>0)
#            img0[index[0], index[1]] = self.img[index[0], index[1]]
#            eyeballs_cont,eyeballs_draw = self.detailtrail_generation.main_part(img0,1)
#            self.canny_trail_write(eyeballs_cont)
#            self.final_draw = cv2.bitwise_and(eyeballs_draw,self.final_draw)
#            self.final_img = cv2.bitwise_and(self.final_img,img0)
#
#            eyeballs_cont, eyeballs_draw = self.detailtrail_generation.line_part(img0)
#            self.final_draw = cv2.bitwise_and(eyeballs_draw[:,:,0], self.final_draw)
#            for i in range(len(eyeballs_cont)):
#                if eyeballs_cont[i][2] == -33:
#                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
#                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2)) + ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'-33'+'\n')
#                elif eyeballs_cont[i][2] == 0:
#                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
#                else:
#                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'0'+'\n')
#                    self.trail_file.write(str(round(self.factor*eyeballs_cont[i][0], 2))+ ' '+str(round(self.factor*eyeballs_cont[i][1], 2))+' '+'33'+'\n')
    #å˜´å·´å’Œé¼»ï¿?
    #çœ‰æ¯›
    def eyebrow_trail(self, scancodeid):
        
#        ##zhujingjie
#        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#        img0 = np.zeros(self.img.shape) + 255
#        img0 = img0.astype(np.uint8)
#        for i in self.eyebrow_combine:
##            img0 = np.zeros(self.img.shape) + 255
##            img0 = img0.astype(np.uint8)
#            index = np.where(self.vis_parsing_anno[:, :, i]>0)
#            self.vis_parsing_anno[:,:,i] = cv2.dilate(self.vis_parsing_anno[:,:,i], element, iterations = 2)
#            img0[index[0], index[1]] = self.img[index[0], index[1]]
#        self.final_img = cv2.bitwise_and(self.final_img, img0)
#
#        eyebrow_cont, eyebrow_draw, final_draw = self.detailtrail_generation.line_part(img0, self.final_draw)
#        self.final_draw = final_draw
#        self.final_draw = cv2.bitwise_and(eyebrow_draw[:,:,0], self.final_draw)
#
#        for i in range(len(eyebrow_cont)):
#            if eyebrow_cont[i][2] == -33:
#                self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2)) + ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
#                self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2)) + ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'-33'+'\n')
#            elif eyebrow_cont[i][2] == 0:
#                self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
#            else:
#                self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'0'+'\n')
#                self.trail_file.write(str(round(self.factor*eyebrow_cont[i][0], 2))+ ' '+str(round(self.factor*eyebrow_cont[i][1], 2))+' '+'33'+'\n')
        #yuzeyuan
        for i in self.eyebrow_combine:
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            #index = np.where(self.vis_parsing_anno==i)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            self.final_img = cv2.bitwise_and(self.final_img,img0)


            cv2.imwrite('eyebrow_trail_{}.png'.format(i),img0)
            eyebrow_cont, eyebrow_draw = self.detailtrail_generation.main_part(img0,10)
            if scancodeid in ['1024']:#'8ha8aztvnq',
                self.eyebrow_file = open(self.eyebrow_save_path,'w')
                # self.canny_trail_write_two(eyebrow_cont)
                self.trail_write(eyebrow_cont, self.eyebrow_file, flag=1)
                self.eyebrow_file.close()
                
                if self.eyebrowoptimflag:
                    self.Optimization(self.eyebrow_save_path)
                f2 = open(self.eyebrow_save_path, "r")
                lines = f2.readlines()
                for line in lines:
                    self.trail_file.write(line)
            else:
                temp = time.time()
                # self.canny_trail_write(eyebrow_cont)
                self.trail_write(eyebrow_cont, self.trail_file, flag=1)
                print('眉毛的轨迹时间=',time.time()-temp)
                self.final_draw = cv2.bitwise_and(eyebrow_draw,self.final_draw)
    
    
    #å˜´å·´å’Œé¼»ï¿?
    def nose_mouse_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        ##yuzeyuan-gujia
         
        for i in self.nose_mouse_combine:
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]

        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=7)


        img0 = cv2.bitwise_or(self.img, vis_parsing_anno_color)
        self.final_img = cv2.bitwise_and(self.final_img,img0)


        cv2.imwrite('nose_mouse_trail.png',img0)
        temp = time.time()
        contours,result = self.maintrail_generation.main_part(img0)

        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont= self.maintrail_generation.contour_split_loop(real_sub_img)

            #å…ˆç»˜ç”»å‡ºçº¿æ¡
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)
            # self.skelte_trail_write(final_cont)
            self.trail_write(final_cont, self.trail_file, flag=0)
        print('鼻子嘴巴的轨迹时间==', time.time()-temp)

    def nose_mouse_trail_dln(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        ##yuzeyuan-gujia
 
        for i in self.nose_mouse_combine:
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]

        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=7)

        img0 = cv2.bitwise_or(self.img, vis_parsing_anno_color)

        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true

       
        ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
        graph =  build_sknw(ske)

        final_cont = self.notes_EndToEnd(graph)
        # self.skelte_trail_write(final_cont)
        self.trail_write(final_cont, self.trail_file, flag=0)
        
         
            
 
    def neck_dress_trail(self):

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
         
        for i in self.neck_dress_combine: 
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
        
        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1)

    
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255
        
        self.final_img = cv2.bitwise_and(self.final_img,img0)
    
        contours,result = self.maintrail_generation.main_part(img0)
        
        for j in range(len(contours)):
            subcont = contours[j]
 
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result) 
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
 
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)
            # self.skelte_trail_write(final_cont)
            self.trail_write(final_cont, self.trail_file, flag=0)
    def neck_dress_trail_dln(self): #

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
         
        for i in self.neck_dress_combine:
            # index = np.where(self.vis_parsing_anno==i)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            vis_parsing_anno_color = cv2.dilate(self.vis_parsing_anno[:,:,i], element, iterations = 9)
            
            index = np.where(vis_parsing_anno_color>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            # cv2.imwrite('img0_{}.jpg'.format(i),img0)
       

        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1)


        # cv2.imwrite('pp.jpg',vis_parsing_anno_color)
        
    
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255
        # cv2.imwrite('img0000.jpg',img0)


        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true

        cv2.imwrite('neck_dress_trail_dln.png', img*255)
        temp = time.time()
        ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
        graph =  build_sknw(ske)
        print('脖子衣服轨迹时间==', time.time() - temp)

        final_cont = self.notes_EndToEnd(graph)
        # self.skelte_trail_write(final_cont)
        self.trail_write(final_cont, self.trail_file, flag=0)




    def others_trail(self):

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
       
        vis_parsing_anno_color = self.getParsingBasedCombine(self.others_combine, element_kernel=5, iterationsNum=9, IsErode=False, IsDilate=True)
            

        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]
        
        self.final_img = cv2.bitwise_and(self.final_img,img0)
        contours, result = self.maintrail_generation.main_part(img0)
        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
            for i in range(len(final_cont)):
                for k in range(len(final_cont[i])-1):
                    cv2.line(self.final_draw,final_cont[i][k],final_cont[i][k+1],0,1)
            self.trail_write_other(final_cont)


    def others_trail_dln(self):

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
       

        vis_parsing_anno_color = self.getParsingBasedCombine(self.others_combine, element_kernel=5, iterationsNum=9, IsErode=False, IsDilate=True)
            
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]


        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true


        cv2.imwrite('others_trail_dln.png',img*255)
        temp = time.time()
        ske = skeletonize(~img).astype(np.uint16)  # 0-1. 白线黑底
        graph =  build_sknw(ske)
        print('其他的轨迹时间==',time.time() - temp)
        final_cont = self.notes_EndToEnd(graph)

        self.trail_write_other(final_cont)
 

    def changepoints(self,scancodeid, filename=None, factor=1):
        if filename:
            f = open(filename, "r")
        else:
            f = open(self.trail_save_path11, "r")
        danpian_f = open(self.robot_path+scancodeid+"/danpian.txt","w")
        flag = False
        lines = f.readlines()
        n = lines[0].strip("\n").split(" ")
        xxxx0, yyyy0 = float(n[0]),float(n[1])
        newx_0, newy_0 = str(int(xxxx0 * 100)).zfill(5), str(int(yyyy0 * 100)).zfill(5)
        danpian_f.write("B+"+newx_0+"+"+newy_0+"1")
        for i in range(1,len(lines)-1):
            line_b = lines[i].strip("\n").split(" ")
            line_a = lines[i+1].strip("\n").split(" ")
            x1 = int((float(line_a[0])*100-float(line_b[0])*100))
            if x1>=0:
                x1 = "+" + str(x1*factor).zfill(5)
            else:
                x1 = str(x1*factor).zfill(6)
            
            x2 = int((float(line_a[1])*100-float(line_b[1])*100))
            if x2>=0:
                x2 = "+" + str(x2*factor).zfill(5)
            else:
                x2 = str(x2*factor).zfill(6)
            
            if i == 1:
                x3 = str(3)
            else:
                x3 = str(int(float(line_b[2]) // 33 + 1))
            if i==1:
                danpian_f.write("/"+x1+x2+x3)
            else:
                danpian_f.write("/"+x1+x2+x3)
        danpian_f.close()
    

    def changefilenum(self,scancodeid):
        listfinalpath = []
        danpian_all = open(self.robot_path+scancodeid+"/danpian.txt","r")
        
        flag = False
        line = danpian_all.read()
        count = len(line)//9996 + 1
        # print(count,len(line))
        for i in range(count):
            listfinalpath.append(self.robot_path+scancodeid+"/tra@$"+str(count)+str(i)+".txt")
            danpian_sub = open(self.robot_path+scancodeid+"/tra@$"+str(count)+str(i)+".txt","w")
            # danpian_sub.write(line[9996 * i:9996 * (i+1)])
            # danpian_sub.write("B+00000-000001")
            # if i == 0:
            #     danpian_sub.write("B"+line[9996 * i+1:9996 * (i+1)])
            # else:
            danpian_sub.write("B"+line[9996 * i+1:9996 * (i+1)])
            danpian_sub.write('D')
            danpian_sub.close()
        
        return listfinalpath
    
    def clearCircleSideout(self, img,robot_circle_path):
        circle_mask = cv2.imread(robot_circle_path,0)
        circle_mask = cv2.resize(circle_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        index = np.where(circle_mask<127)
        img[index[0], index[1]] =255
        return img
        
    def minlvbo(self, img):
        smalllabel = self.eyebrow_combine + self.eyeballs_combine + self.nose_mouse_combine + [1]
        smalllabelmask = np.ones((img.shape[0], img.shape[1]))
        newvis_parsing_anno = self.vis_parsing_anno.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        for pi in smalllabel:
            newvis_parsing_anno[:,:,pi] = cv2.dilate(newvis_parsing_anno[:,:,pi], element, iterations = 8)
            index = np.where(newvis_parsing_anno[:,:,pi]>0)
            smalllabelmask[index[0], index[1]] = 0
        ccc = img.copy()
        ccc = sfr.minimum(ccc, disk(2))
        img = img *(1-smalllabelmask) + ccc * smalllabelmask
        img = img.astype(np.uint8)
        return img
    
    def ziku(self,down_display=200, left_display=200):
        start_pos = [0, w - down_display]  ##上下的位置 ##w
        words_strokes = []
        for i in range(len(strings)):
            start_pos[0] += left_display  ##左右的位置 1024  ##200
            xx = self.calligraphy.get_all_points(start_pos, strings[i])
            words_strokes = words_strokes + xx
        words_strokes = self.calligraphy.get_arm_strokes(words_strokes)
        for word in words_strokes:
            for stroke in word:
                for ff in range(len(stroke[:,0])-1):
#                        print("fdfs")
                    cv2.line(self.final_draw, (stroke[:, 0][ff], w- stroke[:, 1][ff]), (stroke[:, 0][ff+1],w - stroke[:, 1][ff+1]), (0,0,0),1)
                    if ff==0:
                        self.trail_file.write(str(round(self.factor * stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                        self.trail_file.write(str(round(self.factor *stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'-33'+'\n')
                    else:
                        self.trail_file.write(str(round(self.factor *stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                self.trail_file.write(str(round(self.factor *stroke[:, 0][ff+1], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff+1]), 2))+' '+'0'+'\n')
                self.trail_file.write(str(round(self.factor *stroke[:, 0][ff+1], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff+1]), 2))+' '+'33'+'\n')
    def ziku2(self, strings, down_display=200, left_display=200,factor = 300.0/1600.0,shu_dis=120):
        w = 1600
        calligraphy = Calligraphy()
#        factor = 300.0/1600.0 #40.0/1600.0 #117个字的是60.0-8,40.0-5
#        print(factor)
#        trail_file = open("./ff.txt","w")
#        final_draw = (np.ones((1600,1600,3))*255).astype(np.uint8)
        start_pos = [0, w-down_display]  ##上下的位置 ##200 ##200
        words_strokes = []
        for i in range(len(strings)):
            start_pos[0] += left_display  ##左右的位置 1024  ##200 ##150(需要再靠下
            xx = calligraphy.get_all_points(start_pos, strings[i], heng= True, shu_dis=120)
            words_strokes = words_strokes + xx
        words_strokes = calligraphy.get_arm_strokes(words_strokes)
        y_pianyiliang = 0 #34_7 #26_2#42_21_2 #45.5 #28.5 #8   #55
        for word in words_strokes:
            for stroke in word:
                for ff in range(len(stroke[:,0])-1):
                    cv2.line(self.final_draw, (int(factor*stroke[:, 0][ff]), int(factor*(w- stroke[:, 1][ff]))), (int(factor* stroke[:, 0][ff+1]),int(factor*(w - stroke[:, 1][ff+1]))), (0,0,0),1)
                    if ff==0:
                        self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                        self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'-33'+'\n')
                    else:
                        self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                        
                self.trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2))+' '+'0'+'\n')
                self.trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2))+' '+'33'+'\n')
#            print(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2)))
    def main(self, trial_img_path, scancodeid, writeName, robot_circle_path = '/home/zhujingjie/projects/sketch_wood_v3/collected_model/pics/robot/robot_circle.png'):
        # print("scancodeid", scancodeid)
        # if scancodeid != '8ha8aztc5x':#dlnadd# 单机测试使用
        #     _,self.img = cv2.threshold(self.img, 144, 255, cv2.THRESH_BINARY)
        # _,self.img = cv2.threshold(self.img, 144, 255, cv2.THRESH_BINARY)
        scancodeids = ["1001", "1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1013","1014","1015","1020","1021","1022","1024","1026","1028","1029","1030"] ##"1012""1023""1025,"1027"
        scancodeids_bin = ["1012","1023","1025","1027"]  # and black 8ha8aztvlx ,"1031","8ha8azthmj"
        if (scancodeid in self.scancodeid_shuangren) and (self.vis_parsing_anno.shape[2]==21):#####第21通加进了falg=4
            if [4] in np.unique(self.vis_parsing_anno[-1]):
                self.factor = self.factor * 1.4
        print("print(scancodeid)===:",scancodeid)

        print('=====self.vis_parsing_anno.shape',self.vis_parsing_anno.shape)
        # writeName = "朱静洁"
        scancodeids_zi = ["10211"]#,"1031", "8ha8azthmj"
        if scancodeid in scancodeids_zi:
            nonRarelyFlag = self.RareWordJudgment(writeName)
            print("nonRarelyFlag==", nonRarelyFlag, writeName)
            if nonRarelyFlag:
                stringss = "欢迎" + writeName
                down_display = 650
                for stringi in stringss:
                    strings = stringi.split(" ")
                    self.ziku2(strings, down_display=down_display, left_display=170, factor= 50.0/1600.0,shu_dis=120)
                    down_display += 150

                stringss = "到访杭州妙绘"
                down_display = 650
                for stringi in stringss:
                    strings = stringi.split(" ")
                    self.ziku2(strings, down_display=down_display, left_display=30, factor= 50.0/1600.0,shu_dis=120)
                    down_display += 150
            else:
                stringss = "欢迎"
                down_display = 650
                for stringi in stringss:
                    strings = stringi.split(" ")
                    self.ziku2(strings, down_display=down_display, left_display=170, factor= 50.0/1600.0,shu_dis=120)
                    down_display += 150

                stringss = "到访杭州妙绘"
                down_display = 650
                for stringi in stringss:
                    strings = stringi.split(" ")
                    self.ziku2(strings, down_display=down_display, left_display=30, factor= 50.0/1600.0,shu_dis=120)
                    down_display += 150
        # if scancodeid == "8ha8azthmj":# 单机测试使用
        #     print('单机测试使用8ha8azthmj')
        #     if self.vis_parsing_anno.shape[2]==21:
        #         self.img = self.clearCircleSideout(self.img, robot_circle_path)
        #         self.hair_trail_dln() 
        #         self.face_trail()  
        #         self.eyeballs_trail() 
        #         self.eyebrow_trail(scancodeid) 
        #         self.nose_mouse_trail() 
        #         self.neck_dress_trail_dln() 
        #         self.others_trail_dln() 

        #     self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
        #     self.trail_file.close()
        #     self.hair_file.close()
        #     self.face_file.close()
        #     cv2.imwrite(trial_img_path,self.final_draw)

        #     # 黄机器的结尾
        #     # self.changepoints(scancodeid)
        #     # listfinalpath = self.changefilenum(scancodeid)


        #     # 黑机器的结尾
        #     self.changepoints(scancodeid)
        #     listfinalpath = self.Zipper4b.generateTrackInstancesList(self.robot_path+scancodeid+"/danpian.txt", self.robot_path+scancodeid+"/")
           
        #     finalpath = ";".join(listfinalpath)
        #     return finalpath


        if scancodeid in scancodeids:
           # #shumeipai-robot-Q
           # return self.trail_save_path11
           #  #danpianji-robot-T1
           #  #postcard_draw_area is circle
           # if scancodeid == "1022":
           #     self.factor = self.factor * 1.2
            if self.vis_parsing_anno.shape[2]==21:
                self.img = self.clearCircleSideout(self.img,robot_circle_path)

                # self.img = self.minlvbo(self.img) 
                # self.hair_trail()
                # self.face_trail()  
                # self.eyeballs_trail() 
                # self.eyebrow_trail(scancodeid) 
                # self.nose_mouse_trail() 
                # self.neck_dress_trail() 
                # self.others_trail() 
                 

                # sknw的方法
                print('轨迹生成使用的是Sknw的方法')
                self.hair_trail_dln() 
                # self.face_trail_dln()  ###旧parsing时使用
                self.face_trail()  ##SEGnextparsing时使用
                self.eyeballs_trail() 
                self.eyebrow_trail(scancodeid) 
                self.nose_mouse_trail() 
                self.neck_dress_trail_dln() 
                self.others_trail_dln() 

            else:
                _,self.img = cv2.threshold(cv2.imread(self.stick_img_path,0),144,255,cv2.THRESH_BINARY)
                h,w = self.img.shape[:2]
                ##22
                self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
                if h>w:
                    h_out = 1600
                    w_out = w * h_out // h
                else:
                    w_out = 1600
                    h_out = h * w_out // w
                self.img = cv2.resize(self.img, (w_out,h_out), interpolation=cv2.INTER_NEAREST)
                self.final_img = np.zeros(self.img.shape)+255
                self.final_img = self.final_img.astype(np.uint8)
                self.vis_parsing_anno = np.zeros((self.img.shape[0],self.img.shape[1],21))
                self.img = self.clearCircleSideout(self.img, robot_circle_path)
                self.img = self.minlvbo(self.img)
                self.noparsing_trail()
                
                
            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
            self.trail_file.close()
            self.hair_file.close()
            self.face_file.close()
            cv2.imwrite(trial_img_path,self.final_draw)
            self.changepoints(scancodeid)
            listfinalpath = self.changefilenum(scancodeid)
        else:
            ##robot-T2
            ##postcard_draw_area is circle
            if self.vis_parsing_anno.shape[2]==21:
                self.img = self.clearCircleSideout(self.img,robot_circle_path)
                

                # 原始小俞的方法
                # self.img = self.minlvbo(self.img)
                # self.hair_trail() 
                # self.face_trail() 
                # self.eyeballs_trail()    
                # self.nose_mouse_trail() 
                # self.neck_dress_trail() 
                # self.others_trail()

                # sknw的方法
                print('轨迹生成使用的是Sknw的方法')

                temp = time.time()
                self.hair_trail_dln() 
                print('头发的整体处理时间==',time.time()-temp)
                
                # self.face_trail_dln()  ###旧parsing时使用
                temp = time.time()
                self.face_trail()  ##SEGnextparsing时使用
                print('脸部的整体处理时间==',time.time()-temp)
                
                temp = time.time()
                self.eyeballs_trail() 
                print('眼球的整体处理时间==',time.time()-temp)
                
                temp = time.time()
                self.eyebrow_trail(scancodeid) 
                print('眉毛的整体处理时间==',time.time()-temp)
                
                temp = time.time()
                self.nose_mouse_trail() 
                print('鼻子嘴巴的整体处理时间==',time.time()-temp)

                temp = time.time()
                self.neck_dress_trail_dln() 
                print('脖子衣服的整体处理时间==',time.time()-temp)

                temp = time.time()
                self.others_trail_dln() 
                print('其他的整体处理时间==',time.time()-temp)


                # self.hair_trail()
                # self.face_trail()
                # self.eyeballs_trail()
                # self.eyebrow_trail_Z(scancodeid)
                # self.eyebrow_trail(scancodeid)
                # self.nose_mouse_trail()
                # self.neck_dress_trail()
                # self.others_trail()

            # elif self.vis_parsing_anno.shape[2]==19:###针对交互式绘画的机器生成
            #     # 综合考虑，只对眼睛部分做“有人”时候的绘制，
            #     # 其他部件均采用无人的情况，
            #     # 否则机器人绘画中会出现偏移挪位的问题（因为一开始定位的时候 用的是无人情况下的定位，全部进行修改 太过无麻烦）
            #     print('针对机器人的绘画')
            #     # _,self.img = cv2.threshold(cv2.imread(self.stick_img_path,0),144,255,cv2.THRESH_BINARY)
            #     self.img = cv2.imread(self.stick_img_path,0)
            #     if len(self.img.shape)==3:
            #         self.img = self.img[:,:,0]
            #     h,w = self.img.shape[:2]
       
            #     self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
            #     if h>w:
            #         h_out = 1600
            #         w_out = w * h_out // h
            #     else:
            #         w_out = 1600
            #         h_out = h * w_out // w
            #     self.img = cv2.resize(self.img, (w_out,h_out), interpolation=cv2.INTER_NEAREST)
            #     self.final_draw = np.zeros(self.img.shape)+255
            #     self.final_draw = self.final_draw.astype(np.uint8)
            #     self.final_img = np.zeros(self.img.shape)+255
            #     self.final_img = self.final_img.astype(np.uint8)
            #     # print(self.vis_parsing_anno)
            #     print(self.vis_parsing_anno.shape)

            #     self.vis_parsing_anno = cv2.resize(self.vis_parsing_anno, (w_out,h_out))
            #     # print(self.vis_parsing_anno)
            #     print(self.vis_parsing_anno.shape)

            #     cv2.imwrite('ori.jpg',self.img)
            #     # print('--------1self.img.shape,self.final_img.shape',self.img.shape,self.final_img.shape,self.final_draw.shape,)
            #     # 1600,1600

            #     # self.vis_parsing_anno = np.zeros((self.img.shape[0],self.img.shape[1],21))
            #     self.img = self.clearCircleSideout(self.img, robot_circle_path)
            #     # print('--------2self.img.shape ',self.img.shape)# 1600,1600
            #     self.img = self.minlvbo(self.img)
            #     # print('--------3self.img.shape ',self.img.shape)# 1600,1600
            #     cv2.imwrite('ori_lvbo.jpg',self.img)
            #     self.eyeballs_trail() 
            #     self.eyebrow_trail(scancodeid) 
            #     self.noparsing_trail()


            #     pass
            else:# 无人/praising是两通
                
                _,self.img = cv2.threshold(cv2.imread(self.stick_img_path,0),144,255,cv2.THRESH_BINARY)
                if len(self.img.shape)==3:
                    self.img = self.img[:,:,0]
                h,w = self.img.shape[:2]
                ##22
                self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
                self.vis_parsing_anno = cv2.copyMakeBorder(self.vis_parsing_anno, 50, 50, 50, 50, cv2.BORDER_REPLICATE)

                if h>w:
                    h_out = 1600
                    w_out = w * h_out // h
                else:
                    w_out = 1600
                    h_out = h * w_out // w
                self.img = cv2.resize(self.img, (w_out,h_out), interpolation=cv2.INTER_NEAREST)
                self.final_img = np.zeros(self.img.shape)+255
                self.final_img = self.final_img.astype(np.uint8)
                self.vis_parsing_anno = np.zeros((self.img.shape[0],self.img.shape[1],21))
                self.img = self.clearCircleSideout(self.img, robot_circle_path)
                self.img = self.minlvbo(self.img)
                self.noparsing_trail()
            
            
            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
            self.trail_file.close()
            self.hair_file.close()
            self.face_file.close()
            cv2.imwrite(trial_img_path,self.final_draw)
           # changepoints1(scancodeid)
           # listfinalpath = self.Zipper4b.generateTrackInstancesList("./konghua1.txt", self.robot_path+scancodeid+"/")
            
            self.changepoints(scancodeid)
            listfinalpath = self.Zipper4b.generateTrackInstancesList(self.robot_path+scancodeid+"/danpian.txt", self.robot_path+scancodeid+"/")
       #     finalpath = "/home/zhujingjie/projects/sketch_wood_v3/draw_tool/tra@22.bin"
       #     listfinalpath.append()
        
       # if len(listfinalpath)>1:
        finalpath = ";".join(listfinalpath)
       # else:
       #     finalpath = listfinalpath
        return finalpath


    # def main_temptest(self, trial_img_path, scancodeid, writeName, robot_circle_path = '/home/zhujingjie/projects/sketch_wood_v3/collected_model/pics/robot/robot_circle.png'):
        
    #     scancodeids = ["1001", "1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1013","1014","1015","1020","1021","1022","1024","1026","1028","1029","1030"] ##"1012""1023""1025,"1027"
    #     scancodeids_bin = ["1012","1023","1025","1027"]  # and black 8ha8aztvlx ,"1031","8ha8azthmj"
         
       
         
    #     if scancodeid in scancodeids:
    #         pass
    #     else:
    #         ##robot-T2
    #         ##postcard_draw_area is circle
    #         if self.vis_parsing_anno.shape[2]==21:
    #             # self.img = self.clearCircleSideout(self.img,robot_circle_path)
                
    #             # 原始小俞的方法
    #             # self.img = self.minlvbo(self.img)
    #             # self.hair_trail() 
    #             # self.face_trail() 
    #             # self.eyeballs_trail()    
    #             # self.nose_mouse_trail() 
    #             # self.neck_dress_trail() 
    #             # self.others_trail()

    #             # sknw的方法
    #             print('轨迹生成使用的是Sknw的方法')
    #             temp = time.time()
    #             self.hair_trail_dln() 
    #             print('头发的整体处理时间==',time.time()-temp)
                
    #             # self.face_trail_dln()  ###旧parsing时使用
    #             temp = time.time()
    #             self.face_trail()  ##SEGnextparsing时使用
    #             print('脸部的整体处理时间==',time.time()-temp)
                
    #             temp = time.time()
    #             self.eyeballs_trail() 
    #             print('眼球的整体处理时间==',time.time()-temp)
                
    #             temp = time.time()
    #             self.eyebrow_trail(scancodeid) 
    #             print('眉毛的整体处理时间==',time.time()-temp)
                
    #             temp = time.time()
    #             self.nose_mouse_trail() 
    #             print('鼻子嘴巴的整体处理时间==',time.time()-temp)

    #             temp = time.time()
    #             self.neck_dress_trail_dln() 
    #             print('脖子衣服的整体处理时间==',time.time()-temp)

    #             temp = time.time()
    #             self.others_trail_dln() 
    #             print('其他的整体处理时间==',time.time()-temp)



             
             
            
            
    #         self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
    #         self.trail_file.close()
    #         self.hair_file.close()
    #         self.face_file.close()
    #         cv2.imwrite(trial_img_path, self.final_draw)
          
    #         self.changepoints(scancodeid)
    #         listfinalpath = self.Zipper4b.generateTrackInstancesList(self.robot_path+scancodeid+"/danpian.txt", self.robot_path+scancodeid+"/")
       
    #     finalpath = ";".join(listfinalpath)
       
    #     return finalpath
        
        
def changepoints1(scancodeid):
    f = open("./konghua.txt")
    danpian_f = open("./konghua1.txt","w")
    flag = False
    lines = f.readlines()
    n = lines[0].strip("\n").split(" ")
    newx_0, newy_0 = str(int(float(n[0])*100)).zfill(5), str(int(float(n[1])*100)).zfill(5)
    danpian_f.write("B+"+newx_0+"+"+newy_0+"1")
    for i in range(1,len(lines)-1):
        line_b = lines[i].strip("\n").split(" ")
        line_a = lines[i+1].strip("\n").split(" ")
        x1 = int((float(line_a[0])*100-float(line_b[0])*100))
        if x1>=0:
            x1 = "+" + str(x1).zfill(5)
        else:
            x1 = str(x1).zfill(6)
        x2 = int((float(line_a[1])*100-float(line_b[1])*100))
        if x2>=0:
            x2 = "+" + str(x2).zfill(5)
        else:
            x2 = str(x2).zfill(6)
        
        if i == 1:
            x3 = str(3)
        else:
            x3 = str(int(float(line_b[2]) // 33 + 1))
        if i==1:
            danpian_f.write("/"+x1+x2+x3)
        else:
            danpian_f.write("/"+x1+x2+x3)
    danpian_f.close()
    
def ziku2(strings,down_display=200, left_display=200):
    w = 1600
    calligraphy = Calligraphy()
    factor = 60.0/1600.0
#    print(factor)
    trail_file = open("./ff.txt","w")
    final_draw = (np.ones((1600,1600,3))*255).astype(np.uint8)
    start_pos = [0, w-down_display]  ##上下的位置 ##200 ##200
    words_strokes = []
    for i in range(len(strings)):
        start_pos[0] += left_display  ##左右的位置 1024  ##200 ##150(需要再靠下
        xx = calligraphy.get_all_points(start_pos, strings[i], heng= True)
        words_strokes = words_strokes + xx
    words_strokes = calligraphy.get_arm_strokes(words_strokes)
    for word in words_strokes:
        for stroke in word:
            for ff in range(len(stroke[:,0])-1):
                cv2.line(final_draw, (int(factor*stroke[:, 0][ff]), int(factor*(w- stroke[:, 1][ff]))), (int(factor* stroke[:, 0][ff+1]),int(factor*(w - stroke[:, 1][ff+1]))), (0,0,0),1)
                if ff==0:
                    trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                    trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(factor * (w - stroke[:, 1][ff]), 2))+' '+'-33'+'\n')
                else:
                    trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
            trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(factor *(w - stroke[:, 1][ff+1]), 2))+' '+'0'+'\n')
            trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(factor *(w - stroke[:, 1][ff+1]), 2))+' '+'33'+'\n')
#    cv2.imwrite("ff.jpg", final_draw)
if __name__ == '__main__':

#    img_path = "./1.jpg"
#    trail_path = "./result.txt"
#    parsing_path = "parsing.npy"
    ##logo文件
    Zipper4b = Zipper4b()
    strings = "杭电等你".split(" ")
    ziku(strings,down_display=700, left_display=500)
    changepoints1("fff")
    listfinalpath = Zipper4b.generateTrackInstancesList("./ff1.txt", "./")
#        if scancodeid == "1014":
#            ##画非人脸
#            _,self.img = cv2.threshold(cv2.imread("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/28101643250556_.pic.jpg",0),144,255,cv2.THRESH_BINARY)
#            self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
#            self.img= cv2.resize(self.img,(1600,1600), interpolation=cv2.INTER_NEAREST)
#
#            self.vis_parsing_anno = np.ones((self.img.shape[1],self.img.shape[0],1))
#            self.others_trail()
            ##优化算法
##            trail_file1 = open("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/testPath/all_original.txt", "r")#4
###            trail_file1 = open("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/testPath/all_result_baoli_algorithm.txt", "r")#3
###            trail_file1 = open("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/testPath/all_GA.txt", "r") #1
###            trail_file1 = open("/home/zhujingjie/projects/sketch_wood_v3/draw_tool/testPath/all_result_baoliAndGA.txt", "r")#2
####
##            lines = trail_file1.readlines()
##            for line in lines:
##                self.trail_file.write(line)
##            ##ziku
##            w = 1600
##            strings = "朱静洁".split(" ")
##            calligraphy = Calligraphy()
##            self.ziku()
#
#        else:

#        # 测试头发
#        if scancodeid == "1012":
#            # pass
#            optimalPathUtils = OptimalPathUtils()
#            optimalPath = OptimalPath(optimalPathUtils)
#
#            f2 = open(self.trail_save_path11, "r")
#            lines = f2.readlines()
#            BeginPoint, EndPoint, points = optimalPathUtils.readBeginEndPoint(lines)
#            distance_numpy = np.array(optimalPathUtils.countPointDistance(BeginPoint,EndPoint))
#            visited = np.zeros((distance_numpy.shape[0],1)).astype(np.uint8)
#            optimalPathUtils.countOriDistances(distance_numpy)
#            wayIndexList = optimalPath.findMinDis(distance_numpy)
#            optimalPathUtils.changeTxT(points,wayIndexList,self.trail_save_path11)
        
#    ##写字，横向，剧中
#    strings = "杭电等你".split(" ")
#    ziku(strings,down_display=700, left_display=500)
#    cv2.imwrite(trial_img_path,self.final_draw)
#    listfinalpath = self.changefilenum(scancodeid)
# ##特定路径
#    if scancodeid == "1014":
#    self.changepoints(scancodeid,filename = "/home/zhujingjie/projects/sketch_wood_v3/draw_tool/1/th_coord.txt",factor=3)
#    listfinalpath = self.changefilenum(scancodeid)


#        if scancodeid in scancodeids_zi:
##            if scancodeid == "1022":
##                stringss = ["8"]
#            stringss = ["教育之江","生日快乐"]
#            if scancodeid == "1022": #down_display = 180 left_display=-180 stringss=["亲爱的于天源同学：","你好！恭喜你加入天下杭电E家人的大家庭！","我是智能绘画机器人杭小电，我将带领你了解杭电！","在这里，你将跟随众多国家级、省部级高层次教师","的思想，在知识的海洋里遨游；","在这里，你将参与各类学科竞赛，在电子设计、数"]#2 #7
#            elif scancodeid == "1025":
#                stringss=["亲爱的王子杭同学：","你好！恭喜你加入天下杭电E家人的大家庭！","我是智能绘画机器人杭小电，我将带领你了解杭电！","在这里，你将跟随众多国家级、省部级高层次教师","的思想，在知识的海洋里遨游；","在这里，你将参与各类学科竞赛，在电子设计、数"]#2 #7
#            elif scancodeid == "1027":
#                stringss=["亲爱的王梦晗同学：","你好！恭喜你加入天下杭电E家人的大家庭！","我是智能绘画机器人杭小电，我将带领你了解杭电！","在这里，你将跟随众多国家级、省部级高层次教师","的思想，在知识的海洋里遨游；","在这里，你将参与各类学科竞赛，在电子设计、数"]#2 #7
#            else:
#                stringss=["亲爱的同学：","你好！恭喜你加入天下杭电E家人的大家庭！","我是智能绘画机器人杭小电，我将带领你了解杭电！","在这里，你将跟随众多国家级、省部级高层次教师","的思想，在知识的海洋里遨游；","在这里，你将参与各类学科竞赛，在电子设计、数"]#2 #7
#            stringss = ["学建模、ACM程序设计、智能车等领域大显身手；","在这里，你将有机会与华为、阿里、海康、大华等","大型企业的管理专家、技术专家共话创新创业；","在这里，你也将拥有丰富多彩的校园文化生活，创","造属于你的精彩未来。","非常期待你的到来！我在这里等待你哦！"]#26 ##34
#            down_display = 400
##            factor = 100.0/1600.0
#            for stringi in stringss:
#                strings = stringi.split(" ")
#                self.ziku2(strings, down_display=down_display, left_display=100, factor= 100.0/1600.0)
#                down_display += 200 ##180对应实际的5mm ##216-6mm ##252-7mm
##            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
##            self.trail_file.close()
##            cv2.imwrite(trial_img_path,self.final_draw)
#
#            stringss = ["8"]
##            factor = 300.0/1600.0
#            down_display = 210
#            for stringi in stringss:
#                strings = stringi.split(" ")
#                self.ziku2(strings, down_display=down_display, left_display=100, factor= 300.0/1600.0)
#                down_display += 200 ##180对应实际的5mm ##216-6mm ##252-7mm
#            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
#            self.trail_file.close()
#            cv2.imwrite(trial_img_path,self.final_draw)
#
#
#            self.changepoints(scancodeid)#385
#            listfinalpath = self.changefilenum(scancodeid)



#            stringss = ["未来"]
#            down_display = 250
##            factor = 100.0/1600.0
#            for stringi in stringss:
#                strings = stringi.split(" ")
#                self.ziku2(strings, down_display=down_display, left_display=-50, factor= 200.0/1600.0)
#                down_display += 200 ##180对应实际的5mm ##216-6mm ##252-7mm
##            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
##            self.trail_file.close()
##            cv2.imwrite(trial_img_path,self.final_draw)
#
#            stringss = ["生活节"]
##            factor = 300.0/1600.0
#            down_display = 500
#            for stringi in stringss:
#                strings = stringi.split(" ")
#                self.ziku2(strings, down_display=down_display, left_display=-40, factor= 150.0/1600.0)
#                down_display += 200 ##180对应实际的5mm ##216-6mm ##252-7mm
#            self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
#            self.trail_file.close()
#            cv2.imwrite(trial_img_path,self.final_draw)
            
            
