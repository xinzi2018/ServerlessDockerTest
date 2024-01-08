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
from skimage.morphology import skeletonize


# from sknw import build_sknw 
from .sknw_my import build_sknw 
from .Optimal_Path import OptimalPath,OptimalPathUtils 
from .Trail_Generation import maintrail_generation,detailtrail_generation
from .Z_scan import Z_scan
from .calligraphy import Calligraphy
from .Zipper4b import Zipper4b
import os

class Draw_tool_dlntest:
    def __init__(self, stick_img_path, parsing_path, trail_save_path, max_thresh=150):
        print("stick_img_path", stick_img_path,parsing_path)
        
        
        max_thresh = 63 #63
        self.maintrail_generation = maintrail_generation()
        self.detailtrail_generation = detailtrail_generation()
        self.Zipper4b = Zipper4b()
        self.optimalPathUtils = OptimalPathUtils()
        self.optimalPath = OptimalPath(self.optimalPathUtils)
        self.Z_scan = Z_scan()
 
        self.stick_img_path = stick_img_path
        self.img =  cv2.imread(stick_img_path,0)
        
        self.vis_parsing_anno = np.load(parsing_path)
      
        self.img = cv2.resize(self.img,(1550,1550), interpolation=cv2.INTER_NEAREST)
        self.img = cv2.copyMakeBorder(self.img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value = [255,255,255])
        

        self.vis_parsing_anno = cv2.resize(self.vis_parsing_anno, (1550,1550))
        self.vis_parsing_anno = cv2.copyMakeBorder(self.vis_parsing_anno, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
        
       
        if len(self.img.shape)==3:
            self.img = self.img[:,:,0]
        h,w = self.img.shape[:2]
        self.factor = max_thresh/max((h,w))
        
        
        ##14
        # self.eyebrow_combine = [2-1]
        # self.eyeballs_combine = [3-1, 4-1]
        # self.nose_mouse_combine = [7-1, 8-1, 9-1]
        # self.face_combine = [1-1, 5-1, 6-1, 13-1, 14-1]
        # self.neck_dress_combine = [10-1, 11-1, 12-1]
        # self.glasses_combine = [4-1]
        ##19
        self.eyebrow_combine = [2, 3]
        self.eyeballs_combine = [4, 5, 6]
        self.nose_mouse_combine = [10, 11, 12, 13]
        self.hair_combine = [17]
        self.face_combine = [1, 7, 8, 9, 18]
        self.neck_dress_combine = [14, 15, 16]
        self.glasses_combine = [6]
        self.others_combine = [0]
        if self.vis_parsing_anno.shape[-1]==21:
            # 在使用segnextparisng的时候使用
            # 为了在有眼镜的时候能够将双眼皮以及下眼睑画出来 尝试将眼珠子的范围往外扩
            self.handleParsingSegnextparsing()
        
        self.robot_path = "./robot/"
        
        self.trail_save_path = trail_save_path
        trail_save_name = self.trail_save_path.split('/')[-1]

        self.hair_save_path = self.trail_save_path.replace(trail_save_name, "hair_test.txt")
        self.face_save_path = self.trail_save_path.replace(trail_save_name, "face_test.txt")
        self.eyebrow_save_path = self.trail_save_path.replace(trail_save_name, "eyebrow_test.txt")
        self.noparsing_save_path = self.trail_save_path.replace(trail_save_name, "noparsing_test.txt")
        
        self.trail_file = open(self.trail_save_path,'w')
        self.hair_file = open(self.hair_save_path,'w')
        self.face_file = open(self.face_save_path,'w')
        self.eyebrow_file = open(self.eyebrow_save_path,'w')
        self.noparsing_file = open(self.noparsing_save_path,'w')
        
        self.hairoptimflag = True
        self.faceoptimflag = True
        self.noparoptimflag = True
        self.eyebrowoptimflag = True
        
        # self.calligraphy = Calligraphy()
        # self.textjson = self.calligraphy.words.keys()

        # 一代黄色机器的编号
        self.scancodeids = ["1001", "1002","1003","1004","1005","1006","1007","1008","1009","1010","1011","1013","1014","1015","1020","1021","1022","1024","1026","1028","1029","1030"] ##"1012""1023""1025,"1027"

        # 需要额外写字的机器
        self.scancodeids_zi = ["10211"]#,"1031", "8ha8azthmj"

        self.scancodeid_shuangren = ["1024"]#"8ha8aztvnq",


    def handleParsingSegnextparsing(self,):
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

        self.vis_parsing_anno[:,:,1] = self.vis_parsing_anno[:,:,1] + left_skin + right_skin# 后处理皮肤parsing ,将处理出来的眼皮部分parsing和到皮肤层

        # index = np.where(left_skin==1)
        # self.vis_parsing_anno[:,:,6][index[0],index[1]]=0
        # index = np.where(right_skin==1)
        # self.vis_parsing_anno[:,:,6][index[0],index[1]]=0

        # cv2.imwrite('self.vis_parsing_anno[:,:,1].jpg',self.vis_parsing_anno[:,:,1]*255)
        ####way2结束

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
 
    def trail_write_other(self, final_cont, file):
        for c in final_cont:
            cur_node = [0,0]
            write_list = []
            for p in c[1:-1]:
                dic_check=np.sqrt(np.sum(np.square(cur_node[0]-round(self.factor*p[0],2))+np.square(cur_node[1]-round(self.factor*p[1],2))))
                if dic_check>0.5:
                    cur_node = [round(self.factor*p[0],2),round(self.factor*p[1],2)]
                    write_list.append(cur_node)
            file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'0'+'\n')
            file.write(str(round(self.factor*c[0][0],2))+ ' '+str(round(self.factor*c[0][1],2))+' '+'-33'+'\n')
            for j in write_list:
                file.write(str(j[0])+ ' '+str(j[1])+' '+'0'+'\n')
            file.write(str(round(self.factor*c[-1][0],2))+ ' '+str(round(self.factor*c[-1][1],2))+' '+'33'+'\n')

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

    def yzy_trailGenerate(self, img):
        final_cont_list = []
        contours,result = self.maintrail_generation.main_part(img)
        for j in range(len(contours)):
            subcont = contours[j]
            real_sub_img = self.maintrail_generation.drawed_line_img(subcont,result)
            final_cont = self.maintrail_generation.contour_split_loop(real_sub_img)
            
            final_cont_list.append(final_cont)
        return final_cont_list

    def trailSave(self,final_cont_list, save_file, saveType=0, flag=0):
        for final_cont in final_cont_list:
            if saveType==0:
                self.trail_write(final_cont, save_file, flag=flag)
            elif saveType ==1:
                self.trail_write_other(final_cont, save_file)
        # save_file.close()

    def hair_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        vis_parsing_anno_color = self.getParsingBasedCombine(self.hair_combine, element_kernel=5, iterationsNum=9)
            
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]
      
        # self.final_img = cv2.bitwise_and(self.final_img, img0)
        
        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list,self.hair_file,saveType=0,flag=0)
        
 
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



    def getParsingBasedCombine(self, parsing_combine, element_kernel=-1, iterationsNum=-1, IsErode=True, IsDilate=False):# 此时输入的parsing的通道时21通
        vis_parsing_anno_color = (np.zeros((self.vis_parsing_anno.shape[0], self.vis_parsing_anno.shape[1]))).astype(np.uint8) + 255
        for pi in parsing_combine:
            index = np.where(self.vis_parsing_anno[:, :, pi] > 0)
            vis_parsing_anno_color[index[0], index[1]] = 0
            
            if IsErode == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color, element, iterations=iterationsNum)

            if IsDilate == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.dilate(vis_parsing_anno_color, element, iterations=iterationsNum)

        return vis_parsing_anno_color

    def getParsingBasedCombine2(self,):# 此时输入的parsing时1通的，里面的数值范围时0-20
        vis_parsing_anno_color = (np.zeros((self.vis_parsing_anno.shape[0], self.vis_parsing_anno.shape[1]))).astype(np.uint8) + 255
        for pi in parsing_combine:
            index = np.where(self.vis_parsing_anno==pi)
            vis_parsing_anno_color[index[0], index[1]] = 0
            
            if IsErode == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color, element, iterations=iterationsNum)

            if IsDilate == True:
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (element_kernel, element_kernel))
                vis_parsing_anno_color = cv2.dilate(vis_parsing_anno_color, element, iterations=iterationsNum)

        return vis_parsing_anno_color

        

    def hair_trail_dln(self):
        temp = time.time()
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        vis_parsing_anno_color = self.getParsingBasedCombine(self.hair_combine, element_kernel=5, iterationsNum=9)

        index = np.where(vis_parsing_anno_color == 0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]  # 此时得到的img0表示的是头发部分的全部线条
        # self.final_img = cv2.bitwise_and(self.final_img, img0)  ## 表示最终已经画了的图
        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true
        print('头发预处理时间=',time.time()-temp)


        temp = time.time()
        final_cont = self.Sknw_trailGenerate(img)
        print('头发轨迹生成时间',time.time()-temp)
        temp = time.time()
        self.trailSave([final_cont],self.hair_file, saveType=0, flag=0)
        print('保存txt时间=',time.time()-temp)
       
        

        temp = time.time()
        if self.hairoptimflag:
            self.Optimization(self.hair_save_path)
            print('头发轨迹优化时间=',time.time()-temp)
        

        temp = time.time()
        f2 = open(self.hair_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
        print('头发轨迹保存至总文件的时间=',time.time()-temp)
       


    def Sknw_trailGenerate(self, img):# img的数据范围为True/False，可视化出来的图为黑线白底
        ske = skeletonize(~img).astype(np.uint16)  # ~img:0-1. 白线黑底
        graph =  build_sknw(ske)
        final_cont = self.notes_EndToEnd(graph)
        return final_cont

            
    def noparsing_trail(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        vis_parsing_anno_color = (np.zeros((self.vis_parsing_anno.shape[0], self.vis_parsing_anno.shape[1]))).astype(np.uint8) 

         
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        vis_parsing_anno_color = cv2.erode(vis_parsing_anno_color,element, iterations = 1)
        img0 = self.img


        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list,self.noparsing_file, saveType=0, flag=0)
        
        if self.noparoptimflag:
            self.Optimization(self.noparsing_save_path)
            
        f2 = open(self.noparsing_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
            
 
    def face_trail(self):
        temp = time.time()
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
        print('人脸预处理时间==',time.time()-temp)

        
        
        temp = time.time()
        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list, self.face_file, saveType=0,flag=0)
        print('人脸轨迹生成+存txt的时间==',time.time()-temp)
       
        temp = time.time()
        if self.faceoptimflag:
            self.Optimization(self.face_save_path)# 路径的优化
            print('人脸轨迹优化的时间==', time.time()-temp)


        temp = time.time()
        f2 = open(self.face_save_path, "r")
        lines = f2.readlines() # 将优化后的路径合并到总路径中
        for line in lines:
            self.trail_file.write(line)
        print('人脸轨迹保存至总文件的时间=',time.time()-temp)




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

 
        final_cont = self.Sknw_trailGenerate(img)
       
       
        self.trailSave([final_cont], self.face_file, saveType=0,flag=0)
        
        
        
        if self.faceoptimflag:
            self.Optimization(self.face_save_path)

        f2 = open(self.face_save_path, "r")
        lines = f2.readlines()
        for line in lines:
            self.trail_file.write(line)
            
  
   
    def eyeballs_trail(self):
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        ##yuzeyuan
        for i in self.eyeballs_combine:
            temp = time.time()
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            self.vis_parsing_anno[:,:,i] = cv2.dilate(self.vis_parsing_anno[:,:,i], element, iterations = 5)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            print('{}_眼球预处理时间=='.format(i), time.time()-temp)
            
        
            # cv2.imwrite('eyeballs_trail.png', img0)
            temp = time.time()
            eyeballs_cont, eyeballs_draw = self.detailtrail_generation.main_part(img0,8)
            self.trailSave([eyeballs_cont], self.trail_file, saveType=0,flag=1)
            print('{}_眼球轨迹+存储的处理时间=='.format(i), time.time()-temp)
           


    def eyebrow_trail(self, scancodeid):
       
        #yuzeyuan
        for i in self.eyebrow_combine:
            temp = time.time()
            img0 = np.zeros(self.img.shape)+255
            img0 = img0.astype(np.uint8)
            #index = np.where(self.vis_parsing_anno==i)
            index = np.where(self.vis_parsing_anno[:,:,i]>0)
            img0[index[0], index[1]] = self.img[index[0], index[1]]
            # self.final_img = cv2.bitwise_and(self.final_img,img0)
            print('{}_眉毛预处理时间=='.format(i), time.time()-temp)


            temp = time.time()
            # cv2.imwrite('eyebrow_trail_{}.png'.format(i),img0)
            eyebrow_cont, eyebrow_draw = self.detailtrail_generation.main_part(img0,10)
            if scancodeid in ['1024']:#'8ha8aztvnq',
                # self.eyebrow_file = open(self.eyebrow_save_path,'w')
                 
                self.trailSave([eyebrow_cont], self.eyebrow_file, saveType=0,flag=1)
                # self.trail_write(eyebrow_cont, self.eyebrow_file, flag=1)
                # self.eyebrow_file.close()
                
                if self.eyebrowoptimflag:
                    self.Optimization(self.eyebrow_save_path)
                f2 = open(self.eyebrow_save_path, "r")
                lines = f2.readlines()
                for line in lines:
                    self.trail_file.write(line)
            else:
                temp = time.time()
                self.trailSave([eyebrow_cont], self.trail_file, saveType=0,flag=1)
                print('{}_眉毛轨迹+存储的处理时间=='.format(i), time.time()-temp)
    
    
  
    def nose_mouse_trail(self):
        temp = time.time()
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        ##yuzeyuan-gujia
   
        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, IsErode=False, IsDilate=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]


        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=7)
        img0 = cv2.bitwise_or(self.img, vis_parsing_anno_color)
        print('鼻子嘴巴预处理时间==', time.time()-temp)

        # cv2.imwrite('nose_mouse_trail.png',img0)
        temp = time.time()
        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list,self.trail_file, saveType=0,flag=0)
        print('鼻子嘴巴的轨迹+保存时间==', time.time()-temp)


    def nose_mouse_trail_dln(self):
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
    

        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, IsErode=False, IsDilate=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]] 


        vis_parsing_anno_color = self.getParsingBasedCombine(self.nose_mouse_combine, element_kernel=3, iterationsNum=7)

        img0 = cv2.bitwise_or(self.img, vis_parsing_anno_color)

        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true

        final_cont = self.Sknw_trailGenerate(img)
        self.trailSave([final_cont],self.trail_file, saveType=0, flag=0)
 
 
    def neck_dress_trail(self):

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        
        vis_parsing_anno_color = self.getParsingBasedCombine(self.neck_dress_combine, IsErode=False, IsDilate=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]] 


        
        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1)

    
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255
        
        # self.final_img = cv2.bitwise_and(self.final_img,img0)

         

        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list, self.trail_file, saveType=0, flag=0)
     

    def neck_dress_trail_dln(self): #
        temp = time.time()
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
        

        vis_parsing_anno_color = self.getParsingBasedCombine(self.neck_dress_combine,element_kernel=3,iterationsNum=9,IsErode=False, IsDilate=True)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]



        vis_parsing_anno_color = self.getParsingBasedCombine(self.face_combine, element_kernel=3, iterationsNum=1,IsErode=True, IsDilate=False)
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = 255 


        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true

        print('脖子衣服预处理时间==', time.time()-temp)

        # cv2.imwrite('neck_dress_trail_dln.png', img*255)
         
        temp = time.time()
        final_cont = self.Sknw_trailGenerate(img)
        self.trailSave([final_cont], self.trail_file, saveType=0, flag=0)
        print('脖子衣服的轨迹+保存时间==', time.time()-temp)
      



    def others_trail(self):

        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
       
        vis_parsing_anno_color = self.getParsingBasedCombine(self.others_combine, element_kernel=5, iterationsNum=9, IsErode=False, IsDilate=True)
            
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]
        
        # self.final_img = cv2.bitwise_and(self.final_img,img0)


        final_cont_list = self.yzy_trailGenerate(img0)
        self.trailSave(final_cont_list, self.trail_file, saveType=1)
        
       

    def others_trail_dln(self):
        temp = time.time()
        img0 = np.zeros(self.img.shape).astype(np.uint8) + 255
       

        vis_parsing_anno_color = self.getParsingBasedCombine(self.others_combine, element_kernel=5, iterationsNum=9, IsErode=False, IsDilate=True)
            
        index = np.where(vis_parsing_anno_color==0)
        img0[index[0], index[1]] = self.img[index[0], index[1]]


        ret, img = cv2.threshold(img0, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img[img == 255] = 1
        img = (img == 1)  # 0-1 转换成false-true
        print('其他预处理时间==', time.time()-temp)

        # cv2.imwrite('others_trail_dln.png',img*255)
        temp = time.time()
        final_cont = self.Sknw_trailGenerate(img)

        # self.trail_write_other(final_cont)
        self.trailSave([final_cont], self.trail_file, saveType=1)
        print('其他的轨迹+保存时间==', time.time()-temp)
 

    def changepoints(self,scancodeid, filename=None, factor=1):
        if filename:
            f = open(filename, "r")
        else:
            f = open(self.trail_save_path, "r")
        if not os.path.exists(self.robot_path+scancodeid):
            os.mkdir(self.robot_path+scancodeid)
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
    

    def changefilenum(self, scancodeid):
        listfinalpath = []
        if not os.path.exists(self.robot_path+scancodeid):
            os.mkdir(self.robot_path+scancodeid)
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
    
#     def ziku(self,down_display=200, left_display=200):
#         start_pos = [0, w - down_display]  ##上下的位置 ##w
#         words_strokes = []
#         for i in range(len(strings)):
#             start_pos[0] += left_display  ##左右的位置 1024  ##200
#             xx = self.calligraphy.get_all_points(start_pos, strings[i])
#             words_strokes = words_strokes + xx
#         words_strokes = self.calligraphy.get_arm_strokes(words_strokes)
#         for word in words_strokes:
#             for stroke in word:
#                 for ff in range(len(stroke[:,0])-1):
# #                        print("fdfs")
#                     # cv2.line(self.final_draw, (stroke[:, 0][ff], w- stroke[:, 1][ff]), (stroke[:, 0][ff+1],w - stroke[:, 1][ff+1]), (0,0,0),1)
#                     if ff==0:
#                         self.trail_file.write(str(round(self.factor * stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
#                         self.trail_file.write(str(round(self.factor *stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'-33'+'\n')
#                     else:
#                         self.trail_file.write(str(round(self.factor *stroke[:, 0][ff], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
#                 self.trail_file.write(str(round(self.factor *stroke[:, 0][ff+1], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff+1]), 2))+' '+'0'+'\n')
#                 self.trail_file.write(str(round(self.factor *stroke[:, 0][ff+1], 2)) + ' '+str(round(self.factor *(w - stroke[:, 1][ff+1]), 2))+' '+'33'+'\n')
#     def ziku2(self, strings, down_display=200, left_display=200,factor = 300.0/1600.0,shu_dis=120):
#         w = 1600
#         calligraphy = Calligraphy()
# #        factor = 300.0/1600.0 #40.0/1600.0 #117个字的是60.0-8,40.0-5
# #        print(factor)
# #        trail_file = open("./ff.txt","w")
# #        final_draw = (np.ones((1600,1600,3))*255).astype(np.uint8)
#         start_pos = [0, w-down_display]  ##上下的位置 ##200 ##200
#         words_strokes = []
#         for i in range(len(strings)):
#             start_pos[0] += left_display  ##左右的位置 1024  ##200 ##150(需要再靠下
#             xx = calligraphy.get_all_points(start_pos, strings[i], heng= True, shu_dis=120)
#             words_strokes = words_strokes + xx
#         words_strokes = calligraphy.get_arm_strokes(words_strokes)
#         y_pianyiliang = 0 #34_7 #26_2#42_21_2 #45.5 #28.5 #8   #55
#         for word in words_strokes:
#             for stroke in word:
#                 for ff in range(len(stroke[:,0])-1):
#                     # cv2.line(self.final_draw, (int(factor*stroke[:, 0][ff]), int(factor*(w- stroke[:, 1][ff]))), (int(factor* stroke[:, 0][ff+1]),int(factor*(w - stroke[:, 1][ff+1]))), (0,0,0),1)
#                     if ff==0:
#                         self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
#                         self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'-33'+'\n')
#                     else:
#                         self.trail_file.write(str(round(factor * stroke[:, 0][ff], 2)) + ' '+str(round(y_pianyiliang+factor * (w - stroke[:, 1][ff]), 2))+' '+'0'+'\n')
                        
#                 self.trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2))+' '+'0'+'\n')
#                 self.trail_file.write(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2))+' '+'33'+'\n')
# #            print(str(round(factor * stroke[:, 0][ff+1], 2)) + ' '+str(round(y_pianyiliang+factor *(w - stroke[:, 1][ff+1]), 2)))
    

    def wordsTrail(self,writeName):
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



    def main(self, trial_img_path, scancodeid, writeName, robot_circle_path = '/home/zhujingjie/projects/sketch_wood_v3/collected_model/pics/robot/robot_circle.png'):
         

        if self.vis_parsing_anno.shape[2]==21:
            self.img = self.clearCircleSideout(self.img,robot_circle_path)
            # sknw的方法  
            print('轨迹生成使用的是Sknw的方法')
            self.hair_trail_dln() # 头发轨迹生成方法

             

        self.trail_file.write('0'+' '+'0'+' '+'0'+'\n')
        self.trail_file.close()
        self.hair_file.close()
        

        if scancodeid in self.scancodeids: # 黄机器
            self.changepoints(scancodeid)
            listfinalpath = self.changefilenum(scancodeid)

        else: # 黑机器
            self.changepoints(scancodeid)
            listfinalpath = self.Zipper4b.generateTrackInstancesList(self.robot_path+scancodeid+"/danpian.txt", self.robot_path+scancodeid+"/")


        finalpath = ";".join(listfinalpath)
        return finalpath

 

            
