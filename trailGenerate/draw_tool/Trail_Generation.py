# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:26:09 2020

@author: mhxyy
"""
import numpy as np
import cv2
from queue import Queue
from skimage import measure, color
'''
球型
23行 element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
64行 #iteration = 8
drwa-tool.py max_thresh=130
板型
23行element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
64行iteration = 8
drwa-tool.py max_thresh=63
'''
class detailtrail_generation(object):
    def trail_gen(self,img,element_type = None):
        if element_type == 1:
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            img = cv2.erode(img,element)
        elif element_type == 2:
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            img = cv2.dilate(img,element)
        #_,cont,h = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cont,h = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        contours=[]
        C=[]
        for c in cont:
            #if cv2.contourArea(c)>0:
            tmp=[tuple(x[0]) for x in c ]
            contours.append(tmp)
            
        for j in range(len(contours)):
            c=contours[j]
    
            if len(c)<=5 and len(c)>4:
                C.append(c)
            elif len(c)>5:
                C.append(c)
        
        draw_out=np.zeros(img.shape,np.uint8)+255
        #for c in CMOVE:
        #如果对输入图像做形态学处理，那连接点之间的线宽为
        if element_type:
            for c in C:
                for i in range(len(c)-1):
                    cv2.line(draw_out,c[i],c[i+1],0,1)
                cv2.line(draw_out,c[0],c[-1],0,1)
        else:
            for c in C:
                for i in range(len(c)-1):
                    cv2.line(draw_out,c[i],c[i+1],0,1)
                cv2.line(draw_out,c[0],c[-1],0,1)
       #return CMOVE, draw_out
        return C, draw_out
    
    
    def detial_add(self,img,iteration=0):
#        #zhujingjie
#        iteration = 10
        Contours,edge = self.trail_gen(img)
        if iteration:
            for i in range(iteration):
                temp = cv2.subtract(edge,255 - img)
                #cv2.imshow("out222",255 - temp)
    
                #判断分离出来的哪几层需要做形态学处理
                if i<1:
                    contours,edge1 = self.trail_gen(temp)
                else:
                    #elemnt 1 for eroded,2 for dilated
                    contours,edge1 = self.trail_gen(temp, 1)
    
    #                cv2.imshow("out111",edge1)
    #                cv2.waitKey(0)
    #                cv2.destroyAllWindows()
                if i!=1:
                    contours.extend(Contours)
                final = cv2.bitwise_and(edge,edge1)
                Contours = contours.copy()
                edge = final.copy()
            return contours,final
        else:
            return Contours,edge
    
#    def main_part(self,img,filename='trail.txt'):    
    def main_part(self,img,iteration = 3):  
        h,w=img.shape

        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        #计算缩放系数：
#        factor = 150/max((h,w))
        
        #contours,edge = Trail_Gen.trail_gen(img)
        contours,final= self.detial_add(img,iteration)
        #final= detial_add(img,edge,contours,7)
            
#        file1 = open(filename,'a')
        contours = contours[::-1]

        return contours,final

    def line_part(self,img, final_draw):
        h,w=img.shape
        contours = []
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        flag = False    ##
        cont_finalj = -1
        ##法1111
        numbb, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - img, connectivity=8)
        
        ##法222
#        labels = measure.label(img, connectivity = 2)
#        dst = color.label2rgb(labels)
#        print(dst.shape)
#        if len(dst.shape) == 3:
#            numbb = 1
#        else:
#            numbb = dst.shape[3]
        for i in range(numbb):
            if (stats[i][0]+ stats[i][2]) == img.shape[1]:
                print("pass, continue")
            ##遍历每行、每列
            else:
                output = np.ones((img.shape[0], img.shape[1])) * 255
                mask = labels == i
                output[:,:][mask] = 0
                print(np.unique(output))
                output = output.astype(np.uint8)
                eyebrow_cont, eyebrow_draw = self.main_part(output, 1)
                final_draw = cv2.bitwise_and(eyebrow_draw, final_draw)
#                self.canny_trail_write(eyebrow_cont)
                
                for j in range(0, output.shape[0], 4): #8
                    for i in range(0, output.shape[1]):
                        if output[j][i] == 0:
                            if flag == False:
                                if (j == cont_finalj):
                                    contours[-1][2] = 33
                                if (len(contours) == 0):
                                    contours.append([i,j,-33])
                                elif (len(contours) != 0) and (contours[-1][2] == 33):
                                    contours.append([i,j,-33])
                                else:
                                    contours.append([i,j,0])
                                flag = True
                        else:
                            if flag == True:
                                flag =False
                                if (j != cont_finalj):
                                    contours.append([i-1,j,0])
                                else:
                                    contours.append([i-1,j,33])
                                cont_finalj = j
        if len(contours) != 0:
            contours[-1][2] = 33
        final = np.zeros((img.shape[0], img.shape[1],3), np.uint8)+255
        for i in range(0, len(contours)-2, 2):
            pt_pos1 = (int(contours[i][0]), int(contours[i][1]))
            pt_pos2 = (int(contours[i+1][0]), int(contours[i+1][1]))
            cv2.line(final, pt_pos1, pt_pos2, (0, 0, 0), 1)
        return contours, final, final_draw
        
class maintrail_generation(object):
    #def __init__(self):
    #设置窗口函数
    def region(self,img,p):
        #M为画了一个子轮廓的图像
        #p为当前遍历的点
        r=1
        #r为检测返回，r=1位9宫格，相邻8个像素点
        
        l=[(x,y) for y in range(p[1]-r,p[1]+r+1) for x in range(p[0]-r,p[0]+r+1)
           if y>=0 and y<img.shape[0] and x>=0 and x<img.shape[1]]
        #返回所有相邻点的坐标
        return l

    def thinning(self, img):
        img = 255 - img
        img1 = img.copy()
        # Structuring Element
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        # Create an empty output image to hold values
        thin = np.zeros(img.shape,dtype='uint8')
        while (cv2.countNonZero(img1)!=0):
            # Erosion
            erode = cv2.erode(img1,kernel)
            # Opening on eroded image
            opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
            # Subtract these two
            subset = erode - opening
            # Union of all previous sets
            thin = cv2.bitwise_or(subset,thin)
            # Set the eroded image for next iteration
            img1 = erode.copy()
        return 255 - thin
        
    def Skeletonization(self,img,Blur = True):
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        
        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False
        
        img =cv2.bitwise_not(img)    
        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
        
            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True    
        ret,img2 = cv2.threshold(cv2.GaussianBlur(skel,(3,3),0),1,255,0)
        return cv2.erode(img2,element,iterations = 1)
    #广度遍历函数
    def bfs(self,drawed,point):
        #E为画了一个子轮廓的图像
        m=drawed.copy()
        Q=[]
        #建立队列，每检测一个子节点，计入队列
        Q.append(point)
        #当队列不为0
        while len(Q)!=0:
            #cv2.imshow('1',m)
            #cv2.waitKey(0)
            #删除队列中第一个数，并返回该数
            p0=Q.pop(0)
            #对每一个8连通域内的点，如果等于255（线上一点），添加队列
            for p in self.region(drawed,p0):
                if m[p[1]][p[0]]==255:
                    Q.append(p)
    
                    #原矩阵上抹去这个点
                    m[p[1]][p[0]]=0
        #返回最后的点
        pr=p0
        return pr
        
    #深度遍历函数
    def dfs(self, drawed, point):
        #E为画了一个子轮廓的图像
        m = drawed.copy()
        queue= Queue()
        #建立队列，每检测一个子节点，计入队列
        queue.put(point)
        #当队列不为0
        while not queue.empty():
            #删除队列中第一个数，并返回该数
            p0 = queue.get(0)
            #对每一个8连通域内的点，如果等于255（线上一点），添加队列
            for p in self.region(drawed,p0):
                if m[p[1]][p[0]]==255:
                    queue.put(p)
                    #原矩阵上抹去这个点
                    m[p[1]][p[0]]=0
        #返回最后的点
        pr = p0
        return pr

    def findclosest_index(self,C,p):
        #C为每一个子轮廓
        #p为遍历出来的端点
        val=9999
        index=-1
        for i,c in enumerate(C):
            #计算轮廓中的每一个点和端点的距离
            dis2=(c[0]-p[0])**2+(c[1]-p[1])**2
            #如果距离小于val，赋值val，记录当前index
            if dis2<val:
                val=dis2
                index=i
        return index
            
    def cont_detect(self,result,dot_filter=None):
         #首先进行边缘检测
         #result = real_sub_img
        cont,h = cv2.findContours(result,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        #将边缘检测得到的cont以tuple形式储存
    #    area = cv2.contourArea(cont[0])
    #    print(area)
        contours=[]
        C=[]
        for c in cont:
#            area = cv2.contourArea(c)
#            if area>0:
#                tmp=[tuple(x[0]) for x in c ]
#                contours.append(tmp)
            tmp=[tuple(x[0]) for x in c ]
            contours.append(tmp)
            
        if dot_filter:
            for j in range(len(contours)):
                c=contours[j]
        
                if len(c)>4:
                    C.append(c)
            return C
        else:
            return contours

    def trail_generate(self,subcont,drawed_line):
        #取contours上起始点开始搜索，进行bfs，搜索出最远的点
        initial_point=subcont[0]
        #计算bfs从线条图像和初始点钟
        far_point=self.bfs(drawed_line,initial_point)
#        #计算dfs从线条图像和初始点钟
#        far_point=self.dfs(drawed_line,initial_point)
        #从子轮廓中寻找最靠近检索出的端点的点    
        far_point_index=self.findclosest_index(subcont,far_point)
        final_cont=subcont[0:far_point_index]
        #第一轮获取出的线条笔画

        return final_cont      

    def drawed_line_img(self,subcont,formwork):
        #先绘画出线条
        drawed=np.zeros(formwork.shape)
        for i in range(1,len(subcont)):
            cv2.line(drawed,subcont[i-1],subcont[i],(255),2)
        cv2.line(drawed,subcont[-1],subcont[0],(255),2)  
        real_sub_img = cv2.bitwise_and(np.uint8(drawed),formwork)
        return real_sub_img

    def contour_split_loop(self,img):
        #设置一个计数器
        total_cont=[]
        draw_confirm = True
        limit_cont = 0
        while draw_confirm:
            #首先对某线条进行边缘检测，子线条开启dot过滤
            subcont = self.cont_detect(img,dot_filter=True)
            if len(subcont)>1:
                for i in range(len(subcont)):
                    #drawd_line_img函数，从模板上，抠出想要绘画的线条部分
                    #得到的real_sub_img，是模板对应的子线条
                    real_sub_img = self.drawed_line_img(subcont[i],img)
                    #从该子线条中获取路径
                    final_cont=self.trail_generate(subcont[i],real_sub_img)
                    #从原线条图中划区该线条
                    if final_cont:
                        total_cont.append(final_cont)
                        #从原线条图中划区该线条
                        for i in range(len(final_cont)-1):
                            cv2.line(img,final_cont[i],final_cont[i+1],0,2)
            elif len(subcont)==1 and len(subcont[0])>6:
    
                #drawd_line_img函数，从模板上，抠出想要绘画的线条部分
                #得到的real_sub_img，是模板对应的子线条
                real_sub_img = self.drawed_line_img(subcont[0],img)
                #从该子线条中获取路径
                final_cont=self.trail_generate(subcont[0],real_sub_img)
                #从原线条图中划区该线条
                if final_cont:
                    total_cont.append(final_cont)
                    #从原线条图中划区该线条
                    for i in range(len(final_cont)-1):
                        cv2.line(img,final_cont[i],final_cont[i+1],0,2)                         
            else:
                draw_confirm = False
            limit_cont = limit_cont+1
            if limit_cont >3:
                draw_confirm = False
            
            
        return total_cont

    def main_part(self,img):
#        result = self.thinning(img)
        result = self.Skeletonization(img)
        #cv2.imshow("original", img)
        #cv2.imwrite("trail_data/dilate-skeleton.jpg",255 - result)
        contours = self.cont_detect(result)  
        contours = contours[::-1]
        return contours,result
    
if __name__ == '__main__':
    maintrail_generation =maintrail_generation()
    img = cv2.imread("1.png",0)
    filename = 'trail.txt'
    maintrail_generation.main_part(img,filename)
    detailtrail_generation = detailtrail_generation()
    img = cv2.imread('./1s.png',0)
    detailtrail_generation.main_part(img,filename)
