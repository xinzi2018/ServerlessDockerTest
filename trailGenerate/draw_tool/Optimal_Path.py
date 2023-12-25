#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : OptimalPath.py
# @Author: Dailingna
# @time: 2021/12/15 上午11：00
# @Desc  :

import os
import cv2
import numpy as np
import math
import time


class OptimalPathUtils(object):
    def __init__(self):
        pass
    # 调节txt中节点顺序
    def changeTxT(self,points,wayIndexList,savepath):  
        with open(savepath,"w") as f:
#            print("optim_5", len(wayIndexList))
            for i in wayIndexList:
                for point in points[str(i)]:
                    f.write(point) 
#            f.write(points[str(len(points)-1)][0])  # 增加上最后一行0 0 0
        f.close()

    # 计算最原始的消耗距离：
    def countOriDistances(self,distance_numpy):
        total_sum = 0
#        print(distance_numpy.shape)
        for i in range(distance_numpy.shape[0]-1):
            total_sum = total_sum + distance_numpy[i][i+1]
#        print('计算最原始的消耗距离=',total_sum)
        return total_sum

    def countPointDistance(self,BeginPoint,EndPoint):#计算当前这条线终点坐标和另外所有线条的起点坐标的距离，并保存成矩阵 （矩阵大小为n*n）
        distance_numpy = np.zeros((len(BeginPoint),len(EndPoint))).astype(np.float64)
        for i in range(len(EndPoint)): # 遍历每个终点到起点的距离
            for j in range(len(BeginPoint)):
                if j == i : # 如果起点终点是同一条线，则距离给无穷大
                    distance_numpy[j][i] =  float('inf') 

                else:
                    distance_numpy[j][i] =math.sqrt(math.pow(EndPoint[i][0]-BeginPoint[j][0],2) + math.pow(EndPoint[i][1]-BeginPoint[j][1],2)) # 计算距离
        # #优化
        # distance_numpy = math.sqrt(math.pow(EndPoint[:,0]-BeginPoint[:,0],2)+math.pow(EndPoint[:,1]-BeginPoint[:,1],2))
        # for  i in range(len(EndPoint)):
        #     distance_numpy[i,i]=float('inf')
#        print("optim_3", len(distance_numpy))
        return distance_numpy
    def readBeginEndPoint(self,lines): # 处理出每条线（n条线）的起点和终点坐标
        BeginPoint,EndPoint = [],[]
        points = {}
        num=0
        flag=0
#        print("optim_1", len(lines))
        for line in lines:
            # print(line)
            if int(line.split(' ')[-1]) == 0 and flag==0:
                points[str(num)]=[]
                flag=1

            # print( [float(line.split(' ')[0]),float(line.split(' ')[1])])
            if int(line.split(' ')[-1]) == -33:  # -33表示落笔（起点），33表示抬笔（落笔）
                
                BeginPoint.append([float(line.split(' ')[0]),float(line.split(' ')[1])])
            elif int(line.split(' ')[-1]) == 33:
                EndPoint.append([float(line.split(' ')[0]),float(line.split(' ')[1])])  
                flag=0
            # 因为-33和33成对出现。所以BeginPoint[i]和EndPoint[i]表示同一条线的落笔（起点）和抬笔（落笔）
            points[str(num)].append(line)
            
            if int(line.split(' ')[-1]) == 33:
                num = num+1

#        print('------')
#        print("optim_2", len(points), len(BeginPoint))
        return BeginPoint,EndPoint,points



class OptimalPath(object):
    def __init__(self,OptimalPathUtils=None):
        self.OptimalPathUtils = OptimalPathUtils

    # 直接随机选择一个点作为起点，找最短的路径
    #       此时找最短路径有两种方法：第一个方法是直接简单粗暴每一次都找最近的一个点，遍历过的就跳过
    #                            第二个方法是适用贪心算法去得到
    # 需要对比这三种方法的时间、空间复杂度、计算得到的最短距离、以及机器绘画的时间
    ########################方法一#################################
    def findMinIndex(self,distances,MinIndex,visited):
        mindis = float('inf')
        for i in range(distances.shape[0]):
            if distances[i]<mindis and visited[i]==0 and i!= MinIndex:# 如果当前距离比最小距离小，则更新
            
                mindis = distances[i]
                minindex = i
        # print(mindis,minindex)
        return mindis,minindex

    # 直接用最朴素的方法去处理，寻找每一行的最小距离
    def findMinDis(self,distance_numpy):
        # 需要有一个visited去记录节点是否被访问
        visited = (np.zeros((distance_numpy.shape[0],1)).astype(np.uint8))
        # 直接从distance_numpy[0][0]开始访问
        # for i in range(distance_numpy.shape[1]):
        MinIndex = 0
        totaldis = 0
        way = []
        for i in range(distance_numpy.shape[0]-1): # 因为不需要回到起点，所以会有一行距离是不被遍历到的
            visited[MinIndex] = 1
            way.append(MinIndex)
#            print('当前遍历点为：',MinIndex+1)
#            print("fdfss", distance_numpy[MinIndex])
            mindis,minindex = self.findMinIndex(distance_numpy[MinIndex],MinIndex,visited)
            
            # print('visited=',visited)
#            print(mindis)
            totaldis = totaldis+mindis
            MinIndex = minindex
        for i in range(len(visited)):
            if visited[i]==0:
#                print("visitedvisitedvisitedvisitedvisited", i)
                way.append(i)
#        print("optim_4", len(way))
#        print('总的消耗距离为：',totaldis)
        return way
        # print('检查visited:',visited)

    ########################方法二#################################
    def greedy_algorithm(self,D,S,i=0):
        n = D.shape[0]
        # int i,j,k,l;
        # int S[n];//用于存储已访问过的城市
        # int D[n][n];//用于存储两个城市之间的距离
        # int sum = 0;//用于记算已访问过的城市的最小路径长度
        # int Dtemp;//保证Dtemp比任意两个城市之间的距离都大（其实在算法描述中更准确的应为无穷大）
        # int flag;//最为访问的标志，若被访问过则为1，从未被访问过则为0
        # /*初始化*/
        # i = 1;//i是至今已访问过的城市
        # S[0] = 0;
        way = []
        result_sum = 0
        S[0]=0
#        print('n=',n)
        way.append(0)
        while i<n:
            k,Dtemp = 1,1000000000;
            # print('i=',i)
            while k<n:
                # print('k=',k)
                l,flag = 0,0
                while l<i:
                    if S[l]==k: # 判断该城市是否已被访问过，若被访问过，
                        flag=1 #则flag为1
                        # print('break')
                        break
                    else:
                        l = l+1
                if flag==0 and D[k][S[i-1]]<Dtemp: #D[k][S[i - 1]]表示当前未被访问的城市k与上一个已访问过的城市i-1之间的距离
                    # print('D[k][S[i-1]]=',D[k][S[i-1]])
                    j = k #j用于存储已访问过的城市k
                    # print(j)
                    Dtemp = D[k][S[i-1]] #Dtemp用于暂时存储当前最小路径的值
                    # print('Dtemp=',Dtemp)
                k = k+1
            # if i+1 == n:
            #     break

            S[i]=j # 将已访问过的城市j存入到S[i]中
            i =i+1
            way.append(j)

            # print('end-i=',i)
            # print('终Dtemp=',Dtemp)

            result_sum = result_sum+Dtemp #求出各城市之间的最短距离，注意：在结束循环时，该旅行商尚未回到原出发的城市
            # print(result_sum-Dtemp)
#        print('=====')
        # print(S)
#        print(way[:-1])
#        print('一遍贪心算法消耗距离=',result_sum-Dtemp)
        return way[:-1]



def main():
    optimalPathUtils = OptimalPathUtils()
    optimalPath = OptimalPath(optimalPathUtils)
    # 读取txt 文档
    # f2 = open("/home/zhujingjie/projects/dln_testcode/Daily_code/hair_result_greedy_algorithm.txt","r")
    f2 = open("/home/zhujingjie/projects/dln_testcode/Daily_code/hair_result.txt","r")
    lines = f2.readlines() 



    BeginPoint,EndPoint,points = optimalPathUtils.readBeginEndPoint(lines)
    distance_numpy = np.array(optimalPathUtils.countPointDistance(BeginPoint,EndPoint))
    visited = np.zeros((distance_numpy.shape[0],1)).astype(np.uint8)
    # distance_numpy = np.array([[0,2,4,5,1],[2,0,6,5,3],[4,6,0,8,3],[5,5,8,0,5],[1,3,3,5,0]])
#    print('distance_numpy.shape=',distance_numpy.shape)
    
    D = np.array([[0,1,6,4],[1,0,2,3],[6,2,0,5],[4,3,5,0]])
    S = np.zeros((4,1)).astype(np.uint8)

    start = time.time() 
    optimalPathUtils.countOriDistances(distance_numpy)

    # findMinDis(distance_numpy)
    # greedy_algorithm(D,S)
    wayIndexList = optimalPath.greedy_algorithm(distance_numpy,visited)
#    print(len(wayIndexList))
    save_path = "/home/zhujingjie/projects/dln_testcode/Daily_code/hair_result_greedy_algorithm.txt"
    optimalPathUtils.changeTxT(points,wayIndexList,save_path)

    end = time.time()
#    print('总消耗时间=',end-start)

    # print(distance_numpy[:20,:20].shape)
    # points = Hamilton(distance_numpy[:20,:20])
    # points = Hamilton(distance_numpy )

    # print(points)

    pass


if __name__ == '__main__':
    main()


