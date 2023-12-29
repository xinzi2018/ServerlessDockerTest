import cv2
import numpy as np
import os


class Z_scan(object):
    def __init__(self,):
        pass

    # def Skeletonization(self,img,Blur = True):
    #     size = np.size(img)
    #     skel = np.zeros(img.shape,np.uint8)
        
    #     ret,img = cv2.threshold(img,127,255,0)
    #     element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #     done = False
        
    #     img =cv2.bitwise_not(img)
    #     while( not done):
    #         eroded = cv2.erode(img,element)
    #         temp = cv2.dilate(eroded,element)
    #         temp = cv2.subtract(img,temp)
    #         skel = cv2.bitwise_or(skel,temp)
    #         img = eroded.copy()
        
    #         zeros = size - cv2.countNonZero(img)
    #         if zeros==size:
    #             done = True
    #     ret,img2 = cv2.threshold(cv2.GaussianBlur(skel,(3,3),0),1,255,0)
    #     return cv2.erode(img2,element,iterations = 1)


    def ZLine(self,img_cv):
        #
        # img_cv = self.Skeletonization(img_cv)
        # cv2.imwrite('1.jpg',img_cv)

        # 二值处理以防万一
#        img_cv = img_cv[:,:,0]
        img_cv = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img_cv, connectivity=8, ltype=None)
#        print('label.shape:',labels.shape)
#        print(stats)
        '''
        image：也就是输入图像，必须是二值图，即8位单通道图像。（因此输入图像必须先进行二值化处理才能被这个函数接受）
        connectivity：可选值为4或8，也就是使用4连通还是8连通。
        ltype：输出图像标记的类型，目前支持CV_32S 和 CV_16U。 返回值：
        返回值：
        num_labels：所有连通域的数目
        labels：图像上每一像素的标记，用数字1、2、3…表示（不同的数字表示不同的连通域）
        stats：每一个标记的统计信息，是一个5列的矩阵，每一行对应每个连通区域的外接矩形的x、y、width、height和面积，示例如下： 0 0 720 720 291805
        centroids：连通域的中心点
        '''
#        print('连通区域数量',num_labels)


        # # 不同的连通域赋予不同的颜色
        vis = self.VisConnectedArea(img_cv, num_labels, labels)
        cv2.imwrite('vis_t.png',vis)# 最好保存成png. 不然会有颜色的过渡


        # 得到每一个连通区域的起点：
        FirstPoints = self.getFirstPoint(num_labels,labels)# 每个连通区域最左上角的点
#        print(FirstPoints)

        totalPoints = []
        # 因为FirstPoints里面存储了每个连通区域最左上角的点，所以可以直接从该点开始遍历，避免每次都从（0，0）开始
        for firstpointKey  in  FirstPoints.keys():
            PointsConnected = []# 用来存储当前连通区域所有的坐标点
            stack = []# 用来存储各个分支的起始点 # list 有栈的特性 后进先出，（append进，pop出）

            firstpoint = FirstPoints[firstpointKey]# 当前连通区域最左上角的点坐标，从该点切入去找路径
            
            # for  i in range(stats[firstpointKey][0], stats[firstpointKey][2]):# 此时的遍历不要基于整张图片，因为connectedComponentsWithStats中返回的stats
            #     for j in range(stats[firstpointKey][1], stats[firstpointKey][3])


            # 设定规则：Z字形每次的笔画横着走（暂不考虑竖着的Z（类似N）的状态）
            # 由于Z字形设定规则的限制，所以矩阵中“行”的迭代方向为“往上或往下”， 默认directionH=1（上：0，下：1）
            # 设定Z字形的横笔画的方向从左到右，所以默认设置directionW=1(暂时先不考虑从右到左)
            directionW, directionH = 1, 1
            currentPointW,currentPointH = firstpoint[0],firstpoint[1]

            stack.append([firstpoint,(directionW, directionH)]) # [(,),(,)]
            i=0
            while len(stack)!=0:
                cv2.imwrite('{}.jpg'.format(i),labels*255)
                i = i+1
                stack_pop = stack.pop()# 当前这个起点 出栈
#                print('出栈：',stack_pop)
                firstpoint=stack_pop[0]
#                print('第一行的第一个坐标点为：',firstpoint)
#                print('stats[firstpointKey]:',stats[firstpointKey])

                YiBiPoints=[]
                begin_line = firstpoint[1]
                for line_num in range(begin_line, stats[firstpointKey][1]+stats[firstpointKey][3]): #遍历行
#                    print('===========当前检索到第几行：',line_num)
                    labels,linePoints, stack,isContinueNextLine = self.getCurrentLinePointsAndRecordTree(labels,firstpointKey,firstpoint,stack,stats[firstpointKey])# 获取当前行的线条像素点，同时需要记录分支的位置，分支位置需要入栈
#                    print('isContinueNextLine=',isContinueNextLine)
                    

                    # YiBiPoints.append(linePoints)
                    if (line_num-begin_line)%5==0 and isContinueNextLine!=0:# 每隔三行存储
                        YiBiPoints = YiBiPoints+linePoints
                    elif isContinueNextLine==0:
                        YiBiPoints = YiBiPoints+linePoints
#                    print('每行点的数量',len(linePoints))


                    if isContinueNextLine==0:break
                    else:
                        temp = labels[line_num+1,:]
                        # print(temp.shape)
                        index = np.where(temp==firstpointKey)# mask[1][0]w,mask[0][0]h
                        # print(index)
                        firstpoint=(index[0][0],line_num+1)

#                        print('下一行的第一个坐标点的位置w,h：',firstpoint)

                YiBiPoints[-1][-1]=33# 设置结尾点
                    
                # PointsConnected.append(linePoints)# 表示重点，需要抬笔
                

                PointsConnected = PointsConnected+YiBiPoints

            totalPoints = totalPoints+PointsConnected

        

    #    # 吧关键点可视化下：
    #    f = open('/home/zhujingjie/projects/dailingna/dln_project/test.txt', 'w')
    #    draw = np.zeros((img_cv.shape[0],img_cv.shape[1],3))
    #    print(draw.shape)
    #    print(totalPoints)
    #    print(len(totalPoints))
    #
    #    for i in range(len(totalPoints)-1):
    #        f.write(str(totalPoints[i][0]) +' '+ str(totalPoints[i][1])+' '+str(totalPoints[i][2])+'\r\n')
    #        cv2.line(draw, tuple(totalPoints[i][:2]),tuple(totalPoints[i+1][:2]),(0,255,0),1)#绿色，1个像素宽度
    #    f.write(str(totalPoints[i][0]) +' '+ str(totalPoints[i][1])+str(totalPoints[i][2])+'\r\n')
    #
    #    f.close()
    #    cv2.imwrite('1111.png',draw)
#        print(totalPoints)
        return totalPoints
 
    def getCurrentLinePointsAndRecordTree(self, labels,firstpointKey,firstpoint,stack,currentStats):
        # [firstpoint,(directionW, directionH)] = stack_pop
        points = []
        flag=-1
        # print('检索行的最左边/最右边的坐标：',firstpoint[0], currentStats[0]+currentStats[2])
        for i in range(firstpoint[0], currentStats[0]+currentStats[2]):# 方向仔细琢磨（先不考虑笔的方向）
            if i == firstpoint[0]:# 如果是线段的开端，则
                zAxis = -33
            else:
                zAxis = 0
            if labels[firstpoint[1],i] == firstpointKey:

                points.append([i, firstpoint[1],zAxis])  # 记录当前行的点
#                if i == firstpoint[0]:
#                    print('添加的坐标点=',[i, firstpoint[1]])
                    # print('添加的坐标点=',[i, firstpoint[1]])
                labels[firstpoint[1],i] = 0  # 赋值为-1 表示该点被访问过
                # stack, flag = self.checkLeafPoint([i, firstpoint[1]], labels, stack, directionH,)
                stack, flag = self.checkLeafPoint([i, firstpoint[1]], labels, stack, direction=1,flag=flag)
                # print('flag=',flag)

            else:
                break
                # # 表示该行有两个线段
                # return points ,stack

        # 如果最终的flag==0,表示下一行没有线段了，可以stack 出栈开始新的起点
        # 如果最终的flag==1或者2，表示下一行没有分支
        # 如果最终的flag==3,表示下一行有分支一个断点（两条线段（多条没考虑））
        return labels,points ,stack, flag
         

    def checkLeafPoint(self,currentpoint,labels, stack, direction=1,flag=1):
        H,W = labels.shape[:2]
        (currW,currH) = currentpoint
        # 如果当前方向directionH == 1 往下，则需要去查找上面一行（被检索过的行）有没有从-1变0又变1 # 暂不考虑
        # 同时要去查找下面一行（未被检索过的行）有没有从1变0变1 如果有则stack 入栈
        if direction==1:
            if currH<H: # 如果当前行是最后一行，就不去下一行去找入栈的点了
                stack, flag = self.checkAfterLine((currW,currH+1), labels, stack, direction,flag)
        return stack, flag



        ###### 暂不考虑
        # # 如果当前方向directionH==0 往上，则需要去查找上面一行（被检索过的行）有没有从-1变0又变1
        # # 同时要去查找下面一行（未被检索过的行）有没有从1变0变1，如果有则stack 入栈
        # elif direction==0:
        #     pass


    def checkAfterLine(self, nextLinePoint, labels, stack, directionH, flag=-1): # 判断nextLinePoint这个点是否符合入栈队则
        # if labels[nextLinePoint[0],nextLinePoint[1]]==1 and flag == 0: # # 表示nextLinePoint这个点是从1-0过度过来的，是说明有分支，是另一个线的起点，需要入栈
        #     stack.append([nextLinePoint,(1,1)])
        #     flag =-1# 表示分支起点被记录 后续的同行的就不用入栈了
        # print('监测点：',labels[nextLinePoint[1], nextLinePoint[0]],flag)
        if labels[nextLinePoint[1], nextLinePoint[0]]==0 and flag==-1:# 第一个点
            flag = 0

        elif  labels[nextLinePoint[1],nextLinePoint[0]]==1 and flag==-1:
            flag = 1

        if flag == 0 and labels[nextLinePoint[1], nextLinePoint[0]]==1:
            flag = 4

        
        if flag == 1 and labels[nextLinePoint[1], nextLinePoint[0]]==0:
            flag=2
        # elif labels[nextLinePoint[0],nextLinePoint[1]]==1 and flag ==0:

        if flag == 2 and labels[nextLinePoint[1], nextLinePoint[0]]==1:
            flag = 3 #表示分支起点被记录 后续的同行的就不用入栈了 # 重点是这个赋值
            stack.append([nextLinePoint,(1,1)])

        return stack, flag



    # def checkBeforeLine(self,):# # 表示去找当前点的上一行 # 暂不考虑


    def VisConnectedArea(self,img_cv, num_labels, labels):#不同的连通域赋予不同的颜色
        output = np.zeros((img_cv.shape[0], img_cv.shape[1], 3), np.uint8)
        for i in range(1, num_labels):

            mask = labels == i
            # print(np.where(mask==1))
            # print(np.unique(mask.astype(np.uint8)))
            output[:, :, 0][mask] = np.random.randint(0, 255)
            output[:, :, 1][mask] = np.random.randint(0, 255)
            output[:, :, 2][mask] = np.random.randint(0, 255)
        # cv2.imwrite('vis.png',output)

        return output





    def getFirstPoint(self,num_labels,labels):
        FirstPoints = {}
        for i in range(1,num_labels):
            mask = np.where(labels==i)
            print(mask)

            FirstPoints[i]=(mask[1][0],mask[0][0])#mask[0]:H,mask[1]:W

#        print('FirstPoints=',FirstPoints)

        return FirstPoints



def main():
    z = Z_scan()
    # img_path = '/home/zhujingjie/projects/dailingna/dln_project/1631072737307.png'
    # img_path = '/home/zhujingjie/projects/dailingna/dln_project/mouth.png'
    img_path = '/home/zhujingjie/projects/dailingna/dln_project/meimao.png'
    img_cv = cv2.imread(img_path)
    z.ZLine(img_cv)



if __name__ == '__main__':
    main()









