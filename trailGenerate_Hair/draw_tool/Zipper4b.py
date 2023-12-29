# -*- coding: utf-8 -*-
import struct

class Zipper4b():
    '''
        func: data check,
                check whether the string type data of a picture has a right format.
    '''
    def raiseStrFormatError(self,string):  #4
        if string[0] != 'B':# or string[-1] != 'D':
            raise Exception("String type data must begin with 'B' and end with 'D', But receive an wrong format.")
##        if (len(string) - 2) % 14 != 0:
#            raise Exception("The length of string type data is not divisible, please check your string data.")
            
    '''
        func: get the string type data of a picture,
                and separate each part of a single track.
    '''
    def getTracksStr(self, data_path):  #3
            with open(data_path, 'r') as f:
                string = f.read()
                f.close()
            self.raiseStrFormatError(string)  # data check
            string = string[1:-1]  # Remove 'B' and 'D'
            part_list = list()  # Parts of track (string type)
            for i in range(0, len(string), 14):
                part_list.append(string[i:i+13])
            return part_list
    '''
        func: generate Track instances list with class 'Track'.
    '''
    def generatefilespath(self, track_list, save_path):
        listfinalpath = []
        filenums = len(track_list)//10032 + 1
        for i in range(filenums):
            print(filenums, i, save_path+"/tra@$"+str(filenums)+ str(i)+".bin")
            listfinalpath.append(save_path+"/tra@$"+str(filenums)+ str(i)+".bin")
            new_track_list = []
            new_track_list.append(0xAA)
            new_track_list.append(0x00)
            new_track_list.append(0x00)
            new_track_list.append(0x00)
                 
            new_track_list += track_list[i*10032:(i+1)*10032]
            
            new_track_list.append(0xA5)
            new_track_list.append(0xFF)
            new_track_list.append(0xFF)
            new_track_list.append(0xFF)
            
            ###???
            numlen = len(new_track_list)//4 ##2000
            num_3 = (numlen & 0x00FF0000) >> 16
            num_2 = (numlen & 0x0000FF00) >> 8
            num_1 = (numlen & 0x000000FF)
            new_track_list[1] = num_3
            new_track_list[2] = num_2
            new_track_list[3] = num_1
            self.saveTo(new_track_list,save_path+"/tra@$"+str(filenums)+ str(i)+".bin")
        return listfinalpath
    def generatepiecelist(self, x_dis, y_dis):
        maxdis = max(x_dis,y_dis)
        dis_nums = []
        if (maxdis <= 16383):
            dis_nums = [[x_dis,y_dis]]
        else:
            temp_disx,temp_disy = x_dis,y_dis
            while(temp_disx>16383 or temp_disy>16383):
                if temp_disx>16383:
                    temp_disx -= 16383
                    temp_disxx = 16383
                else:
                    temp_disxx = temp_disx
                if temp_disy>16383:
                    temp_disy -= 16383
                    temp_disyy = 16383
                else:
                    temp_disyy = temp_disy
                dis_num = [temp_disxx, temp_disyy]
                dis_nums.append(dis_num)
        return dis_nums
        
    def data2u32(self, dir_x, dir_y, z, x, y):
        ret = 0;
        if dir_x:
            ret += (0x01 << 30)
        if dir_y:
            ret += (0x01 << 29)
        if z:
            ret += (0x01 << 28)
        ret += (x << 14)
        ret += y
        u8_4 = (ret & 0xFF000000) >> 24
        u8_3 = (ret & 0x00FF0000) >> 16
        u8_2 = (ret & 0x0000FF00) >> 8
        u8_1 = (ret & 0x000000FF)
        return u8_4, u8_3, u8_2, u8_1
        
    def generateTrackInstancesList(self, data_path, save_path):  #2
          # Parts of track (class type)
        track_list = []
        part_list = self.getTracksStr(data_path)
        temp_z_mode = 0
        for enum_num in range(len(part_list)): #enum是13个长度
            enum = part_list[enum_num]
#            print("12个长度数是", enum)
            ##处理x,y,z的方向
            temp_x,temp_y = 0,0
            if enum[0]=="+":
                temp_x = 1
            if enum[6]=="+":
                temp_y = 1
                
            if enum_num==0:  ##20220425
                temp_z = 1   ##20220425
            elif (enum_num == (len(part_list)-1)):
                temp_z = 1
            elif int(enum[-1])==2: #33抬
                temp_z = 1
            elif int(enum[-1])==0: ##-33落
                temp_z = 0
            elif int(enum[-1])==1: ##0
                temp_z = temp_z_mode
            else:  ####????? mode=3 落笔
                temp_z = 0
            temp_z_mode = temp_z
                    
            ##处理x,y的距离，超16383就分段
            dis_nums = self.generatepiecelist(int(enum[1:6]),int(enum[7:12]))
            ##得到4n个u8的数据格式
            for dis_num in dis_nums:
                u8_4, u8_3, u8_2, u8_1 = self.data2u32(temp_x, temp_y, temp_z, dis_num[0], dis_num[1])  ###
                track_list.append(u8_4)
                track_list.append(u8_3)
                track_list.append(u8_2)
                track_list.append(u8_1)
        ##分包保存
        listfinalpath = self.generatefilespath(track_list,save_path)
        return listfinalpath
        
    '''
        func: save string to a file.
    '''
    def saveTo(self, data, save_path):
#        print(save_path)
        with open(save_path, 'wb') as f:
            for x in data:
                a = struct.pack('B', x)
                f.write(a)
        f.close()


if __name__ == '__main__':
    Zipper4b = Zipper4b()
    f = open("./konghua.txt")
    danpian_f = open("./konghua1.txt","w")
    flag = False
    lines = f.readlines()
    n = lines[0].strip("\n").split(" ")
    newx_0, newy_0 = str(int(float(n[0])*500)).zfill(5), str(int(float(n[1])*500)).zfill(5)
    danpian_f.write("B+"+newx_0+"+"+newy_0+"1")
    for i in range(1,len(lines)-1):
        line_b = lines[i].strip("\n").split(" ")
        line_a = lines[i+1].strip("\n").split(" ")
        x1 = int((float(line_a[0])*500-float(line_b[0])*500))
        if x1>=0:
            x1 = "+" + str(x1).zfill(5)
        else:
            x1 = str(x1).zfill(6)
        x2 = int((float(line_a[1])*500-float(line_b[1])*500))
        if x2>=0:
            x2 = "+" + str(x2).zfill(5)
        else:
            x2 = str(x2).zfill(6)
        
        if i == 1:
            x3 = str(2)
        else:
            x3 = str(int(float(line_b[2]) // 33 + 1))
        if i==1:
            danpian_f.write("/"+x1+x2+x3)
        else:
            danpian_f.write("/"+x1+x2+x3)
    danpian_f.close()
    listfinalpath = Zipper4b.generateTrackInstancesList('./konghua1.txt', "./")   #文件路径和保存路径
    print(listfinalpath)

