import os
import cv2
import numpy as np
# from  miaohui0204 import MiaoHui1116
from  miaohui0204_robotTrail import MiaoHui1116
 
import time
import sys

miaohui = MiaoHui1116()

# if __name__ == '__main__':
 
#     miaohui = MiaoHui1116()
    
#     start = time.time()
#     outputPath ,_= miaohui.trail_generate('./testData/Photo1_output.png', './testData/Photo1_parsing.npy', 
#                                             './testData/Photo1_temp.txt', './testData/Photo1_trail_temp.png', "8ha8azthmj", '000')# error 报错

#     print('整体消耗的时间=',time.time()-start)
#     print(outputPath,time.asctime( time.localtime(time.time()) ))


 




from flask import Flask
from flask import Flask, request
 
server = Flask(__name__)  
 
@server.route("/trail_generate/", methods=['POST'])
def trail_generate():



    # sketch_path = request.form['sketch_path']
    # parsing_path = request.form['parsing_path']
    # trail_txt_path = request.form['trail_txt_path']
    # trail_png_path = request.form['trail_png_path']
    sketch_path = './testData/Photo1_output.png'
    parsing_path = './testData/Photo1_parsing.npy'
    trail_txt_path = './testData/Photo1_temp.txt'
    trail_png_path = './testData/Photo1_trail_temp.png'
    


    outputPath ,_= miaohui.trail_generate(sketch_path, parsing_path, trail_txt_path, trail_png_path, "8ha8azthmj", '000')# error 报错
 

    print(outputPath,time.asctime( time.localtime(time.time()) ))
     
    return  outputPath 
 
if __name__ == "__main__":
    server.run(host='0.0.0.0',port=8088)








