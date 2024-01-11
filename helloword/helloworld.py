


# import numpy as np
# import os
# import cv2

# def main():
#     print('====myhelloworld=====')
# if __name__ == '__main__':
#     main()


from flask import Flask
from flask import Flask, request
 
server = Flask(__name__)
 
@server.route("/", methods=['POST'])
def hello():
    # img_path = request.form['img_path']

     
    # return "Hello World!"+img_path
    return "Hello World!" 
 
if __name__ == "__main__":
    server.run(host='0.0.0.0',port=9000)
 