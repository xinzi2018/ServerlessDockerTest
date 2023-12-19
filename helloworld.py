


# import numpy as np
# import os
# import cv2

# def main():
#     print('====myhelloworld=====')
# if __name__ == '__main__':
#     main()


from flask import Flask
 
server = Flask(__name__)
 
@server.route("/")
def hello():
  return "Hello World!"
 
if __name__ == "__main__":
  server.run(host='0.0.0.0')
 