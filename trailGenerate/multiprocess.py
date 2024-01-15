import multiprocessing


def multiplication1():
    return 1 

def multiplication2():
    return 10 


class Test(object):
    def __init__(self,):

        pass

    # 设置回调函数
    def setcallback(self,x):
        with open('result.txt', 'a+') as f:
            line = str(x) + "\n"
            f.write(line)


    def multiplication1(self,):
        return 1 

    def multiplication2(self,):
        return 10 
  
    def main(self,):
        pool = multiprocessing.Pool(16)
     
        pool.apply_async(func=self.multiplication1, callback=self.setcallback)
        pool.apply_async(func=self.multiplication2, callback=self.setcallback)
        pool.close()
        pool.join()

if __name__ == '__main__':
    # pool = multiprocessing.Pool(16)
     
    # pool.apply_async(func=multiplication1, callback=setcallback)
    # pool.apply_async(func=multiplication2, callback=setcallback)
    # pool.close()
    # pool.join()
 
    test = Test()
    test.main()
