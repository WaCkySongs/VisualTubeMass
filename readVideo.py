# DEV: by W.C. wangche0731@outlook.com
# TODO: 
# set mask of tubes and frame0 with polygon
# set bound: by 'a,b,c,d' zone, nums '1,2,3,4'  mouse points of polygon: cv2.grabcut(https://blog.csdn.net/qq_30815237/article/details/86763443)
# read

import cv2
from cv2 import WINDOW_AUTOSIZE
import numpy as np
import matplotlib.pyplot as plt

# ROI substract
class points(object):
    def __init__(self,id) -> None:
        self.id = id
        self.num = 0
        self.xy = np.empty((0,2))
        self.lines = []
        self.points = []
        self.mask = []
    def add(self,x,y):
        self.num = self.num + 1
        self.xy = np.append(self.xy,[[x,y]],axis=0)
    def pop(self):
        if self.num > 0:
            self.num = self.num - 1
            self.xy = np.delete(self.xy,self.num,0)
        else:
            print('All bound points have been popped')
    def draw(self,ax):
        if self.lines:
            self.lines[0].remove()
        if self.num == 1:
            x = self.xy[0,0]
            y = self.xy[0,1]
            self.lines = ax.plot([x,x],[0,1000], color= 'r')
        elif self.num > 1:
            self.lines = ax.plot(self.xy[:,0],self.xy[:,1], color = 'r')


def onKeyPress(event):
    global iBound,numBound,bounds
    if event.key in '0123456789': # remove all pts
        if iBound < numBound:
            iBound = int(event.key)
            print('set bound ',iBound,' Num_current',bounds[iBound].num)
        else:
            print('out of max bound')
    if event.key == 'c':
        tf = True
        if bounds[0].num != 1:
                print('bound 0',' num = ',bounds[0].num)
                tf = False
        for i in range(1,numBound):
            if bounds[i].num != 4:
                print('bound',i,' num = ',bounds[i].num)
                tf = False
                break
        if tf:
            plt.close()


def onButtonPress(event,bounds,iBound,frame0):
    x = event.xdata
    y = event.ydata
    b = bounds[iBound]
    if event.button == 1: # left
        print('add ',b.num+1,'st point of bound ',iBound,'at',x,y)
        b.add(x,y)
    elif event.button == 3:  # right
        print('pop ',b.num,'st point of bound ',iBound)
        b.pop()
    elif event.button ==2: # middle complete
        b.mask = np.zeros(frame0.shape, np.uint8) 
        b.points = [np.array(b.xy,dtype=np.int32)]
        if b.num == 1:
            b.mask[:,:int(b.xy[0,0]),:] = 255
        elif b.num >1:
            b.mask = cv2.fillPoly(b.mask,b.points,(255,255,255))
        ROI = cv2.bitwise_and(b.mask,frame0)
        cv2.imshow('ROI',ROI)
        cv2.waitKey(10)
    b.draw(ax)
    fig.canvas.draw()


if __name__ == '__main__':

    # test

    # read video
    folderPath = 'E:\\ba高速摄影仪录像\\2018.06\\2018.07.05\\'
    videoPath = '2018.07.05.1917 21冷启动.mov'
    cap = cv2.VideoCapture(folderPath+videoPath)
    print('Video opened? ', cap.isOpened())

    # set boundary
    print('set boundaries for frame&tubes')
    iBound = 0
    numBound = 5
    bounds = []
    for i in range(numBound):
        bounds.append(points(i))  # include the boundFrame

    if cap.isOpened():
        rev, frame0 = cap.read()
        fig = plt.figure()
        fig.canvas.mpl_connect("key_press_event", lambda event:onKeyPress(event))
        fig.canvas.mpl_connect('button_press_event',lambda event:onButtonPress(event,bounds,iBound,frame0))
        ax = fig.add_subplot(111)
        
        ax.imshow(frame0,cmap='gray',interpolation='bicubic')
        plt.show()
    print('finish img set')
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        k = cv2.waitKey(int(1000/60))
        if (k&0xff == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
