# DEV: by W.C. wangche0731@outlook.com
# TODO: 
# faster *1.5 cost than real time at dt 1s

# feat.
# set mask of tubes and frame0 with polygon
# set bound: by 'a,b,c,d' zone, nums '1,2,3,4'  mouse points of polygon: 
#   cv2.fillpoly cv2.bitwise_XX cv2.grabcut(https://blog.csdn.net/qq_30815237/article/details/86763443)
#   
# read

# mask the tube & frame in SIFT (https://www.lmlphp.com/user/151116/article/item/7779659/)(https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift)

# ref
# [Python进行SIFT图像对准](https://www.jianshu.com/p/f1b97dacc501?tdsourcetag=s_pctim_aiomsg)

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ROI substract
class ROI(object):
    def __init__(self,id,frame0,method = 1) -> None:
        self.id = id
        self.num = 0
        self.xy = np.empty((0,2))
        self.lines = []
        self.frame = []
        self.resetMask(frame0)
        # self.wab = [0,frame0.shape[1]]
        # self.frame0 = frame0[:,:,0]
        # self.mask = np.zeros(self.frame0.shape,dtype= np.uint8)
        # self.ambient = np.zeros(self.frame0.shape,dtype= np.uint8)
        self.method = method # ORB or SIFT
        self.alignor = []
        self.des0 = []
        self.kp0 = []
        self.ifwrite = False
    def resetMask(self,frame0):
        self.wab = [0,frame0.shape[1]]
        self.frame0 = frame0
        self.mask = np.zeros(self.frame0.shape,dtype= np.uint8)
        self.ambient = np.zeros(self.frame0.shape,dtype= np.uint8)
    def add(self,x,y):
        self.num = self.num + 1
        self.xy = np.append(self.xy,[[x,y]],axis=0)
    def pop(self):
        if self.num > 0:
            self.num = self.num - 1
            self.xy = np.delete(self.xy,self.num,0)
        else:
            print('All bound points have been popped')
    # def save(self):
    #     np.save
    def draw(self,ax):
        if self.lines:
            self.lines[0].remove()
            self.lines = []
        if self.num == 1:
            x = self.xy[0,0]
            y = self.xy[0,1]
            self.lines = ax.plot([x,x],[0,1000], color= 'r')
        elif self.num > 1:
            self.lines = ax.plot(self.xy[:,0],self.xy[:,1], color = 'r')
    def setWidth(self,a,b):
        self.wab[0] = int(a)
        self.wab[1] = int(b)
    def getWab(self):
        return self.wab[0], self.wab[1]
    def getWidth(self):
        width = self.wab[1] - self.wab[0] +1
        return width
    def getMask(self):
        pts  = np.array(self.xy,dtype=np.int32)
        width = pts[:,0].max()-pts[:,0].min()
        middle = 0.5*( pts[:,0].max()+pts[:,0].min() )
        self.ambient[:,:] = 0
        self.mask[:,:] = 0
        self.points = [np.array(self.xy,dtype=np.int32)]
        if self.num == 1: # frame bound
            wa = self.wab[1]
            wb = self.frame0.shape[1]
            self.setWidth(wa,wb)
            self.mask[:,:int(self.xy[0,0])] = 255
        elif self.num >= 1:
            wa = int( max([self.wab[0], middle-2*width]) )
            wb = int(min([self.wab[1], middle+2*width]))
            self.setWidth(wa,wb)
            self.ambient[:,wa:wb] = 255
            self.mask = cv2.fillPoly(self.mask,self.points,(255))
            self.ambient = cv2.bitwise_xor(self.mask,self.ambient)
            # crop
            self.mask = self.mask[:,wa:wb]
            self.ambient = self.ambient[:,wa:wb]
            self.frame0 = self.frame0[:,wa:wb]
            self.detectFrame0()
        
    def detectFrame0(self):
        if self.method == 0 :        self.alignor = cv2.ORB_create(nfeatures= 5000)
        else:        self.alignor = cv2.SIFT_create() 
        self.kp0,self.des0 = self.alignor.detectAndCompute(self.frame0,self.ambient)
    def alignFrame(self,im2G): # by mask ambient
        # alignment
        im1G = self.frame0
        kp1 = self.kp0
        des1 = self.des0
        
        kp2,des2 = self.alignor.detectAndCompute(im2G,self.ambient)
        if self.method == 0: # Brute-Force Matcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1,des2)
            matches = sorted(matches, key = lambda x:x.distance)
            # 10% good matches
            GoodPercent = 0.1
            num = int(len(matches)* GoodPercent)
            matches = matches[:num]
        else: # flann matcher for SIFT
            indexParams = dict(algorithm = 0, trees = 5)
            searchParams = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(indexParams, searchParams)
            matchesA = flann.knnMatch(des1,des2,k = 2)
            matches = []
            for m,n in matchesA:
                if m.distance < 0.4 * n.distance:
                    matches.append(m)

        if self.ifwrite:
            imMatches = cv2.drawMatches(im1G,kp1,im2G,kp2,matches,None)
            if self.ifwrite: cv2.imwrite('matches.jpg',imMatches)

        # Extract location of good matches
        pts1 = np.zeros((len(matches),2,),dtype=np.float32)
        pts2 = np.zeros((len(matches),2,),dtype=np.float32)

        for i,match in enumerate(matches):
            pts1[i,:] = kp1[match.queryIdx].pt
            pts2[i,:] = kp2[match.trainIdx].pt

        # homography
        h, mask = cv2.findHomography(pts2,pts1,cv2.RANSAC,ransacReprojThreshold=4)

        a,b = im1G.shape
        im2GReg = cv2.warpPerspective(im2G,h,(b,a))
        if self.ifwrite: cv2.imwrite('im2GReg.jpg',im2GReg)
        if self.ifwrite: im2GDiff = cv2.absdiff(im2GReg,im1G)
        if self.ifwrite: cv2.imwrite('im2GDiff.jpg',im2GDiff)

        self.frame = im2GReg

    def diffFrame(self,im2G , ifdiff = True):
        if not ifdiff:
            return im2G
        self.alignFrame(im2G)
        self.diff = cv2.absdiff(self.frame,self.frame0)
        mask = self.mask//255
        diff = mask * self.diff 
        ave = diff.sum()/mask.sum()
        ambient = self.ambient//225 
        diff2 = ambient * self.diff
        ave2 = diff2.sum()/ambient.sum()
        return ave, ave2

class ROIs(object):
    def __init__(self,numROI,frame0,method = 1) -> None:
        self.numBound = numROI # numb of ROI
        self.iBound = 0
        self.bounds = []
        self.hw = frame0.shape[0:2] # heigh width
    
        for i in range(numROI+1):
            self.bounds.append(ROI(i,frame0,method))
        
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.Key)
        self.fig.canvas.mpl_connect('button_press_event',self.Button)
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(frame0,cmap='gray',interpolation='bicubic')
    def getWidth(self):
        width = 0
        for b in self.bounds:
            width = width + b.getWidth()
        return width
    def Key(self,event):        
        if event.key in '0123456789': # set bounds[key]
            if self.iBound <= self.numBound:
                self.iBound = int(event.key)
                print('set bound ',self.iBound,' Num_current',self.bounds[self.iBound].num)
            else:
                print('out of max bound')
    def Button(self,event):
        iBound = self.iBound
        a = self.bounds[0] # pointer like
        b = self.bounds[self.iBound]
        x = event.xdata
        y = event.ydata
        if event.button == 1: # left
            print('add ',b.num+1,'st point of bound ',iBound,'at',x,y)
            b.add(x,y)
        elif event.button == 3:  # right
            print('pop ',b.num,'st point of bound ',iBound)
            b.pop()
        elif event.button ==2: # middle complete
            wa = int(a.xy[0,0]) # frame0 select
            b.resetMask(a.frame0)
            b.setWidth(0,wa) 
            b.getMask()
            Tube = cv2.bitwise_and(b.mask,b.frame0)
            Ambient = cv2.bitwise_and(b.ambient,b.frame0)//5
            cv2.imshow('ROI',Tube+Ambient)
            cv2.waitKey(10)
        b.draw(self.ax)
        self.fig.canvas.draw()

if __name__ == '__main__':
    # read video
    folderPath = 'E:\\ba高速摄影仪录像\\2018.06\\2018.07.27\\'
    videoPath = '2018.07.27.1126 2启动 吸排气.mov'
    cap = cv2.VideoCapture(folderPath+videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    
    # set boundary
    ret, frame0 = cap.read()
    print('set boundaries for frame&tubes')
    numBound = 2
    tube = ROIs(numBound,frame0[:,:,0])
    plt.ioff()
    plt.show()
    
    hwG = (frame0.shape[0],tube.getWidth())

    print('start processing')
    dTime = 1.0
    Time  = 0.0 # s 
    capLen = int( cap.get(7)/cap.get(5)/float(dTime) ) 
    Lmean = np.zeros((capLen,numBound*2+1))
    tic = time.time()
    resVideo = cv2.VideoWriter('Res.avi',cv2.VideoWriter_fourcc(*'XVID'),float(1),(hwG[1],hwG[0]),isColor = False) # greyscale
    resG = np.zeros(hwG,dtype= np.uint8) 
    while(ret):
        Time = Time + dTime
        iTime = int(Time/dTime)
        cap.set(1,int(Time*fps))
        ret, frame = cap.read()

        try:
            # statistics
            we = -1 # starting of resG
            for i,iTube in enumerate(tube.bounds):
                wa,wb = iTube.getWab()
                ws = we + 1
                we = ws + iTube.getWidth() - 1
                if i == 0 :
                    Lmean[iTime,0] = Time 
                    resG[:,ws:we] = frame[:,wa:wb,0]
                else:
                    Lmean[iTime,i],Lmean[iTime,i+numBound] = iTube.diffFrame(frame[:,wa:wb,0])
                    resG[:,ws:we] = iTube.diff
            resVideo.write(resG)
            cv2.imshow('ROI',resG)
            cv2.waitKey(1)
            print(Lmean[iTime,:])
        except:
            print('align diff fails') # res = 0
            pass
        # cv2.imshow('frame',frame)
    toc = time.time()
    print(toc - tic)
    np.savetxt('Lmean.csv',Lmean,delimiter=',')

    cap.release()
    resVideo.release()
    cv2.destroyAllWindows()
