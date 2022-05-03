""" read the video and PostProcess

flowchart for a video
set boundary -> align -> diff to frame0 -> record statistics -> save data 

DEV: by W.C. wangche0731@outlook.com

# TODO: 
# faster *1.5 cost than real time at dt 1s

# feat.
# set mask of tubes and frame0 with polygon
# set bound: by 'a,b,c,d' zone, nums '1,2,3,4'  mouse points of polygon: 
#   cv2.fillpoly cv2.bitwise_XX cv2.grabcut(https://blog.csdn.net/qq_30815237/article/details/86763443)
#   
# read


# ref
[Python进行SIFT图像对准](https://www.jianshu.com/p/f1b97dacc501?tdsourcetag=s_pctim_aiomsg)
mask the tube & frame in SIFT (https://www.lmlphp.com/user/151116/article/item/7779659/)(https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift)

"""

import time
import cv2
from matplotlib.cbook import ls_mapper
from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
import os

# ROI substract
class ROI(object):
    r""" A region of interest object

    Create a ROI include all ROIs and its postprocess flow

    Parameters
    ----------

    id : int
        id
    a,b : int
        height, width of frame/frame0 
    num: int
        num of boundary points
    xy: np.array num*2
        coord of boundary points
    lines: plt.lines
        lines of boundary with matplotlib

    mask: np.array = 255
        mask of the ROI, valid in diff, invalid in align
    # ambient: np.array = 0 discard
    #     ambient region, diff invalid, valid in align

    """

    def __init__(self,id,height,width) -> None:
        self.id = id
        self.height ,self.width = height,width 
        self.num = 0
        self.xy = np.empty((0,2))
        self.lines = []
        self.mask = np.zeros((height,width),dtype= np.uint8)

    def add(self,x,y):
        ' add boundary point '
        self.num = self.num + 1
        self.xy = np.append(self.xy,[[x,y]],axis=0)
    
    def pop(self):
        ' pop boundary point '
        if self.num > 0:
            self.num = self.num - 1
            self.xy = np.delete(self.xy,self.num,0)
        else:
            print('All bound points have been popped')
    
    def drawLines(self,ax):
        ' draw the boundary of ROI in axis '
        if self.lines:
            self.lines[0].remove()
            self.lines = []
        if self.num == 1:
            x = self.xy[0,0]
            self.lines = ax.plot([x,x],[0,self.height*0.99], color= 'r')
        elif self.num > 1:
            self.lines = ax.plot(self.xy[:,0],self.xy[:,1], color = 'r')
    
    def getMask(self):
        self.mask[:,:] = 0
        points = [np.array(self.xy,dtype=np.int32)]
        if self.num == 1: # frame bound : only include panel
            self.mask[:,int(self.xy[0,0]):] = 255
        elif self.num >= 1:
            self.mask = cv2.fillPoly(self.mask,points,(255))
        return self.mask

class postTubes(object):
    r''' ROIs object whole image

    
    Parameters:
    -------

    frame0: np.array in black/white
        reference frame, to be align/diff
    hw: 2* int
        height, width of frame0
    numROI/numBound: int
        number of ROI sub
    iBound: int
        current ROIs to be set
    bounds: ROI
        detail of ROI
    method,alignor,des0,kp0: int,...etc.
        0: Not align 1: ORB 2: SIFT(slow) 
        how to align  frame
    
    '''
    def __init__(self,fp,fn,fm='.mov',nROI = 0,method = 2) -> None:
        self.fp = fp
        self.fn = fn
        fpn = fp+fn + fm
        if not(os.path.isfile(fpn)): 
            print('wrong file path') 
            return
        self.cap = cv2.VideoCapture(fpn)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # set boundary
        ret, frame0 = self.cap.read()
        self.frame0 = frame0[:,:,0]
        print('set boundaries for frame&tubes')        

        self.hw = self.frame0.shape[0:2] # heigh width

        # read or set ?
        self.iBound = 0
        if os.path.isfile(fpn+'.txt'):
            self.readROIPoints(fpn+'.txt')
        elif os.path.isfile('test.txt'):
            self.readROIPoints('test.txt')
        elif nROI>1:
            self.setnumBound(nROI)
        else:
            print('ROI set wrong, exit')
            return

        self.ambient = np.zeros((self.hw[0],self.hw[1]),dtype=np.uint8) # Region exclude bounds  
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.setiROIBound)
        self.fig.canvas.mpl_connect('button_press_event',self.setROIPoints)
        self.ax = self.fig.add_subplot(111)
        self.drawFrame()
        self.drawLines()
        plt.ioff()
        plt.show()
        # close the plt
        self.saveROIPoints(fpn+'.txt')

        # alignor set
        self.method = method # 0 not align 1 ORB or 2 SIFT
        self.alignor = []
        self.des0 = []
        self.kp0 = []
        self.ifwrite = False
        
        self.detectFrame0()
    def process(self,dTime):
        self.dTime = dTime
        cap,fps = self.cap, self.fps
        Time = 0.0 # [s]
        capLen = int( cap.get(7)/cap.get(5)/float(dTime) ) 
        self.Lmean = np.zeros((capLen,self.numBound+2)) # Res. Mean Lumination 

        tic = time.time()
        self.Diffs = []
        resVideo = cv2.VideoWriter('Res.avi',cv2.VideoWriter_fourcc(*'XVID'),float(1),(analysis.hw),isColor = False) # greyscale
        ret = True

        while(ret):
            Time = Time + dTime
            iTime = int(Time/dTime)-1
            cap.set(1,int(Time*fps))
            ret, frame = cap.read()
            try:
                self.Lmean[iTime,0] = Time
                # statistics
                frameG = frame[:,:,0]
                self.alignFrame(frameG)
                self.Diffs.append( self.diffFrame() )
                self.Lmean[iTime,1:] = self.statisticROI()

                resVideo.write(self.diff)
                cv2.imshow('ROI',self.diff)
                cv2.waitKey(1)
                print(self.Lmean[iTime,:])

            except:
                print('align diff fails') # res = 0
                pass
            # cv2.imshow('frame',frame)
        toc = time.time()
        print('cost',toc - tic)
        np.savetxt('Lmean.csv',self.Lmean,delimiter=',')
        
        resVideo.release()
        cv2.destroyAllWindows()
        # self.cap.release()
    
    def drawFrame(self,frame = None):
        if frame is None: frame = self.frame0
        self.ax.cla()
        self.ax.imshow(frame,cmap='gray',interpolation='bicubic')
        self.fig.canvas.draw()     
    def drawLines(self):
        for b in self.bounds:
            b.drawLines(self.ax)
        self.fig.canvas.draw()     

    def setnumBound(self,n):
        self.numBound = n # numb of ROI
        self.bounds = []
        for i in range(n+1):
            self.bounds.append(ROI(i,self.hw[0],self.hw[1]))

    # bonded Key and Button in fig
    def setiROIBound(self,event):        
        if event.key in '0123456789': # set bounds[key]
            self.iBound = int(event.key)
            if self.iBound <= self.numBound:
                self.drawFrame()
                self.drawLines()
                print('set bound ',self.iBound,' Num_current',self.bounds[self.iBound].num)
            else:
                print('out of max bound')
        elif event.key in 'c': #check
            self.ax.cla()
            img = np.zeros((self.hw[0],self.hw[1]))
            for b in self.bounds:
                img += cv2.bitwise_and(b.mask,self.frame0)
            img += cv2.bitwise_and(self.ambient,self.frame0)//5
            self.drawFrame(img)
        elif event.key in '-': # exit
            pass
    def setROIPoints(self,event):
        iBound = self.iBound
        a = self.bounds[0] # pointer like
        b = self.bounds[self.iBound]
        x,y = event.xdata, event.ydata
        if event.button == 1: # left
            print('add ',b.num+1,'st point of bound ',iBound,'at',x,y)
            b.add(x,y)
        elif event.button == 3:  # right
            print('pop ',b.num,'st point of bound ',iBound)
            b.pop()
        elif event.button ==2: # middle complete
            b.getMask()
            self.getAmbient()
            if iBound == 0: 
                b.mask = cv2.bitwise_not(b.mask)
                self.ambient = cv2.bitwise_not(self.ambient)
            Tube = cv2.bitwise_and(b.mask,self.frame0)
            Ambient = cv2.bitwise_and(self.ambient,self.frame0)//5 # darker ambient
            self.drawFrame(Tube+Ambient)
            cv2.waitKey(10)
            return
        b.drawLines(self.ax)
        self.fig.canvas.draw()   
    
    def saveROIPoints(self,fp='test.txt'):
        set = np.zeros((0,2))
        set = np.append(set,[[self.numBound,0]],axis=0)
        for b in self.bounds:
            set = np.append(set,[[b.num,0]],axis=0)
            set = np.append(set,b.xy,axis=0)
        np.savetxt(fp,set)        
    def readROIPoints(self,fp='test.txt'):
        set = np.loadtxt(fp)
        i = 0
        self.numBound = int(set[i,0])
        self.bounds = []
        for iBound in range(self.numBound + 1):
            self.bounds.append(ROI(iBound,self.hw[0],self.hw[1]))
            b = self.bounds[iBound]
            i = i + 1
            num = int(set[i,0])
            for xy in set[i+1:i+1+num,:]:
                i = i + 1
                b.add(xy[0],xy[1])
        return
    def getAmbient(self):
        ambient = self.bounds[0].mask
        for b in self.bounds[1:]:
            ambient = cv2.bitwise_or(b.mask,ambient) # reduce mask region
        self.ambient = cv2.bitwise_not(ambient)
        return self.ambient

    def detectFrame0(self): 
        if self.method ==0 : return
        elif self.method == 1:  self.alignor = cv2.ORB_create(nfeatures= 5000)
        elif self.method == 2:  self.alignor = cv2.SIFT_create() 
        self.kp0,self.des0 = self.alignor.detectAndCompute(self.frame0,self.ambient)

    def alignFrame(self,im2G, ifmask = 'ambient'): # by mask ambient
        if self.method == 0: # not align
            self.frame = im2G
            return
        
        # alignment
        im1G = self.frame0
        kp1 = self.kp0
        des1 = self.des0

        mask = self.ambient if ifmask == 'ambient' else ifmask
        
        kp2,des2 = self.alignor.detectAndCompute(im2G,mask)
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
        h, maskh = cv2.findHomography(pts2,pts1,cv2.RANSAC,ransacReprojThreshold=4)

        a,b = im1G.shape
        im2GReg = cv2.warpPerspective(im2G,h,(b,a))
        if self.ifwrite: cv2.imwrite('im2GReg.jpg',im2GReg)
        if self.ifwrite: im2GDiff = cv2.absdiff(im2GReg,im1G)
        if self.ifwrite: cv2.imwrite('im2GDiff.jpg',im2GDiff)

        self.frame = im2GReg

    def diffFrame(self):
        self.diff = cv2.absdiff(self.frame,self.frame0)
        return self.diff
    
    def statisticROI(self):
        mAve = np.zeros((1,self.numBound+1))
        ambient = self.ambient//225 
        for i,b in enumerate(self.bounds[1:]):
            mask = b.mask//255
            maskDiff = mask * self.diff 
            mAve[0,i] = maskDiff.sum()/mask.sum()
        ambientDiff = ambient * self.diff
        mAve[0,i+1] = ambientDiff.sum()/ambient.sum()
        return mAve
    
    def plotStat(self):
        plt.ion()
        ax = plt.subplot(111)
        t = self.Lmean[:,0]
        l = self.Lmean[:,1:self.numBound+1]
        ab = self.Lmean[:,self.numBound + 1]
        LumTubes = ax.plot(t,l)
        LumAmb = ax.plot(t,ab,'--')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Lumination [- 255]')
        ax.set_title(self.fn,fontproperties='SimHei')
        label = [str(i) for i in range(1,self.numBound+1)]
        ax.legend(LumTubes,label)
        plt.savefig(self.fn+'.png')
        plt.ioff()
        plt.show()
        return ax

if __name__ == '__main__':
    # read video
    fp = 'E:\\ba高速摄影仪录像\\2018.06\\2018.07.05\\'
    fn = '2018.07.05.1058 7冷启动全过程-测试模式'
    dTime = 0.5 # [s] 0.5s for 2*real time
    iframe0 = 0 # TODO: DOING:
    analysis = postTubes(fp,fn,nROI= 4)

    analysis.process(dTime)

    ax = analysis.plotStat()

    1