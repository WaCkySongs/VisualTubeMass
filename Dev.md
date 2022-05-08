# Visual process the oil/ref mass flow rate in transparent tube

## TODO: further

## Test source

[2018.07.05.191721冷启动.mov](E:\ba高速摄影仪录像\2018.06\2018.07.05\2018.07.05.1917 21冷启动.mov)

## Tech. route

Manually <-> Code

1. Crop, Stablizer
   1. [Feature Based Alignment/Registration ORB](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
2. Segmentation & Enhancement
3. Lumination diff and statistics
4. Relationship to $\dot M$

## Developments

1. Ps [manually Diff](./Dev/PsPreTest/D0s-800s.png)
2. Alignment methods
   1. Get features [web](https://blog.csdn.net/qq_35497808/article/details/74999344)
      1. Pixel based [ECC not good](./Dev/ECC_D10s-0.jpg)
      2. features [mask the TRASPARENT tube at first!](https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift)
         1. [ORB ok](./Dev/ORB_im2GDiff.jpg)
         2. [SIFT ok,slighterly better than ORB](./de/../Dev/SIFT_im2GDiff.jpg)
         3. [SURF been removed]a
   2. Match
      1. Brutal Force: BFMatcher for ORB
      2. FLANN
   3. Alignment NOTE: whole [Better] or part
      1. not very good in whole img, maybe seperately (not good especially ambient not wide enough) ? faster ,but SG2 lost accurate
      2. partly: sightglass2 misalignment during frosting some frames, maybe postprocess
      3. not alignment: [bad] ambientDiff twice larger.
3. faster by save im1G, pts1, des1 ...
4. pandas to existed excel with [openpyxl](https://stackoverflow.com/questions/61933021/how-to-overwrite-data-on-an-existing-excel-sheet-while-preserving-all-other-shee)
5. filter: [mean53 or minInN better](./Dev/FilterTest.png)

## FUTHER DEVELOPMENT

1. classification
   1. 
   2. mean,std,cov,Ra,Rq,R ...
   3. KNN, ResNet ... + Kmeans
2. -15 frost image enhancement
   1. light
   2. [contrast](https://blog.csdn.net/weixin_39543758/article/details/110560791 and https://blog.csdn.net/neil_skype/article/details/122609206)
   3. [by ]
3. Cluster Classification Flow Pattern
4. Process
   1. Hough变换提取几何区域
   2. 图像增强
   3. 分割与特征提取
      1. 腐蚀与膨胀和开闭运算
   4. PostProcess Length, Area, etc.
