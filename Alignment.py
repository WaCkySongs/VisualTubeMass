# TODO: mask the tube (https://www.lmlphp.com/user/151116/article/item/7779659/)(https://stackoverflow.com/questions/42346761/opencv-python-feature-detection-how-to-provide-a-mask-sift)

# ref
# [Python进行SIFT图像对准](https://www.jianshu.com/p/f1b97dacc501?tdsourcetag=s_pctim_aiomsg)

import numpy as np
import cv2

def diffImg(im1, im2, Method = 1, ifwrite = False):
    im1G = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2G = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    # alignment method
    if Method == 0 :        alignor = cv2.ORB_create(nfeatures= 5000)
    else:        alignor = cv2.SIFT_create() 
    kp1,des1 = alignor.detectAndCompute(im1G,None)
    kp2,des2 = alignor.detectAndCompute(im2G,None)

    if Method == 0: # Brute-Force Matcher for ORB
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

    # matchesMask = [[0,0] for i in range(len(matches))]

    # for i, (m,n) in enumerate(matches):
    #     if m.distance < 0.4*n.distance:
    #         matchesMask[i] = [1,0]
    if ifwrite:
        imMatches = cv2.drawMatches(im1G,kp1,im2G,kp2,matches,None)
        if ifwrite: cv2.imwrite('matches.jpg',imMatches)

    # Extract location of good matches
    pts1 = np.zeros((len(matches),2,),dtype=np.float32)
    pts2 = np.zeros((len(matches),2,),dtype=np.float32)

    for i,match in enumerate(matches):
        pts1[i,:] = kp1[match.queryIdx].pt
        pts2[i,:] = kp2[match.trainIdx].pt

    # homography
    h, mask = cv2.findHomography(pts2,pts1,cv2.RANSAC,ransacReprojThreshold=4)

    a,b,c = im1.shape
    im2GReg = cv2.warpPerspective(im2G,h,(b,a))
    im2GDiff = cv2.absdiff(im2GReg,im1G)
    if ifwrite: cv2.imwrite('im2GReg.jpg',im2GReg)
    if ifwrite: cv2.imwrite('im2GDiff.jpg',im2GDiff)

    return im2GReg, im2GDiff,sum(sum(im2GDiff))/im2GDiff.size

if  __name__=='__main__':
    im1 = cv2.imread('D0s.jpg')
    im2 = cv2.imread('D01s.jpg')

    _,im2GDiff,aveDiff = diffImg(im1,im2,Method=1,ifwrite=True)



    print('Finished')



# import numpy as np
# import cv2 


# ## ECC method RES: effect not good
# im1 = cv2.imread("D0s.jpg")
# im2 = cv2.imread("D01s.jpg")
# im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
# im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)


# sz = im1.shape

# warp_mode = cv2.MOTION_AFFINE
# warp_matrix = np.eye(2,3,dtype=np.float32)

# n_iterations = 50000
# n_eps = 1e-10
# criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iterations, n_eps)

# # ECC transform matrix
# (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

# im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

# im3=im1-im2_aligned

# cv2.imwrite("D0s-0.jpg",im3)




# end = 1