from asyncore import write
import numpy as np
import cv2
import matplotlib.pyplot as plt



def alignImg_kp(im,method=0,maxFeatures=5000):
    imG = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # alignment method
    if method == 0 :
        alignor = cv2.ORB_create(maxFeatures)
    else:
        alignor = cv2.SIFT_create() 
    kp,des = alignor.detectAndCompute(imG,None)
    return imG,kp,des

def alignImg_match(des1,des2, GoodPercent = 0.1):
    if True: # Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
    else: # flann matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)

    num = int(len(matches)* GoodPercent)
    matches = matches[:num]
    return matches

def diffImg(im1,im2,ifwrite = False):
    im1G,kp1,des1 = alignImg_kp(im1)
    im2G,kp2,des2 = alignImg_kp(im2)
    
    matches = alignImg_match(des1,des2)
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
    im2Reg = cv2.warpPerspective(im2,h,(b,a))
    im3 = im2Reg - im1  

    if ifwrite: cv2.imwrite('im2Reg.jpg',im2Reg)

    return im2Reg

if  __name__=='__main__':
    im1 = cv2.imread('D01s.jpg')
    im2 = cv2.imread('D0s.jpg')



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