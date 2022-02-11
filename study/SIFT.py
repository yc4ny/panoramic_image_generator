import cv2 
import matplotlib.pyplot as plt

# read images
img1 = cv2.imread('eiffel1.jpg')  
img2 = cv2.imread('eiffel2.jpg') 

# feature descriptor / sift
# sift = cv2.xfeatures2d.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

# nfeatures: 검출 최대 특징 수
# nOctaveLayers: 이미지 피라미드에 사용할 계층 수
# contrastThreshold: 필터링할 빈약한 특징 문턱 값
# edgeThreshold: 필터링할 엣지 문턱 값
# sigma: 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값

sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()

