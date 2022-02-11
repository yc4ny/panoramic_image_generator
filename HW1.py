import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image 

def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """
    
    # TODO Your code goes here

    # Used Nearest Neighbors function imported from sklearn.neighbors
    #Optimal "n" value is the square root of the total number of samples
    #root 8 = 2.xx, set n = 2
    neighbors = NearestNeighbors(n_neighbors = 2)

    #This NearestNeighbors instance is not fitted yet. 
    #Call 'fit' with appropriate arguments before using this estimator.
    neighbors.fit(des2)
    distance, index = neighbors.kneighbors(des1, 2)

    #Distance ratio for Nearest Neighbors
    ratio = 0.7

    #Array for matched keypoint locations 
    x1 = []
    x2 = []

    #Store matched keypoints according to the distance ratio
    for i, distance in enumerate(distance):
        if ratio*distance[1] > distance[0]:
            x1.append(loc1[i])
            x2.append(loc2[index[i][0]])

    #Convert to ndarray of size (n,2)
    x1 = np.array(x1).reshape(len(x1),2)
    x2 = np.array(x2).reshape(len(x2),2)

    return x1, x2


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # TODO Your code goes here

    #estimating homography using the RANSAC Method 
    #keeps the homography with the smallest number of outliers 
    for r in range(ransac_n_iter):

        #randomly pick 4 good matches and compute homography 
        s1 = np.random.randint(x1.shape[0], size=4)
        s2 = np.random.randint(x2.shape[0], size=4)
        x1_s = x1[s1]
        x2_s = x2[s1]

        #check if the 4 matches are collinear if true, proceed
        if collinear(x1_s[0], x1_s[1], x1_s[2]) or collinear(x1_s[0], x1_s[1], x1_s[3]) or collinear(x1_s[1], x1_s[2], x1_s[3]) or collinear(x1_s[0], x1_s[2], x1_s[3]):
            continue
        
        #initialize A,b,x in Ax = b
        A = []
        b = []
        x = []

        #add values in matrix A,b 
        for i in range(len(x1_s)):
            x_1, y_1 = x1_s[i][0], x1_s[i][1]
            x_2, y_2 = x2_s[i][0], x2_s[i][1]
            A.append([x_1, y_1, 1, 0, 0, 0, -x_1*x_2, -y_1*x_2])
            A.append([0, 0, 0, x_1, y_1, 1, -x_1*y_2, -y_1*y_2])
            b.append([x_2])
            b.append([y_2])

        #finding homography
        #solve for x 
        #x = (A^(T)A)^(-1)A^(T)b
        #lecture reference: pg 57 in https://www-users.cse.umn.edu/~hspark/csci5563_S2021/Lec6_ImageTransform.pdf
        A = np.matrix(A)
        b = np.matrix(b)
        x = np.linalg.inv(A.T @ A) @ A.T @ b

        #computed homography, add last value as 1 (property of homogenous coordinates)
        H = np.concatenate((x,np.array([[1]])), axis=0).reshape(3,3)

        #declare inlier indicies
        #inliers: good matches that are consistent with the homography
        #good match: smaller than the error threshold
        inlier_idx=[]
        len_inlier = 0

        #zip: 2* 1D matrix -> 1* 2D matrix
        for i, (point1, point2) in enumerate(zip(x1,x2)):
            #homogenous coordinate 
            homo_coord = H * ((np.append(point1,1)).reshape(3,1))
            #(x,y,z) -> (x/z, y/z)
            eucl_coord = (homo_coord/homo_coord[2])[:2]
            #compare value with error threshold
            if np.linalg.norm(eucl_coord - point2.reshape(2,1)) < ransac_thr:
                inlier_idx.append(i)

        #if needed, update new length of minimum inlier, inlier indices
        #and the homography matrix
        if len(inlier_idx) > len_inlier:
            len_inlier = len(inlier_idx)
            inlier_final = np.array(inlier_idx)
            H_final = H

    return H_final, inlier_final

#Source: https://stackoverflow.com/questions/9608148/python-script-to-determine-if-x-y-coordinates-are-colinear-getting-some-e
def collinear(p0, p1, p2):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < 1e-12

def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """
    
    # TODO Your code goes here

    #openCV function: decomposeHomographyMat
    #num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, K)

    #R = K^(-1) x H x K 
    invK = np.linalg.inv(K)
    R = invK @ H @ K 

    #svd on R
    u, s, v = np.linalg.svd(R)

    #u x v 
    R = u @ v

    #Check sign of determinant and if -1, multiply -1 to make positive
    detR = np.linalg.det(R)
    det_sign = np.sign(detR)

    if det_sign < 0:
        R = R *(-1)

    return R 


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hw, Hc, 3) typing error
        The 3D points corresponding to all pixels in the canvas
    """

    # TODO Your code goes here

    #Surface points set to default
    cylin_surf = np.zeros((Hc, Wc, 3))

    for n in range(Hc):
        for m in range(Wc):
            angle = 2*m*np.pi/Wc

            cylin_surf[n][m][0] = K[0][0]*np.sin(angle) 
            cylin_surf[n][m][1] = n - Hc/2
            cylin_surf[n][m][2] = K[0][0]*np.cos(angle)

    p = cylin_surf

    return p  


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """
    
    # TODO Your code goes here

    #Height and width of canvas
    height = p.shape[0]
    width = p.shape[1]

    #Initialize the 2d projection canvas and mask continaing valid pixels
    u = np.zeros((height,width,2))
    mask = np.zeros((height, width))

    h, w, z = p.shape
    for h in range(height):
        for w in range(width):
            p_coord = np.array([p[h][w]]).reshape(3,1)

            points = R @ p_coord
            h_matrix = K @ points 

            #Normalizing
            u[h][w][0] = int(h_matrix[0]/h_matrix[2])
            u[h][w][1] = int(h_matrix[1]/h_matrix[2])

            #Conditions when pixels are valid (within the canvas)
            if u[h][w][0]>0 and u[h][w][0]<W and u[h][w][1]>0 and u[h][w][1]<H and points[2] > 0 :
                mask[h][w] = 1

    # u: The mapped 2D pixel locations in the source image for pixel transport

    return u, mask

def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """
    
    # TODO Your code goes here

    #height, width of canvas
    height = u.shape[0]
    width = u.shape[1]

    #initialize canvas generated by the i-th source image
    canvas_i=np.zeros((Hc, Wc, 3))

    #warp image to cylindrical canvas when the valid pixel indicator's value is 1
    for h in range(Hc):
        for w in range(Wc):
            if mask_i[h][w]==1:
                canvas_i[h][w][0]=image_i[int(u[h][w][1])][int(u[h][w][0])][2]
                canvas_i[h][w][1]=image_i[int(u[h][w][1])][int(u[h][w][0])][1]
                canvas_i[h][w][2]=image_i[int(u[h][w][1])][int(u[h][w][0])][0]

    #Image data of dtype object cannot be converted to float -> add dtype int 
    canvas_i = np.asarray(canvas_i, dtype = "uint8")

    #the canvas image generated by the i-th source image
    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    
    # TODO Your code goes here

    #height, width of canvas
    height = canvas.shape[0]
    width = canvas.shape[1]

    # if the valididty of pixels is 1 in the mask, update canvas
    for h in range(height):
        for w in range(width):
            #when pixel is valid 
            if mask_i[h][w]==1:
                #draw pixel on the i-th canvas
                canvas[h][w][0] = canvas_i[h][w][0]
                canvas[h][w][1] = canvas_i[h][w][1]
                canvas[h][w][2] = canvas_i[h][w][2]
    
    return canvas


if __name__ == '__main__':
    ransac_n_iter = 500
    ransac_thr = 3
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
		# TODO Your code goes here
        i1 = im_list[i]
        i2 = im_list[i+1]
        # Extract SIFT features
		# TODO Your code goes here

        #Create SIFT object
        sift = cv2.xfeatures2d.SIFT_create()

        #Convert images to greyscaleSS
        i1_grey = cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY)
        i2_grey = cv2.cvtColor(i2,cv2.COLOR_BGR2GRAY)

        #Detect Keypoints and Descriptors using SIFTSS
        key1, des1 = sift.detectAndCompute(i1_grey, None)
        key2, des2 = sift.detectAndCompute(i2_grey, None)

        #Key point locations in I_i and I_i+1
        loc1 = np.array([key1[n].pt for n in range(0, len(key1))])
        loc2 = np.array([key2[n].pt for n in range(0, len(key2))])

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)
		
		# Compute R_new (or R_i+1)
		# TODO Your code goes here
        R_new = R@rot_list[-1]
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]
    
    Hc = Him
    Wc = len(im_list) * Wim // 2
	
    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)