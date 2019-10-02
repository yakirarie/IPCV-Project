import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob

DEBUG_MODE = True



class Panorama:
    def __init__(self, data):
        self.images = data
        self.source_points = []
        self.destination_points = []
        self.homographs = []
        self.__register_images()

        self.__register_images_write_matches()
        # self.__debug('source points',len(self.source_points))
        self.__leastSquareHomography()
        # self.__apply_homography(np.array([[1, 1], [0, 0]]), np.array([[1,0,0],[0,1,0],[0,0,1]])) #test
        self.__debug('homograph', self.homographs)

    def __register_images(self):
        for i in range(len(self.images)-1):
            # Initiate ORB detector
            orb = cv.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(self.images[i], None)
            kp2, des2 = orb.detectAndCompute(self.images[i+1], None)
            # create BFMatcher object
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            self.source_points.append(np.asarray(src_pts))
            self.destination_points.append(np.asarray(dst_pts))

    def __register_images_write_matches(self):
        #------    
        for i in range(0, len(self.images)-1):
            # Initiate ORB detector
            orb = cv.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(self.images[i], None)
            kp2, des2 = orb.detectAndCompute(self.images[i+1], None)
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            goodList = []
            good = []
            # Filtering good matches based on a distance of 0.75 between 
            # keypoint pairs in 2 images
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    goodList.append([m])
                    good.append(m)
            
            pathimWrite = './task_matches_'+str(i)+'_knn.jpg'
            imagePlot = cv.drawMatchesKnn(self.images[i],kp1,self.images[i+1],kp2,goodList,None,flags=2)
            cv.imwrite(pathimWrite,imagePlot)
            print(type(goodList))
        #------

    def __leastSquareHomography(self):
        for i in range(len(self.source_points)):
            p1 = self.source_points[i][:4]
            p2 = self.destination_points[i][:4]
            A = []
            for i in range(0, len(p1)):
                x, y = p1[i][0], p1[i][1]
                u, v = p2[i][0], p2[i][1]
                A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
                A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

            A = np.asarray(A)
            U, D, Vt = np.linalg.svd(A)
            L = Vt[-1, :] / Vt[-1, -1]
            H = L.reshape(3, 3)
            self.homographs.append(H)
        
        

    def __apply_homography(self, pos1, H12):
        pos2 = []
        for i in range(pos1.shape[0]):
            pos2_point = H12.dot(np.append(pos1[i], 1).reshape(3, 1))
            pos2_point = np.array([pos2_point[0]/pos2_point[-1], pos2_point[1]/pos2_point[-1]])
            pos2.append(pos2_point)
        print(pos2)
        return pos2


    # @params: inlierTol − inlier tolerance threshold.
    # @ret: H12 - 3x3 homography matrix
    #       inliers - [] of indices in pos1/pos2 of max set of inlier matched found.
    def __ransacHomography(self, pos1, pos2, numIters, inlierTol):
        if(DEBUG_MODE):
            # Getting homography matrix after applying RANSAC on
            # well matched keypoints on both images with projection error <= 1
            H, mask = cv.findHomography(pos1, pos2, cv2.RANSAC)
            print('Homography Matrix:')
            print(H)
        
        #return (H12, inliers)

    
    def __debug(self, title, data):
        print("LOG::{} \n{}".format(title, data))



def read_images():
    images = [cv.imread(file) for file in glob.glob("./*.jpg")]
    return images


a = read_images()
Panorama(a)




