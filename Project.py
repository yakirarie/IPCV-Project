import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob


class Panorama:
    def __init__(self, data):
        self.images = data
        self.source_points = []
        self.destination_points = []
        self.homographs = []
        self.__register_images()
        self.__leastSquareHomography()

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


def read_images():
    images = [cv.imread(file) for file in glob.glob("./*.jpg")]
    return images


a = read_images()
Panorama(a)

