import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


#
# Matches features given a list of keypoints, descriptors, and images
#
def matchFeatures(kp1, kp2, desc1, desc2, img1, img2):
    print("Matching Features...")
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    return matches


#
# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    # loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    # svd composition
    u, s, v = np.linalg.svd(matrixA)

    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    # normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
# Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    if estimatep2.item(2) == 0:
        estimatep2 = (1 / 0.0001) * estimatep2
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)


#
# Runs through ransac algorithm, creating homographies from random correspondences
#
def ransac(corr):
    maxInliers = []
    finalH = None
    for i in range(1000):
        # find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        print("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

    return finalH, maxInliers


#
# Main parses argument list and runs the functions
#
def register2images(img1name, img2name):

    print("Image 1 Name: " + img1name)
    print("Image 2 Name: " + img2name)

    # query image
    img1 = cv2.imread(img1name, 0)
    # train image
    img2 = cv2.imread(img2name, 0)
    # find features and keypoints

    correspondenceList = []
    if img1 is not None and img2 is not None:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(img1, None)
        kp2, desc2 = sift.detectAndCompute(img2, None)

        print("Found keypoints in " + img1name + ": " + str(len(kp1)))
        print("Found keypoints in " + img2name + ": " + str(len(kp2)))

        keypoints = [kp1,kp2]

        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = matcher.match(desc1, desc2)

        for match in matches:
            (x1, y1) = keypoints[0][match.queryIdx].pt
            (x2, y2) = keypoints[1][match.trainIdx].pt
            correspondenceList.append([x1, y1, x2, y2])

        corrs = np.matrix(correspondenceList)

        #run ransac algorithm
        finalH, inliers = ransac(corrs)
        print("Final homography: ", finalH)
        print("Final inliers count: ", len(inliers))
        return finalH, img1, img2


def create_panorama(images_names, pan_num):
    registerd_images = []
    first_img = cv2.imread(images_names[0], 0)
    height, width = first_img.shape
    h = np.eye(3, 3)
    for i in range(len(images_names)-1):
        h_curr, img_template, img2align = register2images(images_names[i], images_names[i+1])
        h = h_curr.dot(h)
        height += img2align.shape[0]
        width += img2align.shape[1]
        registerd_images.append(cv2.warpPerspective(img2align, h, (width, height),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
        # plt.imshow(registerd_images[i])
        # plt.show()
    panorama = np.zeros((height, width), np.float32)
    for i in range(len(registerd_images)-1, -1, -1):
        panorama[0:registerd_images[i].shape[0], 0:registerd_images[i].shape[1]] = registerd_images[i]
    panorama[0:first_img.shape[0], 0:first_img.shape[1]] = first_img
    cv2.imwrite('panorama'+str(pan_num)+'.jpg', panorama)


if __name__ == "__main__":

    inp = int(input("choose data: 1 - backyard , 2 - office, 3 - oxford"))
    if inp == 1:
        img1name = 'backyard1.jpg'
        img2name = 'backyard2.jpg'
        img3name = 'backyard3.jpg'
        imgs = [img1name, img2name, img3name]

    elif inp == 2:
        img1name = 'office1.jpg'
        img2name = 'office2.jpg'
        img3name = 'office3.jpg'
        img4name = 'office4.jpg'
        imgs = [img1name, img2name, img3name, img4name]

    else:
        img1name = 'oxford1.jpg'
        img2name = 'oxford2.jpg'
        imgs = [img1name, img2name]

    create_panorama(imgs, inp)






