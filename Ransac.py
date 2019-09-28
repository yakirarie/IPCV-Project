# coding: utf-8
import numpy as np
import math
import pylab
import scipy.io as sio
import cv2

class LinearLeastSquares2D(object):

    '''
    2D linear least squares using the hesse normal form:
        d = x*sin(theta) + y*cos(theta)
    which allows you to have vertical lines.
    '''

    def fit(self, data):
        data_mean = data.mean(axis=0)
        x0, y0 = data_mean
        if data.shape[0] > 2: # over determined
            u, v, w = np.linalg.svd(data-data_mean)
            vec = w[0]
            theta = math.atan2(vec[0], vec[1])
        elif data.shape[0] == 2: # well determined
            theta = math.atan2(data[1,0]-data[0,0], data[1,1]-data[0,1])
        theta = (theta + math.pi * 5 / 2) % (2*math.pi)
        d = x0*math.sin(theta) + y0*math.cos(theta)
        return d, theta

    def residuals(self, model, data):
        d, theta = model
        dfit = data[:,0]*math.sin(theta) + data[:,1]*math.cos(theta)
        return np.abs(d-dfit)

    def is_degenerate(self, sample):
        return False


def ransac(data):
    threshold = 2
    max_trials = 1000
    model_class = LinearLeastSquares2D()
    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idx = np.arange(data.shape[0])
    for _ in range(max_trials):
        sample = data[np.random.randint(0, data.shape[0], 2)]
        if model_class.is_degenerate(sample):
            continue
        sample_model = model_class.fit(sample)
        sample_model_residua = model_class.residuals(sample_model, data)
        sample_model_inliers = data_idx[sample_model_residua<threshold]
        inlier_num = sample_model_inliers.shape[0]
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_inliers = sample_model_inliers
    if best_inliers is not None:
        best_model = model_class.fit(data[best_inliers])
    return best_model


def Normalization(nd, x):
    import numpy as N

    x = N.asarray(x)
    m, s = N.mean(x, 0), N.std(x)
    if nd == 2:
        Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    Tr = N.linalg.inv(Tr)
    x = N.dot(Tr, N.concatenate((x.T, N.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x

x = np.mgrid[-5:5:200j]
y = np.mgrid[3:10:200j]
data = np.vstack((x.ravel(), y.ravel())).T
data += np.random.normal(size=data.shape)

# generate some faulty data
data[0,:] = (3, 20)
data[1,:] = (4, 21)
data[2,:] = (5, 22)
data[3,:] = (5, 24)
data[4,:] = (-2, -24)
data[5,:] = (-3, -23)

model = ransac(data)

pylab.plot(data[:,0], data[:,1], '.b', label='DATA')
x = np.arange(-7, 8)
dr, thetar = model
y_ransac = (dr - x*math.sin(thetar)) / math.cos(thetar)
pylab.plot(x, y_ransac, '-b', label='RANSAC')
pylab.legend(loc=4)
pylab.show()





