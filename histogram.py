import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = "sample.jpeg"
img = cv.imread(path,0)
res = cv.imread(path,0) # image to be transformed

def show(img):
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plot(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

def histogramSliding(img):
    h,w=img.shape
    for i in range(h):
        for j in range(w):
            img[i,j] = img[i,j] -50 if img[i,j] >= 50 else 0
    show(img)
    plot(img)

def histogramEqualization(img):
    equ = cv.equalizeHist(img)
    show(equ)
    plot(equ)

def adaptiveHistogramEqualization(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    show(equalized)
    plot(equalized)

if __name__ == '__main__':

    'original (input) image'
    # show(img)
    # plot(img)

    'histogram applications'
    # histogramEqualization(img)
    # histogramSliding(img)
    # adaptiveHistogramEqualization(img)