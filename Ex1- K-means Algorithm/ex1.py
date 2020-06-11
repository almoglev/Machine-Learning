import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from init_centroids import init_centroids
from scipy.misc import imread


# input: array (centroid array)
# output: printing the array with the format required
def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]


# presents a graph on the screen- i use this to create the loss graph
# I kept this to show you how I made the graphs in my report.
# def plot_graph(loss, k_val):
#    plt.plot(loss)
#    plt.title("k = %d:" % k_val)
#    plt.ylabel("Average Loss Value")
#    plt.xlabel("Iteration Number")
#    plt.show()


# each cell in this array is the centroid of the pixel
# for example: cell 0 will contain the centroid of pixel number 0
centerToPixel = []
for k in[2,4,8,16]:
    # data preparation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    # initializing centroids
    centroids = init_centroids(X, k)
    print("k=%d:" % k)
    # array to store the loss of each iteration
    lossArray = []
    for i in range(0,11):
        centerToPixel = []
        print("iter %d: " % i, end='')
        print(print_cent(centroids))
        # distances sum of each iteration
        lossSum = 0
        # assign each pixel to its closest centroid
        for pixel in X:
            # find the minimal distance
            minDistance = np.linalg.norm(pixel - centroids[0])
            minDistance = minDistance * minDistance
            minCentroid = centroids[0]
            for c in centroids:
                tempMin = np.linalg.norm(pixel - c)
                tempMin = tempMin * tempMin
                if tempMin < minDistance:
                    minDistance = tempMin
                    minCentroid = c
            # set the centroid with the minimal distance to the pixel
            centerToPixel.append(minCentroid)
            lossSum = lossSum + minDistance
        # insert into loss array
        lossArray.append(lossSum/len(X))
        # Calculate the average distance of pixels from their centroid,
        # and update the centroid's new center to be that average.
        count = 0
        for c in centroids:
            j = 0
            count = 0
            sumCenterR = 0
            sumCenterG = 0
            sumCenterB = 0
            for ctp in centerToPixel:
                # sum all pixels that belong to the same centroid c
                if np.array_equal(c, ctp):
                    sumCenterR = sumCenterR + X[j][0]
                    sumCenterG = sumCenterG + X[j][1]
                    sumCenterB = sumCenterB + X[j][2]
                    centerToPixel[j][0] = sumCenterR
                    centerToPixel[j][1] = sumCenterG
                    centerToPixel[j][2] = sumCenterB
                    count = count + 1
                j = j + 1
            j = 0
            # if centroid c has at least one pixel assigned to it, then calculate
            # the average, and update centroid (I checked this to prevent dividing by 0).
            if count > 0:
                c[0] = sumCenterR / count
                c[1] = sumCenterG / count
                c[2] = sumCenterB / count
                centerToPixel[j] = c
    # update the original picture with the new centroids - new rgb color for each pixel
    update_i = 0
    for newPixel in centerToPixel:
        X[update_i] = newPixel
        update_i = update_i + 1

    # plot the image- I used it to show the photos in my report.
    # plt.imshow(A)
    # plt.grid(False)
    # plt.show()
    # plot the loss graph - I used it to show the graphs in my report.
    # plot_graph(lossArray, k)