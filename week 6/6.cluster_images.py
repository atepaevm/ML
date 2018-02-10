def get_mean_image(image,n=4):
    mean_image=image
    w,h,d=image.shape
    pixels = pandas.DataFrame(numpy.reshape(image, (w*h, d)), columns=['R', 'G', 'B'])
    kmeans = KMeans(init='k-means++',n_clusters=n, random_state=241).fit(pixels)
    pixels['cluster']=kmeans.fit_predict(pixels)
    means=pixels.groupby('cluster').mean().values
    mean_pixels=[means[c] for c in pixels['cluster'].values]
    mean_image=numpy.reshape(mean_pixels,(w,h,d))
    return mean_image

def get_median_image(image,n=4):
    median_image=image
    w,h,d=image.shape
    pixels = pandas.DataFrame(numpy.reshape(image, (w*h, d)), columns=['R', 'G', 'B'])
    kmeans = KMeans(init='k-means++',n_clusters=n, random_state=241).fit(pixels)
    pixels['cluster']=kmeans.fit_predict(pixels)
    medians=pixels.groupby('cluster').median().values
    median_pixels=[medians[c] for c in pixels['cluster'].values]
    median_image=numpy.reshape(median_pixels,(w,h,d))
    return median_image
def PSNR(first,second):
    MSE=numpy.mean((first - second) ** 2)
    return -10*math.log10(MSE)
	
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.io import imsave
import math
import pylab
import pandas
import numpy
from skimage import img_as_float


image = img_as_float(imread('D:\\1.jpg'))
for i in range(1,21):
    mean=get_mean_image(image,i)
    median=get_median_image(image,i)
    pylab.imsave('D:\\images\\'+str(i)+' mean '+str(PSNR(mean,image))+'.jpg',mean)
    pylab.imsave('D:\\images\\'+str(i)+' median '+str(PSNR(median,image))+'.jpg',median)