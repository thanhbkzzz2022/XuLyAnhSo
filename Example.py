import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from numpy.lib.function_base import blackman

# img = cv.imread('image2.jpg')
# cv.imshow('Sieu Xe', img)
# cv.imwrite('sieuxe.jpg',img)
# cv.waitKey(0)

# img = imread('sieuxe.jpg')
# imshow(img)


# plt.figure()
# img = mpimg.imread('sieuxe.jpg')
# plt.imshow(img)
# plt.title('First Example')
# plt.savefig('123.jpg')
# plt.show()

# image = Image.open('sieuxe.jpg')
# image.save('hypercar.jpg','JPEG')

# #Convert PIL.Image to numpy.ndarray
# img = np.asarray(image)
# cv.imshow('123',img)
# Image.fromarray(img)

# cv.waitKey(0)

# image.show()

##Histogram with OpenCv
# img = cv.imread('sieuxe.jpg')
# hist, bin = np.histogram(img.ravel(), 256, [0,256])
# plt.hist(hist)
# plt.show()

# img = cv.imread('sieuxe.jpg')
# color = ('b','g','r')
# for i, col in enumerate(color):
#     print(i, col)
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()


# img = cv.imread('sieuxe.jpg')
# color = ('b','g','r')
# ## 0 - blue; 1 - green; 2 - red
# hist = cv.calcHist([img],[3],None,[256],[0,256])
# plt.plot(hist,color='g')
# plt.xlim([0,256])
# plt.show()


##Histgram using PIL.Image

# def getRed(redValue):
#     return '#%02x%02x%02x' % (redValue, 0, 0)

# image = Image.open('sieuxe.jpg')
# histogram = image.histogram()

# l1 = histogram[0:256]

# l2 = histogram[256:512]

# l3 = histogram[512:768]

# plt.figure(0)

# for i in range(0, 256):
#     plt.bar(i, l1[i], color = getRed(i), edgecolor=getRed(i),alpha=0.3)

# plt.show()

##### The báic opẻations on digital image

# blank_image = Image.new('RGBA', (400,300), 'white')
# img_draw = ImageDraw.Draw(blank_image)
# img_draw.rectangle((70,50,270,200), outline='red', fill='blue')
# img_draw.text((70,250), 'Hello World', fill='green')
# blank_image.save('drawn_image.tif')


##Draw Image with OpenCV

# img = cv.imread('sieuxe.jpg')
# blank = np.zeros((200,200,3), np.uint8)
# cv.rectangle(blank,(20,30),(50,50),(255,0,0),3)
# cv.putText(blank,'Hello World', (20,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0))

# cv.imshow('Hello',blank)
# cv.waitKey(0)
# cv.destroyAllWindows()


## Paste an image to another image

# image = Image.open('sieuxe.jpg')
# logo = Image.open('logo.png')

# image_copy = image.copy()

# position = ((image_copy.width - logo.width),(image_copy.height - logo.height))

# image_copy.paste(logo, position)
# image_copy.save('pasted_image.jpg')


img = cv.imread('sieuxe.jpg')
blank = np.zeros((200,200,3), np.uint8)
wb, hb, cb = blank.shape
w,h,c = img.shape
img[1:wb+1, 1:hb+1,:] = blank[:,:,:]
cv.imshow('Hello', img)
cv.waitKey(0)
cv.destroyAllWindows()
