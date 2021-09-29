from PIL import Image
import numpy as np
img1=Image.open('lena_opencv_green.jpg')
img2=Image.open('lena_opencv_blue.jpg')

out=Image.new(img1.mode, img2.size)

(l,h)=img1.size
temp = []
print(l,h)
for j in range(0, 400):
    for i in range(0, 400):
        # out.getpixel((i,j)),(image_one.getpixel((i,j)) * (1.0 - 0) +  image_two.getpixel((i,j)) * 0.3 )
            cal1 = img1.getpixel((i,j))
            # cal1 = list(cal1)
            cal1 = np.array(cal1)

            cal2 = img2.getpixel((i,j)) * 1
            # cal2 = list(cal2)
            cal2 = np.array(cal2)
            result = [a + b for a, b in zip(cal1, cal2)]
            result = tuple(result)
            out.putpixel((i,j),result)
out.show()