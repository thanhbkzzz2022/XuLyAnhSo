import tkinter as tk
from tkinter import Tk, Frame, Label, mainloop
from tkinter.ttk import *
from tkinter import *
from os import name, read
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time



class getColor():
    def getRed(self, value):
        return '#%02x%02x%02x' % (value,0,0)

    def getGreen(self, value):
        return '#%02x%02x%02x' % (0,value,0)

    def getBlue(self, value):
        return '#%02x%02x%02x' % (0,0,value)


def green_hist():
    image = Image.open('lena.jpg')
    histogram = image.histogram()
    l2 = histogram[256:512]
    plt.figure(0)
    s = getColor()
    for i in range(0, 256):
        plt.bar(i, l2[i], color = s.getGreen(i), edgecolor=s.getGreen(i),alpha=0.3)
    plt.savefig('green_hist_original.jpg')
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.title('Green Histogram')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    


def red_hist():
    image = Image.open('lena.jpg')
    histogram = image.histogram()
    l1 = histogram[0:256]
    plt.figure(0)
    s = getColor()
    for i in range(0, 256):
        plt.bar(i, l1[i], color = s.getRed(i), edgecolor=s.getRed(i),alpha=0.3)
    plt.savefig('red_hist_original.jpg')
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.title('Red Histogram')
    plt.show(block=False)
    plt.pause(1)
    plt.close()



def blue_hist():
    image = Image.open('lena.jpg')
    histogram = image.histogram()
    l3 = histogram[512:768]
    plt.figure(0)
    s = getColor()
    for i in range(0, 256):
        plt.bar(i, l3[i], color = s.getBlue(i), edgecolor=s.getBlue(i),alpha=0.3)
    plt.savefig('blue_hist_original.jpg')
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.title('Blue Histogram')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

    

######################################################################Exercises#####################################################################################
##1 Write a program to split an input image with RGB space into three images which present on Red, Green, Blue channels and then save on your drive.

def convert_RGB(image):
    img = cv.imread('lena.jpg', cv.IMREAD_COLOR)
    # img_goc = Image.open('lena.jpg')
    # height, width = img_goc.size
    width = image.size[0]
    height = image.size[1]
    # height, width = img.shape[:2]
    ## Get Image Size
    # print(width, height)
    # height, width, channels = img.shape
    # print(height, width, channels)
    ## Declare variable to store R-G-B image

    red = np.zeros((width,height,3), np.uint8)
    green = np.zeros((width,height,3), np.uint8)
    blue = np.zeros((width,height,3), np.uint8)

    red[:] = [0, 0, 0]
    green[:] = [0, 0, 0]
    blue[:] = [0, 0, 0]

    ## Duyet Anh
    for x in range(width):
        for y in range(height):
            ## Lay gia tri diem anh tai vi tri (x, y)
            R = img[x, y, 2]
            G = img[x, y, 1]
            B = img[x, y, 0]

            red[x, y, 2] = R
            green[x, y, 1] = G
            blue[x, y, 0] = B


    ## Display Image
    fig = plt.figure('Convert RGB Lena',figsize=(10,7))
    (ax1, ax2), (ax3, ax4) = fig.subplots(2,2)
    img_original = plt.imread('lena.jpg')
    ax1.axis("off")
    ax1.imshow(img)
    ax1.set_title('Original Lena')

    ax2.axis("off")
    ax2.imshow(blue)
    ax2.set_title('Red Lena')

    ax3.axis("off")
    ax3.set_title('Green Lena')
    ax3.imshow(green)
    

    ax4.axis("off")
    ax4.set_title('Blue Lena')
    ax4.imshow(red)

    cv.imwrite('lena_opencv_red.jpg', red)
    cv.imwrite('lena_opencv_green.jpg', green)
    cv.imwrite('lena_opencv_blue.jpg', blue)


    plt.show()
    ## Show Image
    cv.waitKey(0)

    ## Close Image
    cv.destroyAllWindows()

## Exercise 2: Write a program to show the histogram of images for each channel which is splitted from the previous exercise.
class calcHistogram:
    def calcHistogram_red(self,image):
        his = np.zeros(256)
        w = image.size[0]
        h = image.size[1]
        print(w,h)
        for x in range(w):
            for y in range(h):
                ## Lay gia tri mau tai diem (x,y)
                r, g, b = image.getpixel((x,y))
                his[r] += 1
        return his

    def calcHistogram_green(self,image):
        his = np.zeros(256)
        w = image.size[0]
        h = image.size[1]
        print(w,h)
        for x in range(w):
            for y in range(h):
                ## Lay gia tri mau tai diem (x,y)
                r, g, b = image.getpixel((x,y))
                his[g] += 1
        return his

    def calcHistogram_blue(self,image):
        his = np.zeros(256)
        w = image.size[0]
        h = image.size[1]
        print(w,h)
        for x in range(w):
            for y in range(h):
                ## Lay gia tri mau tai diem (x,y)
                r, g, b = image.getpixel((x,y))
                his[b] += 1
        return his

def plot_histogram(his,color):
    fig = plt.figure('So sánh Histogram ảnh gốc và ảnh R-G-B', figsize=(16,9), dpi=100)
    trucX = np.zeros(256)
    trucX = np.linspace(0,256,256) ## Tao ra mot ndarray gom 256 phan tu trong khoang tu 0 den 256
    rows = 2
    columns = 3
    fig.add_subplot(rows, columns, 1)
    plt.plot(trucX, his[0], color='orange')
    plt.title('BIỂU ĐỒ HISTOGRAM ' + color[0].upper())
    plt.xlabel('Giá trị màu ' + color[0])
    plt.ylabel('Số điểm cùng giá trị màu ' + color[0])
    plt.savefig('histogram_' + color[0] + '.png') ##Luu histogram

    fig.add_subplot(rows, columns, 2)
    plt.plot(trucX, his[1], color='orange')
    plt.title('BIỂU ĐỒ HISTOGRAM ' + color[1].upper())
    plt.xlabel('Giá trị màu ' + color[1])
    plt.ylabel('Số điểm cùng giá trị màu ' + color[1])
    plt.savefig('histogram_' + color[1] + '.png') ##Luu histogram

    fig.add_subplot(rows, columns, 3)
    plt.plot(trucX, his[2], color='orange')
    plt.title('BIỂU ĐỒ HISTOGRAM ' + color[2].upper())
    plt.xlabel('Giá trị màu ' + color[2])
    plt.ylabel('Số điểm cùng giá trị màu ' + color[2])
    plt.savefig('histogram_' + color[2] + '.png') ##Luu histogram
    

    ## Show histogram from original image to compare with RGB histogram
    dsize1 = (800,600)
    ori_red_hist = cv.imread('red_hist_original.jpg',cv.IMREAD_COLOR)
    ori_red_hist = cv.resize(ori_red_hist,dsize1)
    ori_green_hist = cv.imread('green_hist_original.jpg',cv.IMREAD_COLOR)
    ori_blue_hist = cv.imread('blue_hist_original.jpg',cv.IMREAD_COLOR)

    fig.add_subplot(rows, columns, 4)
    plt.imshow(ori_red_hist)
    plt.axis("off")
    plt.title('Histogram đỏ từ ảnh gốc')
    plt.xlabel("Giá trị màu đỏ")
    plt.ylabel("Số điểm cùng giá trị màu đỏ")

    fig.add_subplot(rows, columns, 5)
    plt.imshow(ori_green_hist)
    plt.axis("off")
    plt.title('Histogram xanh lá từ ảnh gốc')
    plt.xlabel("Giá trị màu xanh lá")
    plt.ylabel("Số điểm cùng giá trị màu xanh lá")

    fig.add_subplot(rows, columns, 6)
    plt.imshow(ori_blue_hist)
    plt.axis("off")
    plt.title('Histogram xanh lục từ ảnh gốc')
    plt.xlabel("Giá trị màu xanh lục")
    plt.ylabel("Số điểm cùng giá trị màu xanh lục")

    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()





class readImage:
    file_hinh = r'lena.jpg'
    file_hinh_1 = r'lena_opencv_red.jpg'
    file_hinh_2 = r'lena_opencv_green.jpg'
    file_hinh_3 = r'lena_opencv_blue.jpg'

img_read = readImage()
imgPLI = Image.open(readImage.file_hinh)
imgPIL_1 = Image.open(readImage.file_hinh_1)
imgPIL_2 = Image.open(readImage.file_hinh_2)
imgPIL_3 = Image.open(readImage.file_hinh_3)

image = imgPLI
image_1 = imgPIL_1
image_2 = imgPIL_2
image_3 = imgPIL_3

calcHis = calcHistogram()
his_red = calcHis.calcHistogram_red(image_1)
his_green = calcHis.calcHistogram_green(image_2)
his_blue = calcHis.calcHistogram_blue(image_3)

def plot_histogram_1(his,color):
    fig = plt.figure('Bieu do Histogram', figsize=(16,9), dpi=100)
    trucX = np.zeros(256)
    trucX = np.linspace(0,256,256) ## Tao ra mot ndarray gom 256 phan tu trong khoang tu 0 den 256
    plt.plot(trucX, his, color='orange')
    plt.title('Biểu đồ Histogram')
    plt.xlabel('Giá trị màu ' + color)
    plt.ylabel('Số điểm cùng giá trị màu ' + color)
    plt.savefig('histogram_' + color + '.png') ##Luu histogram
    plt.show()

def plot_histogram_function():

    his_red_original = calcHis.calcHistogram_red(image)
    his_green_original = calcHis.calcHistogram_green(image)
    his_blue_original = calcHis.calcHistogram_blue(image)


    # print(his_red)
    # print(his_red_1)
    # print(his_green)
    # print('---------------------------')
    # print(his_green_1)
    # print(his_blue)
    # print(his_blue_1)

    plot_histogram_1(his_red,'Đỏ')
    red_hist()
    plot_histogram_1(his_red_original, 'Xanh lá cây')
    plot_histogram_1(his_green, 'Xanh lá cây')
    plot_histogram_1(his_blue, 'Xanh lục')

    cv.waitKey(0)
    cv.destroyAllWindows()


# def show_multiple_his():
#     fig = plt.figure('Biểu đồ Histogram', figsize=(16,9))
#     (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12) = fig.subplots(4,3)
#     image = cv.imread('histogram_red.png', cv.IMREAD_COLOR)
#     ax1.imshow(image)
#     ax1.set_title("Red Histogram")
#     plt.show()



## Exercise 3: Write a program to crop a region of input image and save it:
##• PIL library (hint: image.crop((x, y, w, h)) function )
def image_crop_PIL():
    img = Image.open(r'sieuxe.jpg')
    np_img = np.array(img)
    width, height = img.size
    print(width, height)

    left = 110
    top = 130
    right = 550
    bottom = 380

    img_cropped = img.crop((left,top,right,bottom))
    img_cropped_1 = img_cropped.save('lamborgini.jpg')
    np_img_cropped = np.array(img_cropped)
    fig = plt.figure('Crop Image', figsize=(10,7))
    ax1, ax2 = fig.subplots(1,2)

    ax1.imshow(np_img)
    ax1.axis("off")
    ax1.set_title("Original Image")

    ax2.imshow(np_img_cropped)
    ax2.axis("off")
    ax2.set_title("Cropped Image")
    plt.show()


##• OpenCV library (hint: image[y : y+h, x : x+h] with x, y are vertical, horizontal value coordinate plane of image and w, h are the width and height of region)

def image_crop_OpenCV():
    img = cv.imread('sieuxe.jpg',cv.IMREAD_COLOR)
    # blank = np.zeros((200,200,3),np.uint8)
    # wb, hb, cb = blank.shape
    width = 440
    height = 250
    img_1 = img[130:height+130, 110:width+110,:]
    cv.imshow('Hello', img_1)
    cv.waitKey()
    cv.destroyAllWindows()

## Write a program to draw a rectangle, circle, eclipse shape on image and put text Hello world on an input image by OpenCV library.
## Draw Circle
def draw_rectangle():
    img = cv.imread('sieuxe.jpg', cv.IMREAD_COLOR)
    # blank = np.zeros((200,200,3), np.uint8)
    cv.rectangle(img,(110,130),(550,380),(0,100,0),3)
    cv.putText(img,'Lamborghini Aventador',(150,100),cv.FONT_HERSHEY_SIMPLEX,1,(0,69,255),thickness=2)
    cv.imshow('Rectangle Image', img)
    cv.imwrite('Rectangle.jpg',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_circle():
    img = cv.imread('sieuxe.jpg', cv.IMREAD_COLOR)
    center_coordinates = (330,250)
    radius = 220
    color = (0,0,255)
    thickness = 2
    cv.circle(img,center_coordinates,radius=radius,color=color,thickness=thickness)
    cv.putText(img,'Lamborghini Aventador',(150,150),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2)
    cv.imshow('Circle Image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_eclipse():
    img = cv.imread('sieuxe.jpg',cv.IMREAD_COLOR)
    center_coordinates = (320,310)
    axesLength = (100, 50)
    angle = 0
    startAngle = 0
    endAngle = 360
    color = (0, 0, 255)
    thickness = 5
    cv.ellipse(img ,center_coordinates ,axesLength ,angle,startAngle,endAngle,color,thickness)
    cv.putText(img,'Lamborghini Aventador',(150,150),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),thickness=2)
    cv.imshow('Ellipse Image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

## Exercise 5: Write a program to overlay a smaller image on a larger image using OpenCV library.
def overlay_image():
    img = cv.imread('sieuxe.jpg')
    logo = cv.imread('logo_xe.jpg')
    # width, height, channel = logo.shape
    # print(width,height)
    logo = cv.resize(logo,(150,150))
    # img_np = np.array(img)
    # logo_np = np.array(logo)
    # img_np[0:200,0:200,0] = logo_np
    x_offset=y_offset=10
    # img[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1]] = logo
    y1, y2 = y_offset, y_offset + logo.shape[0]
    x1, x2 = x_offset, x_offset + logo.shape[1]

    alpha_s = logo[:, :, 2] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * logo[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
    cv.imshow('Overlay Image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

## Exercise 6: Write a program to try on many different glasses for an input of face image.
class glasses:
    glasses_1 = '1a.png'
    glasses_2 = '2a.png'
    glasses_3 = '3a.png'
    glasses_4 = '4a.png'
    glasses_5 = '1b.png'
    glasses_6 = '2b.png'
    glasses_7 = '3b.png'
    glasses_8 = '4b.png'
    face_1 = 'a.png'
    face_2 = 'b.png'

mat_kinh = glasses()

def try_glasses(img1,img2,glasses):
    face_1 = cv.imread(img1,cv.IMREAD_COLOR)
    face_1 = cv.resize(face_1,(384,319))

    face_2 = cv.imread(img2,cv.IMREAD_COLOR)
    face_2 = cv.resize(face_2,(384,319))

    kinh = cv.imread(glasses,cv.IMREAD_COLOR)
    kinh = cv.resize(kinh,(245,91))

    cv.imwrite('resize_image_1.png',face_1)
    cv.imwrite('resize_image_2.png',face_2)

    x_offset_1=60 #40
    y_offset_1=100 #100

    x_offset_2=75 #40
    y_offset_2=75 #100
    
    y1, y2 = y_offset_1, y_offset_1 + kinh.shape[0]
    x1, x2 = x_offset_1, x_offset_1 + kinh.shape[1]

    y3, y4 = y_offset_2, y_offset_2 + kinh.shape[0]
    x3, x4 = x_offset_2, x_offset_2 + kinh.shape[1]

    face_1[y_offset_1:y_offset_1+kinh.shape[0], x_offset_1:x_offset_1+kinh.shape[1]] = kinh

    face_2[y_offset_2:y_offset_2+kinh.shape[0], x_offset_2:x_offset_2+kinh.shape[1]] = kinh
    # alpha_s = kinh[:,:,2] / 255.0
    # alpha_l = 1.0 - alpha_s
    # for c in range(0, 3):
    #     face[y1:y2, x1:x2, c] = (alpha_s * kinh[:, :, c] + alpha_l * face[y1:y2, x1:x2, c])


    cv.imwrite('try_on_glasses_1.png',face_1)
    cv.imwrite('try_on_glasses_2.png',face_2)
    
    process()
    img_1 = "try_on_glasses_1.png"
    img_2 = "try_on_glasses_2.png"
    img_show = Image.open(img_1)
    img_show = ImageTk.PhotoImage(img_show)
    lbl.configure(image=img_show)
    lbl.image = img_show

    img_show_1 = Image.open(img_2)
    img_show_1 = ImageTk.PhotoImage(img_show_1)
    lbl1.configure(image=img_show_1)
    lbl1.image = img_show_1

    # cv.imshow('Try Glasses Woman 1',face_1)
    # cv.imshow('Try Glasses Woman 2',face_2)

    cv.waitKey(0)
    cv.destroyAllWindows()


def interface_project():
    global window
    window = tk.Tk()
    w, h = window.winfo_screenwidth(), window.winfo_screenheight()
    window.geometry("%dx%d+0+0" % (w, h))
    window.title("Chuong 2")
    anh = Image.open('Xulyanhso1.png')
    resizeImage = anh.resize((1920,1080))
    imagebia = ImageTk.PhotoImage(resizeImage)
    l1 = Label(image=imagebia)
    l1.grid(column=0,row=0)
    photo_1 = PhotoImage(file = "1a.png")
    photo_2 = PhotoImage(file = "2a.png")
    photo_3 = PhotoImage(file = "3a.png")
    photo_4 = PhotoImage(file = "4a.png")

    photo_5 = PhotoImage(file = "1b.png")
    photo_6 = PhotoImage(file = "2b.png")
    photo_7 = PhotoImage(file = "3b.png")
    photo_8 = PhotoImage(file = "4b.png")

    face_show_1 = PhotoImage(file = "resize_image.png")
    face_show_2 = PhotoImage(file = "b.png")

    face_show_1_glasses = PhotoImage(file = "resize_image_1.png")
    face_show_2_glasses = PhotoImage(file = "resize_image_2.png")

    global lbl
    lbl = Label(window,bg="white")
    lbl.place(relx=0.3, rely=0.8, anchor=CENTER)

    global lbl1
    lbl1 = Label(window,bg="white")
    lbl1.place(relx=0.7, rely=0.8, anchor=CENTER)

    Btn = tk.Button(window,
                    text="",
                    image = photo_1,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_1),
                    borderwidth=0
                    )
    Btn1 = tk.Button(window,
                    text="",
                    image = photo_2,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_2),
                    borderwidth=0
                    )
    Btn2 = tk.Button(window,
                    text="",
                    image = photo_3,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_3),
                    borderwidth=0
                    )
    Btn3 = tk.Button(window,
                    text="",
                    image = photo_4,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_4),
                    borderwidth=0
                    )
    face_1 = tk.Button(window,
                    text="",
                    image = face_show_1,
                    width = 384,
                    height = 319,
                    command='',
                    borderwidth=0
                    )
    # global face_1_try_glasses
    # face_1_try_glasses = tk.Button(window,
    #                     text="",
    #                     image = face_show_1_glasses,
    #                     width = 384,
    #                     height = 319,
    #                     command='',
    #                     borderwidth=0
    #                     )    
    
    Btn.place(relx=0.1, rely=0.1, anchor=CENTER)
    Btn1.place(relx=0.1, rely=0.25, anchor=CENTER)
    Btn2.place(relx=0.1, rely=0.40, anchor=CENTER)
    Btn3.place(relx=0.1, rely=0.55, anchor=CENTER)
    face_1.place(relx=0.1, rely=0.8, anchor=CENTER)
    # face_1_try_glasses.place(relx=0.35, rely=0.85, anchor=CENTER)
    

    Btn4 = tk.Button(window,
                    text="",
                    image = photo_5,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_5),
                    borderwidth=0
                    )
    Btn5 = tk.Button(window,
                    text="",
                    image = photo_6,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_6),
                    borderwidth=0
                    )
    Btn6 = tk.Button(window,
                    text="",
                    image = photo_7,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_7),
                    borderwidth=0
                    )   
    Btn7 = tk.Button(window,
                    text="",
                    image = photo_8,
                    width = 259,
                    height = 96,
                    command=lambda:try_glasses(mat_kinh.face_1,mat_kinh.face_2,mat_kinh.glasses_8),
                    borderwidth=0
                    )
    face_2 = tk.Button(window,
                    text="",
                    image = face_show_2,
                    width = 384,
                    height = 319,
                    command=try_glasses,
                    borderwidth=0
                    )
    # face_2_try_glasses = tk.Button(window,
    #                     text="",
    #                     image = face_show_2_glasses,
    #                     width = 384,
    #                     height = 319,
    #                     command='',
    #                     borderwidth=0
    #                     )  
    Btn4.place(relx=0.9, rely=0.1, anchor=CENTER)
    Btn5.place(relx=0.9, rely=0.25, anchor=CENTER)
    Btn6.place(relx=0.9, rely=0.40, anchor=CENTER)
    Btn7.place(relx=0.9, rely=0.55, anchor=CENTER)
    face_2.place(relx=0.9, rely=0.8, anchor=CENTER)
    # face_2_try_glasses.place(relx=0.7, rely=0.8, anchor=CENTER)


    global percent, text, bar, percentLabel, taskLabel
    percent = StringVar()
    text = StringVar()

    bar = Progressbar(window,orient=HORIZONTAL,length=300)
    bar.place(relx=0.5, rely=0.659, anchor=CENTER)

    percentLabel = Label(window,bg="white",font=("Arial", 16),textvariable=percent).place(relx=0.5, rely=0.69, anchor=CENTER)
    taskLabel = Label(window,bg="white",font=("Arial", 16),textvariable=text).place(relx=0.5, rely=0.72, anchor=CENTER)
    
    window.mainloop()

def process():
        bar['value'] = 0
        size_Image = 100
        start_Processing = 0
        speed = 5
        while(start_Processing<size_Image):
            time.sleep(0.05)
            bar['value']+=(speed/size_Image)*100
            start_Processing+=speed
            percent.set("Đã hoàn thành: "+str(int((start_Processing/size_Image)*100))+"%")
            # text.set(str(round(start_Processing,3))+"/"+str(size_Image)+" MB")
            if start_Processing == 100:
                text.set("Đã xử lý ảnh hoàn tất!!!")
            if start_Processing < 100:
                text.set("                                      ")
            window.update_idletasks()
        

##Exercise 7: Write a program to add, subtract, multiply two image and then the results are saved with format jpg, bmp. What do you observe?
class Exercise7(object):
    def __init__(self, img1,img2):
        self.img1 = img1
        self.img2 = img2

    def add_image(self):
        img1 = self.img1
        img2 = self.img2
        newsize = (300, 300)
        image_1 = Image.open(img1)
        image_1 = image_1.convert('RGB')
        image_1 = image_1.resize(newsize)
        image_1 = np.asarray(image_1)
        image_2 = Image.open(img2) 
        # Neu la CMYK thi convert sang RGB (RGB - channel = 3; CMYK - channel = 4)
        image_2 = image_2.convert('RGB')
        image_2 = image_2.resize(newsize)
        image_2 = np.asarray(image_2)

        ##result = np.add(image_1,image_2)
        result = image_1 + image_2
        img = Image.fromarray(result, 'RGB')
        img.save("Add Image JPG.jpg")
        img.save("Add Image BMP.bmp")
        img.show()

    def sub_image(self):
        img1 = self.img1
        img2 = self.img2
        newsize = (300, 300)
        image_1 = Image.open(img1)
        image_1 = image_1.convert('RGB')
        image_1 = image_1.resize(newsize)
        image_1 = np.asarray(image_1)
        image_2 = Image.open(img2) 
        # Neu la CMYK thi convert sang RGB (RGB - channel = 3; CMYK - channel = 4)
        image_2 = image_2.convert('RGB')
        image_2 = image_2.resize(newsize)
        image_2 = np.asarray(image_2)
        

        result = image_1 - image_2
        img = Image.fromarray(result, 'RGB')
        img.save("Subtract Image JPG.jpg")
        img.save("Subtract Image BMP.bmp")
        img.show()

    def mul_image(self):
        img1 = self.img1
        img2 = self.img2
        newsize = (300, 300)
        image_1 = Image.open(img1)
        image_1 = image_1.convert('RGB')
        image_1 = image_1.resize(newsize)
        image_1 = np.asarray(image_1)
        image_2 = Image.open(img2) 
        # Neu la CMYK thi convert sang RGB (RGB - channel = 3; CMYK - channel = 4)
        image_2 = image_2.convert('RGB')
        image_2 = image_2.resize(newsize)
        image_2 = np.asarray(image_2)
        

        result = image_1 * image_2
        img = Image.fromarray(result, 'RGB')
        img.save("Multiply Image JPG.jpg")
        img.save("Multiply Image BMP.bmp")        
        img.show()

## Exercise 8: Write functions to add two images f1(x) and f2(x) based on the following equations:
## g(x, y) = (1 − a)f1(x, y) + af2(x, y), where a from [0, 1] (Image blending)
# def equation_1(img1,img2):
#     image_one=Image.open(img1)
#     image_two=Image.open(img2)

#     out=Image.new(image_one.mode, image_two.size)

#     (l,h)=image_one.size
#     for j in range(0, h):
#         for i in range(0, l):
#             out.getpixel((i,j)),(image_one.getpixel((i,j)) * (1.0 - 0.3) +  image_two.getpixel((i,j)) * 0.3 )

#     out.save("testaando.jpg","JPEG")
#     out.show()

#     print(ketqua)

def changeImageSize(image):
    
    widthRatio  = 800/image.size[0]
    heightRatio = 500/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def blending_image(image1,image2,a):
    img1=cv.imread(image1)
    img2=cv.imread(image2)
    out_img = np.zeros(img1.shape,dtype=img1.dtype)
    out_img[:,:,:] = (a * img1[:,:,:]) + ((1-a) * img2[:,:,:])
    cv.imshow('Output',out_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # out=Image.new(img1.mode, img2.size)

    # (l,h)=img1.size
    # for j in range(0, l):
    #     for i in range(0, h):
    #         # out.getpixel((i,j)),(image_one.getpixel((i,j)) * (1.0 - 0) +  image_two.getpixel((i,j)) * 0.3 )
    #             cal1 = img1.getpixel((i,j))
    #             # cal1 = list(cal1)
    #             cal1 = np.array(cal1) * (1 - a)

    #             cal2 = img2.getpixel((i,j)) 
    #             # cal2 = list(cal2)
    #             cal2 = np.multiply(np.array(cal2),a)
    #             result = [int(a) + int(b) for a, b in zip(cal1, cal2)]
    #             result = tuple(result)
    #             print(result)
    #             out.putpixel((i,j),(result))

    # out.show()

## g(x, y) = a.f1(x, y) + b.f2(x, y) + k, where a, b from [0, 1] and k = 0
def blending_image_with_k(image1,image2,a,b,k=0):
    img1=cv.imread(image1)
    img2=cv.imread(image2)

    out_img = np.zeros(img1.shape,dtype=img1.dtype)
    out_img[:,:,:] = (a * img1[:,:,:]) + (b * img2[:,:,:]) + k
    cv.imshow('Output with k value',out_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

## Exercise 9: Write a function to compute a mean image from many different images and than it is saved on your drive.






if __name__ == "__main__":
    convert_RGB(image)
    green_hist()
    red_hist()
    blue_hist()
    plot_histogram([his_red,his_green,his_blue],['Đỏ','Xanh lá cây','Xanh lục'])
    image_crop_PIL()
    image_crop_OpenCV()
    draw_rectangle()
    draw_circle()
    draw_eclipse()
    overlay_image()
    interface_project()
    # Ex7 - Nhận xét: Ảnh được lưu bằng định dạng JPG mờ hơn so với ảnh được lưu với định dạng BMP
    Ex7 = Exercise7('logo_xe.jpg','lena.jpg')
    Ex7.sub_image()
    Ex7.add_image()
    Ex7.mul_image()
    blending_image('lena_opencv_blue.jpg','lena_opencv_red.jpg',0.7)
    blending_image_with_k('lena_opencv_blue.jpg','lena_opencv_red.jpg',0.8,0.8)
    # plot_histogram_function()
    # show_multiple_his()



