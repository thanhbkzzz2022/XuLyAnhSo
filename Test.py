from PIL import Image

def image_crop(img):
    img = Image.open(r + "'" + img + "'")
    width, height = img.size
    