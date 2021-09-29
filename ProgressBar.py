from tkinter import *
from tkinter.ttk import *
import time
import os
import enum

class SIZE_UNIT(enum.Enum):
   BYTES = 1
   KB = 2
   MB = 3
   GB = 4

def convert_unit(size_in_bytes, unit):
    if unit == SIZE_UNIT.KB:
        return size_in_bytes/1024
    elif unit == SIZE_UNIT.MB:
        return size_in_bytes/(1024*1024)
    elif unit == SIZE_UNIT.GB:
        return size_in_bytes/(1024*1024*1024)
    else:
        return size_in_bytes

def get_file_size(file_name, size_type = SIZE_UNIT.BYTES):
    size = os.path.getsize(file_name)
    return round(convert_unit(size,size_type),3)




window = Tk()

percent = StringVar()
text = StringVar()

bar = Progressbar(window,orient=HORIZONTAL,length=300)
bar.pack(pady=10)

percentLabel = Label(window,textvariable=percent).pack()
taskLabel = Label(window,textvariable=text).pack()


size_Image = 100
start_Processing = 0
speed =5
while(start_Processing<size_Image):
    time.sleep(0.05)
    bar['value']+=(speed/size_Image)*100
    start_Processing+=speed
    percent.set(str(int((start_Processing/size_Image)*100))+"%")
    text.set(str(round(start_Processing,3))+"/"+str(size_Image)+" MB")
    window.update_idletasks()

window.mainloop()