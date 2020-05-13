import cv2
import glob, os


import glob

for file_ in glob.iglob("dataset/*/*/*.jpg"): # generator, search immediate subdirectories 
   
    if "r3_" in file_:
        os.remove(file_)
        print(file_)
    
    
    #img = cv2.resize(img,(96,96),interpolation=cv2.INTER_AREA)   
    #cv2.imwrite(file_, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #print("ok")