

# -*- coding: utf-8 -*-


"""
/* -----------------------------------------------------------------------------
  Daniel Lu
  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist
  GitHub : 
*/

"""

#%%
# Standard library----------------------------------------------
import sys
sys.path.append('../data')
sys.path.append('D:\SourceCode\Python\DanielSourceCode\DeepLearning\RvLib')
import os
import time
from datetime import datetime

import cv2
from skimage.filters import threshold_mean

# Third-party libraries------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm




        
#%%%  Function Section
font = cv2.FONT_HERSHEY_SIMPLEX
fntSize = 1
fntThickness = 1

DoContrast = False
threshold = 150
dVal = 50



def Draw_Text(img, sTxt, aX=30, aY=30):
    if ""==sTxt: return
    cv2.putText(image, str(sTxt) ,(aX,aY), font, 
        fntSize,(0,255,255), fntThickness,cv2.LINE_AA)
    

def ForceDir(path):
    if not os.path.isdir(path):
        os.mkdir(path) 
                

        
outputPath = 'dataSet\\'
outputFn = "DanielLu"
incId=0

name = input("Enter your name (Now={}): ".format(outputFn) )

if ""!=name: outputFn = name

outputPath = "{}{}\\".format( outputPath,outputFn)
ForceDir(outputPath)
        

camera = cv2.VideoCapture(0)
while True:
    return_value,image = camera.read()  
    imgInfo = np.asarray(image).shape     
    if len(imgInfo)<2: break
    imgH=imgInfo[0]
    imgW=imgInfo[1]
    imgChannel=imgInfo[2] 
    
    if DoContrast:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[:,:] = [[ min(pixel + dVal, 255) 
            for pixel in row] for row in image[:,:]]
    #cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    try:                 
        Draw_Text(image, "esc:exit, s:save image")
        cv2.imshow('image',image)
        #plt.imshow( cv2.cvtColor(image,cv2.COLOR_BGR2RGB) )
            
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  #esc   ord('s'):
            #cv2.imwrite('test.jpg',image)
            break
        elif key == ord('s'):
            cv2.imwrite("{}{}_{}.jpg".format(outputPath, outputFn, incId), image)
            incId+=1
            
    except ValueError:
        break

camera.release()
cv2.destroyAllWindows()
        
print("{} images saved in : \"{}\"!".format(incId, outputPath))
        
        

    