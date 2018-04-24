

# -*- coding: utf-8 -*-


"""
/* -----------------------------------------------------------------------------
  Copyright: (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist

  This software may be freely copied, modified, and redistributed
  provided that this copyright notice is preserved on all copies.
  The intellectual property rights of the algorithms used reside
  with the Daniel Lu, RasVector Technology.

  You may not distribute this software, in whole or in part, as
  part of any commercial product without the express consent of
  the author.

  There is no warranty or other guarantee of fitness of this
  software for any purpose. It is provided solely "as is".

  ---------------------------------------------------------------------------------
  版權宣告  (C) Daniel Lu, RasVector Technology.

  Email : dan59314@gmail.com
  Web :     http://www.rasvector.url.tw/
  YouTube : http://www.youtube.com/dan59314/playlist

  使用或修改軟體，請註明引用出處資訊如上。未經過作者明示同意，禁止使用在商業用途。
*/

"""


import cv2,os
import numpy as np
from PIL import Image



#%% Function Section       
   
def ForceDir(path):
    if not os.path.isdir(path):
        os.mkdir(path) 
        
 
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imageFiles=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imageFile in imageFiles:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imageFile).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imageFile)[-1].split('.')[1])
        faceSamples.append(imageNp)
        Ids.append(Id)
    return faceSamples, np.array(Ids)       


        
#%%
    
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()

path = 'Face\\'
fnTrain = 'trainner.yml'



faces, Ids = getImagesAndLabels(path)
recognizer.train(faces,Ids)
recognizer.write(fnTrain)
cv2.destroyAllWindows()

if len(faces)>0:
   print("\nTraining data saved as : \"{}\"".format(fnTrain))

