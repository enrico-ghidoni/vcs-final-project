#import numpy as np
import cv2
from matplotlib import image
from rectification import *

#images loading 
#TODO: implement the image loading by json files
image_original = cv2.imread(r"Image you want")
    
print("\n RECTIFICATION \n Rectification image process starting .....")

images = [image_original]
img_p = perspective_correction(perspective_correction(image_original))
images.append(img_p)

#considering an image matching from the paintings DB and the corrisponded rectified image
# from previous step
image_db = image.imread(r"Corrisponent image from DB")
image_original, cont = rectification_db(img_p,image_db)
images.append(image_db)
images.append(cont)

#show all the images
for i in images:
    cv2.imshow("",i)
cv2.waitKey(0)

#process to save a common image and the corresponded rectified image
# substitue Folder name with your desired repository
name = input("Write image name:") # define the name for both the images to save
print("Original image saved: ", cv2.imwrite(r"Folder\i{0}.png".format(name),image_original))
print("Rectified image saved: ", cv2.imwrite(r"Folder\i{0}.png".format(name),img_p))
