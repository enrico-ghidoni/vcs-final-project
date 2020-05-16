#import numpy as np
import cv2
from matplotlib import image
from rectification import *

#images loading 
#TODO: implement the image loading by json files
image_original = cv2.imread(r"C:\Users\simof\Pictures\Saved Pictures\gallery_2.jpg")
#image_db = image.imread(r"C:\Users\simof\Pictures\Saved Pictures\gallery_db1.jpg")
    
print("\n RECTIFICATION \n Rectification image process starting .....")

images = [image_original]
img_p = perspective_correction(perspective_correction(image_original))
images.append(img_p)
#image_original, cont = rectification_db(img_p,image_db)
#images.append(image_db)
images.append(img_p)

#show a sequence of images
for i in images:
    cv2.imshow("",i)
cv2.waitKey(0)

#process to save a common image and the corresponded rectified image
name = input("Write image name:")
print("Original image saved: ", cv2.imwrite(r"C:\Users\simof\Pictures\Saved Pictures\Rec_Image\Video Images\i{0}.png".format(name),image_original))
print("Rectified image saved: ", cv2.imwrite(r"C:\Users\simof\Pictures\Saved Pictures\Rec_Image\Rectified\i{0}.png".format(name),img_p))
show_images(images)
 
