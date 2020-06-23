#import numpy as np
import cv2
from matplotlib import image
from rectification import *
import json
import sys

"""
Parameters: 
    1 JSON file
    2 video
Return: Rectified images from the video
"""

def saveImg(image, img_pers):
    """
    Function used for Testing. Useful to save an image in a folder.
    Keyword arguments: original image, rectified image
    Return: None
    """
    #process to save a common image and the corresponded rectified image
    name = input("Write image name:")
    print("Original image saved: ", cv2.imwrite(r"C:\Users\simof\Pictures\Saved Pictures\Rec_Image\Video Images\i{0}.png".format(name),image))
    print("Rectified image saved: ", cv2.imwrite(r"C:\Users\simof\Pictures\Saved Pictures\Rec_Image\Rectified\i{0}.png".format(name),img_pers))

print("RECTIFICATION \nRectification image process starting .....")
# check if the number of arguments are good
if not len(sys.argv) == 3:
    print("ERROR 4: no arguments")
# list of arguments
arg_list = sys.argv
#inizialise all the structured list
list_coordinates = []
frames = []
images = []
frame_count = 0
try:
    # JSON file opening
    f = open(arg_list[1],)
    json_file = json.load(f)

    for times in json_file:
        list_coordinates.append(json_file[times])
    onein = list(json_file.keys())[0]
except Exception:
    print("ERROR 5: first argument must be the json file")

try:
    # video capture and creation of every frame
    cap = cv2.VideoCapture(arg_list[2])
    print(f"Video lecture {arg_list[2]}")
    while cap.isOpened():
        ret, fr = cap.read()
        frame_count += 1
        if frame_count % int(onein) == 0 and ret:
            frames.append(fr)
        if not ret: 
            break
    cap.release()
except Exception:
    print("ERROR 7: image not valid")

print(f"Frames number: {len(frames)}")
#image_db = image.imread(r"C:\Users\simof\Pictures\Saved Pictures\gallery_db1.jpg")
frame_count = 0
try:
    # Main process and rectification
    print("Inizialization images ...")
    for image_original in frames:

        coordinates = list_coordinates[frame_count]
        print(f"FRAME: {coordinates}")
        frame_count +=1
        for coordinate in coordinates:
            try:
                x,y,w,h = coordinate    
                print(f"COORDINATE JSON: {coordinate}")        
                image_crop = image_original[y:y+h,x:x+w,:]
                img_pers = perspective_correction(perspective_correction(image_crop)) #double rectification to be more sure !

                #image_original, cont = rectification_db(img_p,image_db)
                #images.append(image_db)
                images.append(img_pers) #OUTPUT: list of rectified images

            except Exception:
                print(f"ERROR 8: Error at coordinate {coordinate}")
    # TESTING:show the sequence of images
    for i in images:
        cv2.imshow(str(i),i)
        cv2.waitKey(0)
        
except Exception:
    print("ERROR 6: second argument must be a video")
