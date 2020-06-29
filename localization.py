import numpy as np
import cv2
import json
import sys
import csv



class PeopleLocalization(object):

    def __init__(self, path_csv):
        self.room = 0
        self.path_data = path_csv

    def openCSV(self, path):    
        try:
            # CSV file opening
            f = open(path,)
            csv_file = csv.reader(f, delimiter=',')
            return csv_file
        except Exception:
            print("ERROR 3: cannot open csv file")
            return None

    def drawBbox(self, frame_num, bbox, img, room):
        coordinates = bbox[frame_num]
        for coordinate in coordinates:
                x,y,w,h = coordinate
                cv2.rectangle(img, (x, y), (x+w, y+h), (255,100,100), 3)
                cv2.putText(img, f"Room {room}",(x - 10, y - 10),cv2.FONT_ITALIC,1,(255,100,100),2)

    def assign_rooms(self, frame_num, image_name):

        try:
            csv_file = self.openCSV(self.path_data)
            for image in csv_file:
                if image[0] == image_name:
                    room = image[2]

            return int(room)    
        except Exception:
            return 0
        
    def people_localization(self, frame, frame_num, bbox, image_name):

        room = self.assign_rooms(frame_num,image_name)
        if room:
            print(f"Find paint \"{image_name}\" in room {room} with {len(bbox[frame_num])} people")
            self.drawBbox(frame_num,bbox,frame, room)
        else:
            print("No room")
