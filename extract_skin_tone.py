from PIL import Image, ImageDraw
import argparse
import csv
import face_recognition
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to csv of facial encodings")
args = vars(ap.parse_args())


movies = os.listdir(args["encodings"])
print(movies)
movies = ['Aelay']
# Constants for finding range of skin color in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

for movie in movies:
    file_names, encodings, header = [], [], None
    movie_path = os.path.join(args["encodings"], movie)
    csv_path  = os.path.join(movie_path, "embeddings.csv")
    with open(csv_path) as f:
        data =  csv.reader(f, delimiter=',')
        for i, row in enumerate(data):
            if(i == 0):
                header = row
                continue
            file_names.append(row[0])
            raw_encoding = row[1][1:-1].replace('\n', '')
            encodings.append([float(i) for i in raw_encoding.split(' ') if len(i) > 0])
            if(len(encodings[-1]) != 128):
                print("problem!!!")
    for file in file_names:
        image = cv2.imread(file)
        imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)

        # Find region with skin tone in YCrCb image
        skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

        # Do contour detection on skin region
        contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contour on the source image
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 1000:
                cv2.drawContours(image, contours, i, (0, 255, 0), 3)

        # Display the source image
        cv2.imshow('Camera Output',image)

        # Check for user input to close program
        keyPressed = cv2.waitKey(1) # wait 1 milisecond in each iteration of while loop

        

