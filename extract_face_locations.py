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
        image = face_recognition.load_image_file(file)
        face_landmarks = face_recognition.face_landmarks(image)
        if(len(face_landmarks) > 0):
            face_landmarks = face_landmarks[0]

            pil_image = Image.fromarray(image)
            
            d = ImageDraw.Draw(pil_image, 'RGBA')
            print('nose_bridge', face_landmarks['nose_bridge'])
            print('nose_tip',face_landmarks['nose_tip'])

            d.polygon(face_landmarks['nose_tip'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['nose_tip'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['nose_tip'], fill=(68, 54, 39, 150), width=1)
            d.line(face_landmarks['nose_tip'], fill=(68, 54, 39, 150), width=1)

            d.polygon(face_landmarks['nose_bridge'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['nose_bridge'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['nose_bridge'], fill=(68, 54, 39, 150), width=1)
            d.line(face_landmarks['nose_bridge'], fill=(68, 54, 39, 150), width=1)

            nose_bridge = face_landmarks['nose_bridge']

            points1 = np.linspace(nose_bridge[0],nose_bridge[1],10)
            points2 = np.linspace(nose_bridge[1],nose_bridge[2],10)
            points3 = np.linspace(nose_bridge[2],nose_bridge[3],10)

            points = np.unique(np.concatenate((points1, points2, points3), axis=0).astype('int32'),  axis=0)

            colors = image[points[:,0], points[:,1], :].astype('uint8')
           

            m, n = 4, 6
            indices = np.random.randint(0, len(colors), size=(4, 6))
            io.imshow(colors[indices])
            pil_image.show()
            plt.show()
