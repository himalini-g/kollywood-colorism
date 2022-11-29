
import cv2
import os
import numpy as np
import pickle
import argparse
import rasterfairy
import colorsys

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required=True,
	help="path to pkl of clusters")
ap.add_argument("-p","--photo", required=True, help="where to save large photo" )
args = vars(ap.parse_args())


dimension = 128


movies = os.listdir(args["clusters"])
print(movies)
total_points = []
total_imgs = []
for movie in movies:
    pkl_path = os.path.join(args["clusters"], movie)
  
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        print(data.columns)
    data2d = []
    for c in data['color_individual']:
        if(len(c) > 0):
            max_c = max(c, key=lambda x:x['color_percentage'])['color']
            
            data2d.append(np.array(colorsys.rgb_to_hsv(max_c[0] / 255.0, max_c[1] / 255., max_c[2] / 255.)).reshape(1, 3))
            # print(data2d[-1])
        else:
            data2d.append(np.array(colorsys.rgb_to_hsv(0,0,0)).reshape(1, 3))
            # print(data2d[-1])

    imgs = [cv2.resize(cv2.imread(file),(dimension, dimension)) for file in data['image_path']]
    total_points = total_points + data2d

    total_imgs = total_imgs +imgs
# print(total_points)
total_points = np.concatenate(total_points, axis = 0) * 255
# print(total_points.shape)
video_name = '/Users/hima/Desktop/video.avi'
total_points = total_points[:,2]
total_points_inds = total_points.argsort()
total_imgs = np.array(total_imgs)
print(total_imgs.shape)
# print(total_points_inds)
# print(total_points.shape, total_imgs.shape)
# points_imgs = np.concatenate((total_points, total_imgs), axis=0)
# print(total_points.shape, total_imgs.shape, points_imgs.shape)

print(total_imgs.shape)
total_imgs[total_points.argsort()]
video = cv2.VideoWriter(video_name, 0, 1, (dimension,dimension))

for image in total_imgs:
    video.write(image)

cv2.destroyAllWindows()
video.release()
