
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





movies = os.listdir(args["clusters"])
print(movies)
total_points = []
total_imgs = []
for movie in movies:
    pkl_path = os.path.join(args["clusters"], movie)
  
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        # print(data.head())

    data_path =  data[['x','y','image_path', 'colors']]
    data2d = []
    for c in data['color_individual']:
        if(len(c) > 0):
            max_c = max(c, key=lambda x:x['color_percentage'])['color']
            data2d.append(np.array(colorsys.rgb_to_hsv(max_c[0] / 255.0, max_c[1] / 255., max_c[2] / 255.)).reshape(1, 3))
        else:
            data2d.append(np.array(colorsys.rgb_to_hsv(0,0,0)).reshape(1, 3))

    # data2d =  [np.array(colorsys.rgb_to_hsv(c[0] / 255.0, c[1] / 255., c[2] / 255.)).reshape(1, 3) for c in data_path['colors']]
    imgs = [cv2.resize(cv2.imread(file),(128, 128)) for file in data['image_path']]
    total_points = total_points + data2d
    total_imgs = total_imgs +imgs
# print(total_points)
total_points = np.concatenate(total_points, axis = 0) * 255
print(total_points.shape)
total_points = total_points[:,1:]

indices = np.where(np.any(total_points != 0, axis=1))
print(indices)
total_points = total_points[indices[0]]
total_imgs = np.array(total_imgs)[indices[0]]


arrangements = rasterfairy.getRectArrangements(total_points.shape[0])
# print(total_points)
grid_xy, grid_shape = rasterfairy.transformPointCloud2D(total_points)
# print(grid_xy)

#User defined variables

photo_path = os.path.join(args["photo"], movie + ".jpg")
name =photo_path 
margin = 0
w = grid_shape[0] # Width of the matrix (nb of images)
h = grid_shape[1]  # Height of the matrix (nb of images)
n = w*h
print("w, h, n", w, h, n)




#Define the shape of the image to be replicated (all images should have the same shape)
img_h, img_w, img_c = total_imgs[0].shape
print(img_h, img_w, img_c)

#Define the margins in x and y directions
m_x = margin
m_y = margin

#Size of the full size image
mat_x = img_w * w
mat_y = img_h * h
print("mat_y,mat_x ", mat_y,  mat_x,)
#Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
imgmatrix.fill(0)

#Prepare an iterable with the right dimensions
positions = grid_xy

#screen out the columns

for (x_i, y_i), img in zip(positions, total_imgs):
    
    x = int(x_i) * img_w
    y = int(y_i) * img_h
    # print(img.shape, x_i, y_i, x, y, img_h, img_w)
    # print(imgmatrix[y:y+img_h, x:x+img_w, :].shape)
    imgmatrix[y:y+img_h, x:x+img_w, :] = img

resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
print(name)
cv2.imwrite(name, resized, compression_params)