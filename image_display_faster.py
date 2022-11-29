
import cv2
import os
import numpy as np
import pickle
import argparse

import rasterfairy

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required=True,
	help="path to pkl of clusters")
ap.add_argument("-p","--photo", required=True, help="where to save large photo" )
args = vars(ap.parse_args())





movies = os.listdir(args["clusters"])
print(movies)
for movie in movies:
    movie_path = os.path.join(args["clusters"], movie)
    pkl_path  = os.path.join(movie_path, "clusters.pkl")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        print(data.head())

    data_path =  data[['x','y','image_path', 'color']]
    
    data2d =  data[['x','y']]
    
    labels = data['person'].astype(int).to_numpy()
    colormap= np.array(['aqua', 'blue', 'fuchsia', 'green', 'lime', 'maroon', 'navy', 'olive', 'purple', 'red', 'silver', 'teal', 'orange', 'DarkViolet', 'DeepSkyBlue', 'Goldenrod'])
    arrangements = rasterfairy.getRectArrangements(data2d.shape[0])
    # print(data2d)
    grid_xy, grid_shape = rasterfairy.transformPointCloud2D(data2d.to_numpy())

    print(grid_xy)

    #User defined variables
    dirname = "my_directory" #Name of the directory containing the images
    photo_path = os.path.join(args["photo"], movie + ".jpg")
    name =photo_path #"/Users/hima/Desktop/im_grid" + ".jpg" #Name of the exported file
    margin = 0
    w = grid_shape[0] # Width of the matrix (nb of images)
    h = grid_shape[1]  # Height of the matrix (nb of images)
    n = w*h
    print("w, h, n", w, h, n)

    imgs = [cv2.resize(cv2.imread(file),(128, 128)) for file in data['image_path']]

    #Define the shape of the image to be replicated (all images should have the same shape)
    img_h, img_w, img_c = imgs[0].shape
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

    for (x_i, y_i), img in zip(positions, imgs):
        
        x = int(x_i) * img_w
        y = int(y_i) * img_h
        print(img.shape, x_i, y_i, x, y, img_h, img_w)
        print(imgmatrix[y:y+img_h, x:x+img_w, :].shape)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    print(name)
    cv2.imwrite(name, resized, compression_params)