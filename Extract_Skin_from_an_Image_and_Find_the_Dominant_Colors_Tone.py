import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import os
import csv
import argparse
import pprint
import pandas as pd
import pickle
from matplotlib import pyplot as plt

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 60], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


"""## Section Two.4.2 : Putting it All together: Pretty Print
The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()



ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to csv of facial encodings")
ap.add_argument("-co", '--colors', required=True,
    help="path to embedding, color, and size data")
args = vars(ap.parse_args())


movies = os.listdir(args["encodings"])[1:]
print(movies)


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
    faces_df = pd.DataFrame()
    faces_df['image_path'] = file_names
    faces_df['encodings'] = encodings
    # print(faces_df)
    image_sizes = []
    colors = []
    for (i, file) in enumerate(file_names):
        # Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")
        image = cv2.imread(file)
        image_sizes.append(image.shape)
   
        # Resize image to a width of 250
        image = cv2.resize(image, (250, 250))

        # Apply Skin Mask
        skin = extractSkin(image)

        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
        dominantColors = extractDominantColor(skin, hasThresholding=True)
        # prety_print_data(dominantColors)
        print("cur image ", i)
        colors.append(dominantColors)
    faces_df['colors'] = colors
    faces_df['image_sizes'] = image_sizes
    pkl_path = os.path.join(args["colors"], movie + ".pkl")
    pickle.dump(faces_df, open(pkl_path, 'wb'))
