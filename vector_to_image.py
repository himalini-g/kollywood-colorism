import pickle
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import rasterfairy
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--clusters", required=True,
	help="path to pkl of clusters")
# ap.add_argument("-p","--photo", required=True, help="where to save large photo" )
args = vars(ap.parse_args())

with open(args['clusters'], 'rb') as f:
    data = pickle.load(f)
    print(data.head())



# side = int(data.shape[0]**0.5) + 1

data_path =  data[['x','y','image_path']]
data2d =  data[['x','y']]
labels = data['person'].astype(int).to_numpy()
colormap= np.array(['aqua', 'blue', 'fuchsia', 'green', 'lime', 'maroon', 'navy', 'olive', 'purple', 'red', 'silver', 'teal', 'orange', 'DarkViolet', 'DeepSkyBlue', 'Goldenrod'])
arrangements = rasterfairy.getRectArrangements(data2d.shape[0])
print(data2d)
grid_xy = rasterfairy.transformPointCloud2D(data2d.to_numpy())[0]
print(grid_xy)
# fig = plt.figure(figsize=(10.0,10.0))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_facecolor('black')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# ax.autoscale_view(True,True,True)
# ax.invert_yaxis()
# ax.scatter(grid_xy[:,0],grid_xy[:,1], c = colormap[labels] ,edgecolors='none',marker='s',s=9)    
# plt.show()


fig = plt.figure(figsize=(grid_xy.shape[0], grid_xy.shape[1]))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(grid_xy.shape[0], grid_xy.shape[1]),  # creates 2x2 grid of axes
                #  axes_pad=0.1,  # pad between axes in inch.
                 )
def img_reshape(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((128,128))
    img = np.asarray(img)
    print(path)
    return img
images = [img_reshape(path) for path in data['image_path'] ]
for (i, image) in enumerate(images):
    index = grid_xy[i]
    oneDindex = (index[0] * grid_xy.shape[0]) + index[1]
    grid[oneDindex].imshow(image)

# for ax, im in zip(grid, [images]):
#     # Iterating over the grid returns the Axes.
#     ax.imshow(im)

plt.show()