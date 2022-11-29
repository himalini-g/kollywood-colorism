# import the necessary packages
from PIL import Image
from sklearn.cluster import DBSCAN
import umap
import numpy as np
import argparse
import csv
import pandas as pd
import pickle
import os
from sklearn.cluster import DBSCAN


from io import BytesIO
import base64
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral11
# from sklearn.manifold import TSNE


reducer = umap.UMAP()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to csv of facial encodings")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of parallel jobs to run (-1 will use all CPUs)")
ap.add_argument("-c","--clusters", required=True, help="where to save cluster data" )
ap.add_argument("-f", "--first", required=True, help="first index")
ap.add_argument("-l", "--last", required=True, help="last index")
args = vars(ap.parse_args())


movies = os.listdir(args["encodings"])
print("all movies len = ", len(movies))

movies = sorted(os.listdir(movies))[args["first"]:args["last"]]
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

    encodings = np.array(encodings)

    embedding = reducer.fit_transform(encodings)
    # embedding = TSNE(perplexity=50, n_iter = 5000).fit_transform(encodings)
    print(embedding.shape)

    def embeddable_image(file):
        data = np.asarray(Image.open(file))
        # img_data = 255 - 15 * data.astype(np.uint8)
        image = Image.fromarray(data).resize((64, 64))
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

    db = DBSCAN(
        min_samples=10
    ).fit(embedding)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters: %d" % n_clusters_)

    unique_labels = set(labels)

    output_file("graph-a.html")

    faces_df = pd.DataFrame(embedding, columns=('x', 'y'))
    faces_df['person'] = [str(x) for x in labels]
    faces_df['image'] = list(map(embeddable_image, file_names))
    faces_df['image_path'] = file_names
    color_mapping = CategoricalColorMapper(factors=[str(l) for l in  list(unique_labels)], palette=Spectral11)

    datasource = ColumnDataSource(faces_df)

    plot_figure = figure(
        title='UMAP projection of the Digits dataset',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Digit:</span>
            <span style='font-size: 18px'>@person</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='person', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)
    print(faces_df)

    delete = input()
    faces_df2 = faces_df[faces_df['person'] != delete]
    faces_df3 = faces_df2[faces_df2['person'] != '-1']
    pkl_path = os.path.join(args["clusters"], movie + ".pkl")
    pickle.dump(faces_df3, open(pkl_path, 'wb'))