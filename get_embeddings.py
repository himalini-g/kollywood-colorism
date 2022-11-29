import face_recognition
import os
import csv
import time
import argparse
start_time = time.time()




# for movie name in footage
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--faces", required=True,
	help="path to faces")
ap.add_argument("-e", "--embeddings", required=True,
	help="path to csv of facial embeddings")

args = vars(ap.parse_args())



faces_dir = path = args['faces']
print("faces", faces_dir)
movies = sorted(os.listdir(faces_dir))
print(movies)
for movie in movies:
    print("movie", movie)

    embeddings = []

    movie_faces_path = os.path.join(faces_dir, movie)
    for face in sorted(os.listdir(movie_faces_path)):

        face_path =  os.path.join(movie_faces_path, face)
        print(face)

        face_image = face_recognition.load_image_file(face_path)

        face_encoding = face_recognition.face_encodings(face_image, [[0, 0, face_image.shape[0], face_image.shape[1]]])[0]
        # print(face_encoding)
        embeddings.append([face,face_encoding])
    
    csv_file_path = os.path.join(args['embeddings'], movie + ".csv")
    header = ['face_path', 'embedding_vector']
    print("--- %s seconds ---" % (time.time() - start_time))

    with open(csv_file_path, 'w') as open_csv:
        writer = csv.writer(open_csv)
        writer.writerow(header)
        writer.writerows(embeddings)
   
    print(csv_file_path)

