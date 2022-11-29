from PIL import Image
import sys
import face_recognition
import os
import csv
import time
start_time = time.time()




# for movie name in footage
parent_dir = sys.argv[1:]
print(sys.argv[1:])
first, last = int(sys.argv[2]), int(sys.argv[3])

frames_dir = path = os.path.join(parent_dir[0], "frames/")
print("frames_dir", frames_dir)
movies = sorted(os.listdir(frames_dir))[first:last]
print(movies)
for movie in movies:
    print("movie", movie)
    faces_path =  os.path.join(os.path.join(parent_dir[0], "faces"), movie)
    embeddings_path =  os.path.join(os.path.join(parent_dir[0], "embeddings"), movie)
    os.mkdir(faces_path)
    os.mkdir(embeddings_path)
    embeddings = []

    
   
    movie_frames_path = os.path.join(frames_dir, movie)
    for frame in sorted(os.listdir(movie_frames_path)):

        frame_path =  os.path.join(movie_frames_path, frame)
        print(frame)

        image = face_recognition.load_image_file(frame_path)
        face_locations = face_recognition.face_locations(image)
        

        print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for (i, face_location) in enumerate(face_locations):

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            face_image = face_image.copy()
            pil_image = Image.fromarray(face_image)
            # pil_image.show()
            # face_image = face_image.resize(128, 128)
            

            face_save_path = faces_path + "/" +  frame[:-4]  + "_face_" + str(i) + '.jpg'
            print(face_image.shape)
            face_encoding = face_recognition.face_encodings(face_image, [[0, 0, face_image.shape[0], face_image.shape[1]]])[0]
            # print(face_encoding)
            embeddings.append([face_save_path,face_encoding])
       
            pil_image.save(face_save_path )
    
    csv_file_path = os.path.join(embeddings_path, "embeddings.csv")
    header = ['face_path', 'embedding_vector']
    print("--- %s seconds ---" % (time.time() - start_time))

    with open(csv_file_path, 'w') as open_csv:
        writer = csv.writer(open_csv)
        writer.writerow(header)
        writer.writerows(embeddings)
    print(faces_path)
    print(embeddings_path)

