import face_recognition
import cv2
import numpy as np
import os 

# --- Real-Time Face Recognition ---

#1. Render every video frame at 1/4 resolution.
#2. Detecting faces by looking at video frames.


# OpenCV library was used to read images from the webcam.

# Webcam opening. (0 = Default Camera)
video_capture = cv2.VideoCapture(0)


#-----------------------------

# # Uploading a picture and learning how to recognize it:
# obama_image = face_recognition.load_image_file("obama.jpg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

#-----------------------------

# # Generating sequences for known face codes:
# known_face_encodings = [
#     obama_face_encoding,
#     biden_face_encoding
# ]
# known_face_names = [
#     "Barack Obama",
#     "Joe Biden"
# ]

#-----------------------------

# Variable assignment:
encodings = []
names = []
face_locations = []
process_this_frame = True

# File with training data: "data"
data = os.listdir("data")

# For each person in the file with training data:
for person in data:
    pix = os.listdir("data/" + person)

    # For each image of the current contact:
    for person_img in pix:
        # Getting the face encoding for each image of the current contact:
        face = face_recognition.load_image_file("data/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # We use this method if we want to automatically find all faces in images.
        # First the image is taken, then the positions of all the faces in the image are calculated.
        # This is an example of "Face Recognition".
                

        # If the training data contains a face:
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0] # face_encodings: List of face encodings to compare

            
            # Given a face coding list to the model, it compares them with a known face coding and obtains a Euclidean distance for each compared face. 
            # This distance gives the similarity between the faces. 

            # It always returns a list of found faces, so only the first face is retrieved. 
            # (Assuming one face per image, as a human cannot exist twice in an image.)

            
            # Adding a face encoding for the current image with "label (name)" corresponding to the training data:
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")


# face_locations = []
# # face_encodings = []
# # face_names = []
# process_this_frame = True

while True:
    # Video framing:
    ret, frame = video_capture.read()

    # Resize the frame of the video in 1/4 size for faster face detection:
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


    # Converting the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses):
    rgb_small_frame = small_frame[:, :, ::-1]


    # Processing video frames to save time and run the application faster:
    if process_this_frame:
        # Finding the positions and face encodings of the faces in the current video frame:
        face_locations = face_recognition.face_locations(rgb_small_frame) # Yüz lokasyonları
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) # Yüz kodlamaları


        face_names = []
            # Checks if the face matches the known face. 
            # This method individually checks each component of the two faces being compared 
            # and indicates whether the component at hand has changed within tolerance limits.

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding) # The training data and test data are compared.
            name = "Bilinmiyor" # If no match is found, the "Unknown" label is printed.

# if True in matches:
#     first_match_index = matches.index(True)
#     name = known_face_names[first_match_index]

            # The distance of the test image taken from the webcam to the known faces is checked.

            # Given a list of face encodings, they are compared to a known face encoding 
            # and a euclidean distance is obtained during each comparison. 
            # This distance indicates how similar the faces are.

            face_distances = face_recognition.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances) # If the distance is small, the faces are similar, but if it is large, the faces are different.
                                                         # The smallest distance is tried to be found with the "argmin" command.

            if matches[best_match_index]: # If the distance is small, the name of the person is drawn from the database and printed as a label under the rectangle.
                name = names[best_match_index]

            face_names.append(name) # The name is drawn according to the match result.
        

    process_this_frame = not process_this_frame


    # Show results:

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scaling face positions in the detected 1/4 size frame:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Drawing a rectangle around the detected face:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Printing a label with the person's name just below the face:
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # View the resulting image:
    cv2.imshow('Video', frame)

    # Press 'q' on the keyboard to exit the terminal screen.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turn off the webcam and open windows.
video_capture.release()
cv2.destroyAllWindows()