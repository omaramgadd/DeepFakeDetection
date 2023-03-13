import cv2
import os
import dlib
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


# Define input and output folders
input_folder = '/Users/omaramgad/Downloads/archive/test'
output_folder = '/Users/omaramgad/Downloads/archive/test_2'
output_txt_folder = '/Users/omaramgad/Downloads/archive/test_2'
# Initialize face detector
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# Get a list of all files in the input folder
file_list = os.listdir(input_folder)

# Randomly select 100 files from the list
random_files = random.sample(file_list, 1)

counter = 0
# Loop over the selected files
for filename in os.listdir(input_folder):
    print(counter)
    if filename.endswith('.jpg'):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Detect faces using the dlib CNN face detector
        detections = detector(image, 1)

        # Loop over the face detections
        for i, detection in enumerate(detections):
            # Get the bounding box coordinates
            x1, y1, x2, y2 = detection.rect.left(), detection.rect.top(), detection.rect.right(), detection.rect.bottom()

            # Crop the face from the image
            face = image[y1-13:y2+13, x1-3:x2+3]

            # Save the cropped face to a new image file
            output_path = os.path.join(output_folder, f'{filename}.jpg')
            cv2.imwrite(output_path, face)

            # Print the confidence
            confidence = detection.confidence
            print(f"confidence: {confidence}")

            # Save the bounding box coordinates to a text file
            box_path = os.path.join(output_txt_folder, f'{filename}.txt')
            with open(box_path, 'w') as f:
                f.write(f'{x1} {y1} {x2} {y2}\n')

    counter += 1
