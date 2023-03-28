import cv2 as cv
import numpy as np
import argparse
import cv2
import os
import random
import time

frame_folder = '/Users/omaramgad/Downloads/gp/FakeFrames'
results_folder = '/Users/omaramgad/Downloads/gp/FakeFaces'
proFile = 'deploy.prototxt'  # network architecture
caffeFile = 'res10_300x300_ssd_iter_140000.caffemodel'  # pre-trained weights

def extract_face():
  # load our serialized model from disk
  net = cv2.dnn.readNetFromCaffe(proFile, caffeFile)
  all_images = [f for f in os.listdir(frame_folder)]
  
  # sample for testing
  # random_image = random.sample(all_images, 100)

  # loop over frame folder
  for image_file in all_images:
    # read image and get its dimensions
    image_path = os.path.join(frame_folder,image_file)
    img = cv2.imread(image_path)

    h, w = img.shape[0], img.shape[1]
    
    # feed forward to neural network
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward() 
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
      confidence = detections[0,0,i,2]
      if confidence > 0.64:
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # crop the face
        face_img = img[startY:endY, startX:endX]
        if face_img.size != 0:
          frame_filename = os.path.join(results_folder, f'{image_file}')
          cv2.imwrite(frame_filename, face_img)

if __name__ == "__main__":
    start_time = time.time()
    extract_face()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} minutes".format(elapsed_time/60))