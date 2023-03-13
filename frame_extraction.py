import cv2
import os

# Set the path to the folder containing the videos
video_folder = '/Users/omaramgad/Downloads/archive/manipulated_sequences/Deepfakes/c23/videos'

# Set the path to the folder where you want to save the frames
frame_folder = '/Users/omaramgad/Downloads/archive/manipulated_frames/Deepfakes'

# Set the frame interval between frames to extract
frame_interval = 10

# Set the desired width and height of the resized frames
resized_width = 256
resized_height = 256

run

# Loop through each video file in the folder
for filename in os.listdir(video_folder):
    if not filename.endswith('.mp4'):
        continue
    
    # Open the video file
    video_capture = cv2.VideoCapture(os.path.join(video_folder, filename))

    # Set the frame count
    frame_count = 0

    # Loop through the frames and extract them
    while True:
        # Read the next frame
        ret, frame = video_capture.read()

        # If the end of the video has been reached, break out of the loop
        if not ret:
            break

        # Check if the current frame index is a multiple of the frame interval
        if frame_count % frame_interval == 0:
            # Resize the frame to the desired width and height
            resized_frame = cv2.resize(frame, (resized_width, resized_height))
            
            # Save the frame as an image file
            frame_filename = os.path.join(frame_folder, f'{filename}_frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, resized_frame)

        # Update the frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()