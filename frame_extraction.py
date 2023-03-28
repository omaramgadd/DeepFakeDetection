import cv2
import os

# Set the path to the folder containing the videos
video_folder = '/Users/omaramgad/Downloads/gp/Celeb-synthesis'
# Set the path to the folder where you want to save the frames
frame_folder = '/Users/omaramgad/Downloads/gp/FakeFrames'
# Set the frame interval between frames to extract
frame_interval = 10
# Set the desired width and height of the resized frames
resized_width = 256
resized_height = 256

def extract_frames():
    # Loop through each video file in the folder
    for filename in os.listdir(video_folder):
        if not filename.endswith('.mp4'):
            continue
        print(filename)
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
                # Save the frame as an image file
                frame_filename = os.path.join(frame_folder, f'{filename}_frame_{frame_count}.jpg')
                cv2.imwrite(frame_filename, frame)

            # Update the frame count
            frame_count += 1

        # Release the video capture object
        video_capture.release()

if __name__ == "__main__":
    extract_frames()