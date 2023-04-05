# Graduation Project: Deepfake Detection using EfficientNet

This repository contains the source code for a deepfake detection system based on the EfficientNet architecture. The model is trained to classify images as real or fake using transfer learning. We used the Celeb-DF-v2 dataset for training and extracted approximately 250,000 frames from the videos. The project includes frame extraction, face detection, and the deep learning model for classification. The model has been pretrained on ImageNet and fine-tuned on the face dataset. The project was carried out using Google Colab.

## Table of Contents

1. [Celeb-DF-v2](#celeb-df-v2)
2. [Algorithms and Models](#algorithms-and-models)
3. [File Structure](#file-structure)
4. [Training](#training)
5. [Testing and Results](#testing-and-results)
6. [Usage](#usage)

## Celeb-DF-v2

![demo](https://user-images.githubusercontent.com/57623082/230083869-b7b1bc85-1758-43f1-a79e-ec0dc6227b6d.png)

## Algorithms and Models

1. **Frame extraction**: OpenCV is used to extract frames from video files.
2. **Face detection**: A pre-trained Caffe model (ResNet-10) is used for face detection in the extracted frames.
3. **Classification**: An EfficientNet-B4 model is used as the base model for classification. The model is fine-tuned on the face dataset to classify images as real or fake.

## File Structure

### frame_extraction.py

This script extracts frames from video files in a specified folder at a given interval. It resizes the frames to the desired dimensions and saves them to a separate folder.

- Main function: `extract_frames()`
  - Loops through video files, extracts frames, resizes them, and saves them to a specified folder.

<img width="1266" alt="Screenshot 2023-04-05 at 3 39 28 PM" src="https://user-images.githubusercontent.com/57623082/230098402-42195e07-5be2-41c5-930d-69ebc86c00f0.png">
<img width="1266" alt="Screenshot 2023-04-05 at 3 40 17 PM" src="https://user-images.githubusercontent.com/57623082/230098465-bb23ac72-c0ec-4afe-93a7-8cb1735c7fbe.png">

### face_detection.py

This script detects faces in the frames extracted using the `frame_extraction.py` script. It uses a pre-trained Caffe model (ResNet-10) for face detection and saves the cropped faces to a separate folder.

- Main function : `extract_face()`
  - Loops through frame images, detects faces using the pre-trained Caffe model, and saves the cropped faces to a specified folder.

![image](https://user-images.githubusercontent.com/57623082/230098712-71bc1073-708e-49ee-9e23-13956d76068e.png)


### model.py

This script contains the deep learning model for classification using the EfficientNet-B4 architecture. It also contains helper functions for data processing, model training, evaluation, and saving.

- Functions:
  - `display_sample_images(data, data_dir, num_samples=10)`
    - Displays sample images from dataset
  - `create_dataframe_from_directory(data_dir)`
    - Creates a DataFrame from the image files in a directory.
  - `create_generators(train_data, val_data, test_data, batch_size=generator_batch)`
    - Splits the data into training, validation, and test sets.
    - Creates data generators for training, validation, and testing.
  - `create_model(input_shape=(244, 244, 3))`
    - Creates the EfficientNet-B4 model for classification.
  - `train_model(model, train_generator, val_generator, epochs=EPOCHS, batch_size=BATCH_SIZE)`
    - Trains the model on the face dataset.
  - `evaluate_model(model, test_generator, batch_size=BATCH_SIZE)` 
    - Evaluates the model on the test dataset.
  - `save_model(model, file_name)`
    - Saves the trained model to a specified file.
<img width="526" alt="Screenshot 2023-04-05 at 3 45 35 PM" src="https://user-images.githubusercontent.com/57623082/230099698-da101110-e932-4d66-a560-3623ebc2df65.png">
<img width="425" alt="Screenshot 2023-04-05 at 3 45 25 PM" src="https://user-images.githubusercontent.com/57623082/230099799-0457e81e-7bae-498e-9fbb-bc1c719efe82.png">


## Results

<img width="913" alt="image" src="https://user-images.githubusercontent.com/57623082/230087882-c3993838-17f1-4278-9703-d8fb39e0fe81.png">

After training the model on the extracted frames and evaluating it on the test set, we obtained impressive results, demonstrating the effectiveness of the EfficientNet architecture in deepfake detection tasks. The model was able to achieve the following performance metrics:

- Loss: 0.1
- Accuracy: 99%
- AUC: 99%

These results shows that our deepfake detection system is highly effective and can be utilized for real-world applications.

The Model produces similar results to the adapted paper [Das et al., 2021].

<img width="654" alt="Screenshot 2023-04-05 at 3 27 50 PM" src="https://user-images.githubusercontent.com/57623082/230096297-bbafb570-b6c3-41e3-b266-2c91e4cb8391.png">

## Usage

1. Run `frame_extraction.py` to extract frames from video files.
2. Run `face_detection.py` to detect faces in the extracted frames.
3. Run `model.py` to train and evaluate the EfficientNet-B4 model on the face dataset.

Note: The paths in the scripts need to be adjusted according to your directory structure.

## References

[Das et al., 2021] Das, S., Seferbekov, S., Datta, A., Islam, M. S., & Amin, M. R. (2021). Towards Solving the DeepFake Problem: An Analysis on Improving DeepFake Detection Using Dynamic Face Augmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops (pp. 3776-3785).
