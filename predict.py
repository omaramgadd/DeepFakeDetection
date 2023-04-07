import os
import numpy as np
import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow_addons.optimizers import RectifiedAdam

def custom_preprocess_input(img_array):
    img_array = img_array / 255.0
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - imagenet_mean) / imagenet_std
    return img_array

def make_predictions(model_path, folder_path):
    model = load_model(model_path, custom_objects={'RectifiedAdam': RectifiedAdam})
    label = ''

    predictions = []
    for img_name in os.listdir(folder_path)[:10]:
        img_path = os.path.join(folder_path, img_name)
        img = image.load_img(img_path, target_size=(244, 244))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = custom_preprocess_input(img_array)
        prediction = model.predict(img_array)
        print(prediction)
        predictions.append(prediction[0][0])

    prob = sum(predictions)/len(predictions)

    if prob > 0.5:
      label = 'Real'
    else:
      label = 'Fake'
      prob = 100 - prob
    return label, int(prob)

# Example usage
model_path = '/content/drive/MyDrive/trained_models/model_epoch_03.h5'
folder_path = '/content/drive/MyDrive/test gp data/Fake'

label, prob = make_predictions(model_path, folder_path)
print("Label:", label, "Probability:", prob)