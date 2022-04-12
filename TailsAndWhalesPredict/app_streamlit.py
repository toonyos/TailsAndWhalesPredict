import streamlit as st
import numpy as np
import cv2
import matplotlib.image as mpimg
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input



# def predict_class(image):
#     classifier_model = tf.keras.models.load_model(path)
#     shape = ((64, 64, 3))
#     model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
#     test_image = image.resize((64, 64))
#     test_image = preprocessing.image.img_to_array(test_image)
#     #test_image = test_image/255.0
#     normalize_data_train(test_image)


def load_trained_model(model):
  path = "../raw_data/"
  #print (path + model)
  reconstructed_model = load_model(path + model + "/")
  return reconstructed_model

reconstructed_model = load_trained_model('model_7')

#Title of the app
st.title("HappyWhale : app to classify whale, dolphin and beluga")


img_1 = Image.open('../raw_data/image/1c342cdd745998.jpg')

print(type(img_1))
st.title("Here is the image selected")

st.image(img_1)
print(f"img_1 ----------------{type(img_1)}")

bouton1 = st.button(label='Predict')

if bouton1:
    img = mpimg.imread('../raw_data/image/1c342cdd745998.jpg')
    print(f"A ----------------{type(img)}")
    img = cv2.resize(img, dsize=(64, 64), interpolation= cv2.INTER_LINEAR)

    X = np.array(img)
    print(f"B ----------------{X}")
    X = preprocess_input(X)
    print(f"C ----------------{X}")
    X = np.expand_dims(X, axis=0)

    print(f"----------------{X.shape}")

    prediction = reconstructed_model.predict(X)
    print(f"----------------{prediction}")
