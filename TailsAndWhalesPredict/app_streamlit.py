import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
# ATTENTION Les path sont à redéfinir suivant l'adresse avec gcp

def load_trained_model(model):
  path = "/home/jeh/code/asoggia/tails-and-whales/scripts/"
  #print (path + model)
  reconstructed_model = load_model(path + model + "/")
  return reconstructed_model

reconstructed_model = load_trained_model('model_7')

#Title of the app
st.title("HappyWhale : app to classify whale, dolphin and beluga")


img_1 = Image.open('images/crop1.jpg')
img_2 = Image.open('images/crop2.jpg')
img_3 = Image.open('images/crop3.jpg')
img_4 = Image.open('images/crop4.jpg')


st.title("Here is the image selected")

st.image(img_1)
bouton_1 = st.button(label='Predict_1')

st.image(img_2)
bouton_2 = st.button(label='Predict_2')

st.image(img_3)
bouton_3 = st.button(label='Predict_3')

st.image(img_4)
bouton_4 = st.button(label='Predict_4')


def button_predict(name_img):
    # A redéfinir car le process du predict doit se faire dans gcp via call api
    # Nous devons simplement faire un requests
    img = mpimg.imread('images/' + name_img)
    img = cv2.resize(img, dsize=(64, 64), interpolation= cv2.INTER_LINEAR)

    X = np.array(img)
    X = preprocess_input(X)
    X = np.expand_dims(X, axis=0)
    print(X.shape)


    prediction = reconstructed_model.predict(X)
    #chart_data = pd.DataFrame(prediction, columns = ['whale', 'dolphin', 'beluga'])
    #st.bar_chart(chart_data)
    print(prediction)

    fig, ax = plt.subplots()
    ax.bar(['whale', 'dolphin', 'beluga'], prediction[0])

    #ax.title('Class Prediction')
    ax.set_xlabel('Class')
    ax.set_ylabel('')
    st.pyplot(fig)


if bouton_1:
   button_predict('crop1.jpg')

if bouton_2:
   button_predict('crop2.jpg')

if bouton_3:
   button_predict('crop3.jpg')

if bouton_4:
   button_predict('crop4.jpg')
