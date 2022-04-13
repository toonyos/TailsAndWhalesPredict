import streamlit as st
import matplotlib.pyplot as plt
import requests

URI = "" # A redefinir selon GCP bucket


def print_plot(result):

    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(10,3))

    ax.bar(['whale', 'dolphin', 'beluga'], result["prediction"])
    ax.set_xlabel('Class')
    ax.set_ylabel('Prediction')

    st.pyplot(fig)


#def print_metric(result):


st.markdown("""
    # TAILS AND WHALES MAKE HAPPYWHALE
    [People behind the dataset](https://www.happywhale.com/home)
    - bullet points

    ## PART1: GET CONFIDENT WITH THE MODEL

    **bold** or *italic* text with [links](https://www.happywhale.com/home) and:
    - bullet points
""")


col1, col2 = st.columns([2, 2])

col1.subheader("A Whale")
with col1:
    st.image('image/1c342cdd745998.jpg')
    bouton_1 = st.button(label='Predict_1')

col2.subheader("Prediction")
with col2:
    if bouton_1:
        #model_predict('1c342cdd745998.jpg')
        result = requests.get(URI+'1c342cdd745998.jpg')
        print_plot(result)



col3, col4 = st.columns([2, 2])

col3.subheader("A Dolphin")
with col3:
    st.image('image/1c814b03d3e28d.jpg')
    bouton_2 = st.button(label='Predict_2')

col4.subheader("Prediction")
with col4:
    if bouton_2:
        #model_predict('1c814b03d3e28d.jpg')
        result = requests.get(URI+'1c814b03d3e28d.jpg')
        print_plot(result)


col5, col6 = st.columns([2, 2])

col5.subheader("A Beluga")
with col5:
    st.image('image/1deb81035cb1d5.jpg')
    bouton_3 = st.button(label='Predict_3')

col6.subheader("Prediction")
with col6:
    if bouton_3:
        #model_predict('1deb81035cb1d5.jpg')
        result = requests.get(URI+'1deb81035cb1d5.jpg')
        print_plot(result)


st.markdown("""

    ## PART2: GET IMPRESSED WITH THE MODEL

    **bold** or *italic* text with [links](http://github.com/streamlit) and:
    - bullet points
""")

col7, col8, col9 = st.columns(3)

with col7:
    st.header("What")
    with st.expander("See Picture - Cropped"):
        st.image('image/crop1.jpg')
        bouton_4 = st.button(label='Predict_5')
        if bouton_4:
            #model_predict('/croped_img/crop1.jpg')
            result = requests.get(URI+'crop1.jpg')

with col8:
    st.header("is this")
    with st.expander("See Picture - Uncropped"):
        st.image('image/uncropped.jpg')
        bouton_5 = st.button(label='Predict_6')
        if bouton_5:
            #model_predict('/croped_img/uncropped.jpg')
            result = requests.get(URI+'uncropped.jpg')

with col9:
    st.header("class?")
    with st.expander("See Picture - Background"):
        st.image('image/crop2.jpg')
        bouton_6 = st.button(label='Predict_7')
        if bouton_6:
            #model_predict('/croped_img/crop2.jpg')
            result = requests.get(URI+'crop2.jpg')
