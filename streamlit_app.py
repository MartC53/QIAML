import streamlit as st
from PIL import Image
import numpy as np
from model import loadmodel, predict


st.title("QIAML Test Results")

file = st.file_uploader(
    "Upload the image that you are wanting to classify here, either browse or drag ", type=("jpg"))

model = loadmodel()
photo_names = ["<select>", "30.1.jpg", "100.1.jpg", "300.1.jpg",
               "1000.1.jpg", "3000.1.jpg", "10000.1.jpg", "100000.1.jpg"]
class_names = ['High', 'Low', 'Medium']
photo_dir = "Datasets/cropped_jpgs/A/"
photo_list = st.sidebar.selectbox('Select an image:', list(photo_names))

if file:
    img = Image.open(file)
    st.title("Image Selected")
    resized_image = img.resize((200, 200))
    st.image(resized_image)
    predictions = predict(img, model)
    predictions_label = np.argmax(predictions[0])
    predictions_label = class_names[predictions_label]
    st.title("Predicted Result")
    st.header(predictions_label)

if photo_list == "<select>":
    st.title("Select or Upload an image")
else:
    img = Image.open("{}{}".format(photo_dir, photo_list))
    st.title("Image Selected")
    resized_image = img.resize((200, 200))
    st.image(resized_image)
    predictions = predict(img, model)
    predictions_label = np.argmax(predictions[0])
    predictions_label = class_names[predictions_label]
    st.title("Predicted Result")
    st.header(predictions_label)
