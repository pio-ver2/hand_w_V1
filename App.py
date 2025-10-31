import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Estilo visual con temática oceánica
st.markdown("""
    <style>
        body {
            background-color: #004d40;  /* Azul océano profundo */
            color: #ffffff;  /* Texto blanco */
        }
        .stTitle {
            color: #00bcd4;  /* Azul océano claro para el título */
        }
        .stSubheader {
            color: #80deea;  /* Azul claro para los subtítulos */
        }
        .stButton>button {
            background-color: #00796b;  /* Botones de color verde mar */
            color: white;  /* Texto blanco en los botones */
        }
        .stImage>div>img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .stSidebar {
            background-color: #80deea;  /* Azul claro para la barra lateral */
        }
        .stTextInput>div>div>input {
            background-color: #4db6ac;  /* Fondo de los campos de texto en verde suave */
        }
        .stTextArea>div>div>textarea {
            background-color: #4db6ac;  /* Fondo del área de texto */
        }
        .stMarkdown {
            color: #ffffff;  /* Texto de Markdown en blanco */
        }
        /* Barra lateral con texto en azul oscuro */
        .stSidebar .stText {
            color: #003366; /* Azul oscuro para el texto en la barra lateral */
        }
    </style>
""", unsafe_allow_html=True)

# App
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# Streamlit 
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='wide')
st.title('🌊 **Reconocimiento de Dígitos escritos a mano** 🤖')
st.subheader("🖊️ **Dibuja el dígito en el panel y presiona 'Predecir'**")

# Add canvas component
# Specify canvas parameters in application
drawing_mode = "freedraw"
stroke_width = st.slider('🔲 **Selecciona el ancho de línea**', 1, 30, 15)
stroke_color = '#FFFFFF'  # Set background color to white
bg_color = '#80deea'  # Azul claro para el fondo

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

# Add "Predict Now" button
if st.button('🔍 **Predecir**'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'),'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('🌊 **El Dígito es** : ' + str(res))
    else:
        st.header('⚠️ **Por favor dibuja en el canvas el dígito.**')

# Add sidebar
st.sidebar.title("🏖️ **Acerca de**:")
st.sidebar.text("En esta aplicación se evalúa la capacidad de un RNA para reconocer")
st.sidebar.text("dígitos escritos a mano.")
st.sidebar.text("Basado en desarrollo de Vinay Uniyal")

# ==============================
# CONFIGURACIÓN DE LA IMAGEN
# ==============================
st.sidebar.subheader("📸 **Configuración de Imagen**")
uploaded_image = st.file_uploader("📥 **Sube una imagen para probar el modelo**", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Imagen cargada", use_column_width=False, width=200)
    img = Image.open(uploaded_image)
    if st.button("🔍 **Predecir Imagen Cargada**"):
        res = predictDigit(img)
        st.success(f"🌊 **El dígito reconocido es**: {res}")
