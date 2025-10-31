import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import base64

# ---- Función para convertir imagen a base64 (fondo personalizado) ----
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---- Fondo oceánico ----
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- Cargar fondo ----
set_background("/mnt/data/5f4f0614-fb31-4ae0-9933-0ebac983e6e6.png")

# ---- Modelo de predicción ----
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# ---- Configuración de la App ----
st.set_page_config(page_title='🌊 Reconocimiento Oceánico de Dígitos 🐠', layout='wide')

# ---- Estilo general ----
st.markdown("""
    <style>
    h1, h2, h3, h4 {
        color: #E0FFFF;
        text-shadow: 1px 1px 2px #000;
    }
    .stButton>button {
        background-color: #00CED1;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #20B2AA;
        transform: scale(1.05);
        transition: 0.3s;
    }
    .css-1aumxhk {
        background-color: rgba(0, 0, 50, 0.5);
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Título ----
st.title("🌊 Tablero Oceánico para Reconocimiento de Dígitos 🐚")
st.subheader("Dibuja el dígito en el panel y presiona **'Predecir'**")

# ---- Configuración del lienzo ----
drawing_mode = "freedraw"
stroke_width = st.slider('✏️ Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'  # Blanco para contraste
bg_color = '#001F3F'      # Azul profundo oceánico

# ---- Lienzo ----
canvas_result = st_canvas(
    fill_color="rgba(0, 255, 255, 0.3)",  # color agua
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=250,
    width=250,
    drawing_mode=drawing_mode,
    key="canvas",
)

# ---- Botón de predicción ----
if st.button('🌊 Predecir Dígito'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f'🐬 El dígito reconocido es: **{res}**')
    else:
        st.warning('Por favor dibuja un dígito en el lienzo.')

# ---- Sidebar ----
st.sidebar.title("🌴 Acerca de:")
st.sidebar.markdown("""
**Aplicación Oceánica de Reconocimiento de Dígitos**
- Utiliza una red neuronal convolucional (CNN)
- Entrenada con el dataset **MNIST**
- Fondo inspirado en el **océano** 🌊  
""")
