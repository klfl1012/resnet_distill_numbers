import streamlit as st 
import numpy as np
from streamlit_drawable_canvas import st_canvas


st.set_page_config(layout="wide")
st.title("Test")


canvas_col, features_col = st.columns([1, 1])

with canvas_col:
    canvas_result = st_canvas(
        fill_color="white",  
        stroke_width=10,      
        stroke_color="black", 
        background_color="white", 
        height=600,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

with features_col:
    st.write("Models and features will be displayed here")
    st.button("Predict")