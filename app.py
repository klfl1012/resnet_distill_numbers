import streamlit as st 
from streamlit_drawable_canvas import st_canvas
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from models.teacher import Teacher
from models.student import StudentCNN 
import time 
import onnx, onnxruntime as ort 

from loadmodels import ModelManager


class App:

    def __init__(self):
        st.set_page_config(
            layout="wide",
            page_title="Distillation Comparisoin App"
        )
        st.title("Distillation Comparison App")
        
        self.manager = ModelManager(config_paths="./models/configs/models_config.json") 


    @st.cache_resource
    def _get_models(self, model_name):
        return self.manager.load_model(model_name)

    def _create_canvas(self):
        canvas_col, features_col = st.columns(2)
    
        with canvas_col:
    
            st.write("Draw a digit from 0 to 9 and submit")
            canvas_result = st_canvas(
                fill_color="white",  
                stroke_width=10,      
                stroke_color="black", 
                background_color="white", 
                height=600,
                width=800,
                drawing_mode="freedraw",
                key="canvas",
            )

        submit = st.button("Submit")

        if submit:
            if canvas_result.image_data is not None:
                img_tensor = self.manager.transform_img(canvas_result.image_data[:, :, :3].astype("uint8"))                

        with features_col:
            st.write("Choose one or more models to predict the digit")

            toggle_teacher = st.toggle("Teacher")
            toggle_student = st.toggle("Student")

            predict = st.button("Predict")

            if submit and predict:
                with st.spinner("Predicting..."):
                    try:
                        predictions = {}

                        if toggle_teacher:
                            predictions["Teacher"] = self.manager.predict("teacher", img_tensor)

                        if toggle_student:
                            predictions["Student"] = self.manager.predict("student", img_tensor)

                        for model_name, probs in predictions.items():
                            st.write(f"{model_name} predicted: {probs.argmax()} with probability {probs.max():.2f}")
                        
                        time.sleep(3)
                        st.success("Prediction done!")

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    def run(self):
        self._create_canvas()

if __name__ == "__main__":
    app = App()
    app.run()