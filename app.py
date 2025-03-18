import streamlit as st 
from streamlit_drawable_canvas import st_canvas
import pandas as pd 
import time 
from loadmodels import ModelManager


class App:

    def __init__(self):
        st.set_page_config(
            layout="wide",
            page_title="Distillation Comparisoin App"
        )
        st.title("Distillation Comparison App")
        
        self.manager = ModelManager(config_paths="./models/configs/models_config.json") 

        if "submitted" not in st.session_state:
            st.session_state.submitted = False

        if "img_tensor" not in st.session_state:
            st.session_state.img_tensor = None


    @st.cache_resource
    def _get_models(self, model_name):
        return self.manager.load_model(model_name)

    def _highlight_max(self, row, label):
        colors = [""] * len(row)
        max_index = row[1:].argmax().astype(int)   
        argmax_col_name = str(max_index + 1)
        if int(argmax_col_name) == label:
            colors[row.index.get_loc(argmax_col_name)] = "background-color: green; color: white;"
        else:
            colors[row.index.get_loc(argmax_col_name)] = "background-color: red; color: white;"
        return colors

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
                st.session_state["img_tensor"] = self.manager.transform_img(canvas_result.image_data[:, :, :3].astype("uint8")) 
                st.session_state["submitted"] = True               

        with features_col:
            st.write("Choose one or more models to predict the digit")

            toggle_teacher = st.toggle("Teacher")
            toggle_student = st.toggle("Student")
            true_label = st.selectbox("True label", list(range(10)))
            predict = st.button("Predict", disabled= not st.session_state["submitted"] or st.session_state["img_tensor"] is None)

            if predict and true_label is not None: 
                with st.spinner("Predicting..."):
                    try:
                        predictions = {}

                        if toggle_teacher:
                            predictions["Teacher"] = self.manager.predict("teacher", st.session_state["img_tensor"]).tolist()

                        if toggle_student:
                            predictions["Student"] = self.manager.predict("student", st.session_state["img_tensor"]).tolist()
                        
                        st.session_state["submitted"] = False
                        st.session_state["img_tensor"] = None   

                        predictions_flat = {model_name: probs[0] for model_name, probs in predictions.items()}
                        df_predictions = pd.DataFrame.from_dict(predictions_flat, orient="index", columns=[str(i) for i in range(1, 11)])
                        df_predictions.insert(0, "Model", df_predictions.index)
                        df_predictions.reset_index(drop=True, inplace=True)
                        styled_df = df_predictions.style.apply(lambda row: self._highlight_max(row, true_label), axis=1)

                        time.sleep(1)
                        st.success("Prediction successful")
                        st.write("### Model Predictions")
                        st.write(styled_df)

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    def run(self):
        self._create_canvas()

if __name__ == "__main__":
    app = App()
    app.run()