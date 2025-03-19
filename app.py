import streamlit as st, pandas as pd, numpy as np, time, requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
from loadmodels import ModelManager
from models.utils import get_dataloaders

API_URL = "http://127.0.0.1:8000"


class App:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Distillation Comparison App")
        st.title("Distillation Comparison App")
        
        self.manager = ModelManager(config_paths="./models/configs/models_config.json")
        self._initialize_session_state()

    def _initialize_session_state(self):
        if "submitted" not in st.session_state:
            st.session_state["submitted"] = False
        if "img_tensor" not in st.session_state:
            st.session_state["img_tensor"] = None
        if "true_label" not in st.session_state:
            st.session_state["true_label"] = None

    @st.cache_resource
    def _get_models(self, model_name):
        return self.manager.load_model(model_name)
    
    def _get_available_models(self):
        response = requests.get(f"{API_URL}/models/")   
        if response.status_code == 200:
            return response.json()["available models"]
        return []

    def _highlight_max(self, row, label):
        colors = [""] * len(row)
        max_index = row[1:].argmax().astype(int)
        argmax_col_name = str(max_index)
        color = "green" if int(argmax_col_name) == label else "red"
        colors[row.index.get_loc(argmax_col_name)] = f"background-color: {color}; color: white;"
        return colors

    def _load_random_img(self):
            _, testloader = get_dataloaders(batch_size=32, resize=(28, 28))

            for img, label in testloader:
                rand_idx = np.random.randint(0, 32)
                img = img[rand_idx]
                label = label[rand_idx]

                unnormalize = transforms.Compose([
                    transforms.Resize((150, 150)),  
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.ToPILImage()
                ])

                img_resized = unnormalize(img).convert("L")
                st.session_state["true_label"] = label.item()
                st.session_state["img_tensor"] = img
                st.session_state["submitted"] = True
                st.image(img_resized, caption=f"True label: {label.item()}")
                break

    def _render_model_selection(self):
        st.write("Choose one or more models to predict the digit")
        toggle_teacher = st.toggle("Teacher", value=True)
        toggle_student = st.toggle("Student", value=True)
        true_label = st.selectbox("True label", list(range(10)), index=st.session_state["true_label"] if st.session_state["true_label"] is not None else 0)

        predict = st.button("Predict", disabled=not st.session_state["submitted"] or st.session_state["img_tensor"] is None)

        if predict and true_label is not None: 
            self._predict_from_api(true_label, toggle_teacher, toggle_student)

    def _predict(self, true_label, toggle_teacher, toggle_student):
        with st.spinner("Predicting..."):
            try:

                if st.session_state["img_tensor"] is None:
                    st.error("No image to predict")
                    return

                predictions = {}
                if toggle_teacher:
                    predictions["Teacher"] = self.manager.predict("teacher", st.session_state["img_tensor"]).tolist()

                if toggle_student:
                    predictions["Student"] = self.manager.predict("student", st.session_state["img_tensor"]).tolist()

                predictions_flat = {model_name: probs[0] for model_name, probs in predictions.items()}
                df_predictions = pd.DataFrame.from_dict(predictions_flat, orient="index", columns=[str(i) for i in range(10)])
                df_predictions.insert(0, "Model", df_predictions.index)
                df_predictions.reset_index(drop=True, inplace=True)
                styled_df = df_predictions.style.apply(lambda row: self._highlight_max(row, true_label), axis=1)

                time.sleep(2)
                st.success("Prediction successful")
                st.write("### Model Predictions")
                st.write(styled_df)

            except Exception as e:
                st.error(f"Prediction failed: {e}")


    def _predict_from_api(self, true_label, toggle_teacher, toggle_student):
        with st.spinner("Predicting..."):
            try:

                if st.session_state["img_tensor"] is None:
                    st.error("No image to predict")
                    return
                
                files = {"file": st.session_state["img_tensor"]}
                selected_models = []

                if toggle_teacher:
                    selected_models.append("teacher")
                if toggle_student:
                    selected_models.append("student")

                img_numpy = st.session_state["img_tensor"].cpu().numpy().tolist()

                payload = {
                    "image": img_numpy,
                    "model_list": selected_models
                }
                response = requests.post(f"{API_URL}/predict/", json=payload)

                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    predictions_flat = {model_name: probs for model_name, probs in predictions.items()}
                    df_predictions = pd.DataFrame.from_dict(predictions_flat, orient="index", columns=[str(i) for i in range(10)])
                    df_predictions.insert(0, "Model", df_predictions.index)
                    df_predictions.reset_index(drop=True, inplace=True)
                    styled_df = df_predictions.style.apply(lambda row: self._highlight_max(row, true_label), axis=1)
                    time.sleep(2)
                    st.success("Prediction successful")
                    st.write("### Model Predictions")
                    st.write(styled_df)
                else:
                    st.error(f"Prediction failed: {response.json()["detail"]}")
                
            except Exception as e:
                st.error(f"Error: {e}")


    def _create_canvas(self):
        canvas_col, features_col = st.columns(2)

        with canvas_col:
            st.write("Draw a digit from 0 to 9 and submit")
            canvas_result = st_canvas(
                fill_color="black",  
                stroke_width=55,
                stroke_color="white", 
                background_color="black", 
                height=600,
                width=600,
                drawing_mode="freedraw",
                key="canvas",
            )

        with features_col:
            submit = st.button("Submit")
            random_img_button = st.button("Load random image")

            img_tensor = None

            if submit and canvas_result.image_data is not None:
                img_tensor = canvas_result.image_data[:, :, :3].astype("uint8")  

                img_tensor = self.manager.transform_img(img_tensor)

            if random_img_button:
                try:
                    self._load_random_img()
                    img_tensor = st.session_state["img_tensor"]
                except Exception as e:
                    st.error(f"Failed to load random image: {e}")

            if img_tensor is not None:
                st.session_state["img_tensor"] = img_tensor
                st.session_state["submitted"] = True

            self._render_model_selection()

    def run(self):
        self._create_canvas()

if __name__ == "__main__":
    app = App()
    app.run()