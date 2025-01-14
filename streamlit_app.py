import streamlit as st
import subprocess
import inference
import preprocessing



##### function to be called when user clicks 'setup environment button"
def setup_environment():
    try:
        subprocess.run(["bash", "install.sh"], check=True)  # Place `install.sh` in the root directory
        st.success("Environment setup complete.")
    except subprocess.CalledProcessError as e:
        st.error(f"Setup failed: {e}")

#title
st.title("Detect DeepFake Media")

#setup environment button
if st.button("Setup Environment"):
    setup_environment()

# File Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#image detector
detector = st.selectbox("Select which detection model to be used for image based detection", ["MesoNet", "CapsuleNet"])


###### run_inference functionality
if st.button("Run Inference"):
    if uploaded_file is not None:
        # Detect file format
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ["png", "jpg", "jpeg"]:
            st.error("Unsupported file format. Please upload a PNG, JPG, or JPEG file.")
        else:
            # Save uploaded file locally with its original format
            file_path = f"uploaded_image.{file_extension}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run inference
            try:
                st.write("Running inference...")
                cropped_face, landmarks, masks, cropped_landmarks = preprocessing.preprocess_image(file_path)  # Custom function
                st.image(result["croppped_landmarks"], caption="Output Image")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please upload a file to proceed.")

