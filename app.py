import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space  # Install with pip if needed
from streamlit_extras.colored_header import colored_header  # Install with pip if needed

# Load the trained model
model_path = "vgg19_lung_disease_finetuned.keras"  # Ensure this is in the same directory or provide the correct path
model = tf.keras.models.load_model(model_path, compile=False)

# Class labels (ensure these match the order in your dataset's class_indices)
class_labels = ['Covid', 'Normal', 'Pneumonia', 'Tuberculosis']

# Function to classify an uploaded image
def classify_image(uploaded_image):
    # Preprocess the image
    img = uploaded_image.resize((224, 224))  # Resize image to 224x224
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return class_labels[predicted_class], confidence

# Streamlit Frontend
def main():
    # Set page config with a background image
    st.set_page_config(
        page_title="Lung Disease Classifier",
        page_icon="🫁",
        layout="wide",
    )
    
    # Custom CSS for background and text
    st.markdown(
        """
        <style>
        /* Add a background image */
        .stApp {
            background: url('https://images.unsplash.com/photo-1606107562676-dc2f2d9a0c37');
            background-size: cover;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App header
    st.title("🫁 Lung Disease Classifier")
    st.markdown("""
    **Empowering Healthcare with AI**  
    Upload chest X-ray images to classify them into one of the following categories:
    - **Covid**
    - **Normal**
    - **Pneumonia**
    - **Tuberculosis**
    """)

    add_vertical_space(2)

    # Sidebar
    with st.sidebar:
        st.image("D:/mini/lung.png", width=150)
        st.markdown("""
        ## Instructions
        - Upload a chest X-ray image in **JPG, JPEG, or PNG** format.  
        - Wait for the model to process the image.  
        - View the classification result and suggestions for further action.  
        """)
        add_vertical_space(2)

        # About Section
        st.markdown("## About the Model")
        st.markdown("""
        This application uses a deep learning model to analyze chest X-ray images and classify them into four categories:  
        - **Covid**: Signs indicative of COVID-19.  
        - **Normal**: No abnormalities detected in the lungs.  
        - **Pneumonia**: Inflammation in the lungs detected.  
        - **Tuberculosis**: Potential signs of tuberculosis detected.  

        The classifier is intended for educational purposes to demonstrate how AI can aid in medical imaging analysis. 
        However, it should not replace professional medical consultation or diagnosis.
        """)

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("Uploaded Image Preview")
        image = Image.open(uploaded_file)
        
        # Create two columns for the image display
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display smaller thumbnail
            st.image(image, caption="Uploaded Image", width=200)
        
        with col2:
            # Show only file name
            st.write(f"**File name:** {uploaded_file.name}")

        # Perform classification
        with st.spinner("🔍 Classifying... Please wait..."):
            predicted_class, confidence = classify_image(image)

        # Display results
        colored_header(label="🩺 Diagnosis Result", color_name="blue-70")
        st.success(f"**Prediction**: {predicted_class}")
        st.info(f"**Confidence**: {confidence:.2f}")

        # Suggestions based on prediction
        st.markdown("### Suggestions:")
        if predicted_class == "Normal":
            st.success("You are perfectly fine! No signs of lung disease detected. 😊")
        elif predicted_class == "Covid":
            st.warning("Consult a healthcare provider for COVID-19 testing and treatment.")
        elif predicted_class == "Pneumonia":
            st.warning("Seek medical attention for pneumonia treatment. Follow prescribed medications.")
        elif predicted_class == "Tuberculosis":
            st.warning("Visit a healthcare specialist for tuberculosis diagnosis and treatment.")

        # Add progress bar animation
        st.markdown("""<div style="text-align: center;"><h4>Confidence Progress:</h4></div>""", unsafe_allow_html=True)
        progress_bar = st.progress(0)
        for percent in range(int(confidence * 100)):
            progress_bar.progress(percent + 1)
        
        add_vertical_space(1)
        st.markdown(f"""
        ### Explanation:  
        - **Prediction**: The model predicts the most likely class for the given X-ray image.  
        - **Confidence**: Indicates the model's certainty in its prediction (out of 1.00).
        """)

    else:
        st.warning("👈 Please upload an image to get started.")

# Run the app
if __name__ == "__main__":
    main()
