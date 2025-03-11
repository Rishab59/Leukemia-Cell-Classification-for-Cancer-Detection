import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# Load the saved model
MODEL_NAME = "Leukemic-model-v3-82%-accuracy.h5"
MODEL_PATH = "D:/FinalYearProject/Leukemia/SavedModel/" + MODEL_NAME

print("Model Loading...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded successfully from: {MODEL_PATH}")

# Define constants
IMG_SIZE = 224  # Ensure this matches the training image size
testing_folder = "D:/FinalYearProject/Leukemia/Testing_Model_Images/"
os.makedirs(testing_folder, exist_ok = True)  # Ensure folder exists

# Allowed file types
allowed_extensions = {".bmp", ".png", ".jpg", ".jpeg"}

# GUI to select image
print("Waiting for input image...")
Tk().withdraw()  # Hide root Tkinter window
file_path = filedialog.askopenfilename(title = "Select an image for prediction", 
                                       filetypes = [("Image Files", "*.bmp;*.png;*.jpg;*.jpeg")])

if not file_path:
    print("⚠️ No file selected. Exiting...")
else:
    file_name = os.path.basename(file_path)
    save_path = os.path.join(testing_folder, file_name)

    # Save uploaded file for testing
    with open(file_path, "rb") as src, open(save_path, "wb") as dst:
        dst.write(src.read())
    print(f"✅ Image saved at: {save_path}")

    # Load and preprocess the image
    img = image.load_img(save_path, target_size = (IMG_SIZE, IMG_SIZE))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis = 0)  # Add batch dimension

    # Make prediction
    print("Making Prediction...")
    prediction = model.predict(img_array)[0][0]  # Get prediction score

    # Compute probability percentages
    leukemia_prob = round(prediction * 100, 2)
    non_leukemia_prob = round(100 - leukemia_prob, 2)

    # Display result
    print("\n\n🔍 \"Prediction Result\":")
    if prediction >= 0.5:
        print(f"🛑 The uploaded cell image is \"Leukemic\" with {leukemia_prob:.2f}% probability.")
    else:
        print(f"✅ The uploaded cell image is \"Non-Leukemic\" with {non_leukemia_prob:.2f}% probability.")

    # Show image
    img_display = cv2.imread(save_path)
    img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title("Uploaded Image")
    plt.axis("off")
    plt.show()

    # Ask to delete file
    delete_choice = input("The uploaded image is currently saved at \"" + testing_folder + "\"\n🗑 Do you want to delete the uploaded image? (yes/no): ").strip().lower()
    if delete_choice in ["yes", "y", "ye"]:
        os.remove(save_path)
        print("✅ Image deleted successfully.")
    else:
        print("📂 Image kept in: \"", testing_folder + "\"")

print("\nScript Closed...\nThank You !!!")