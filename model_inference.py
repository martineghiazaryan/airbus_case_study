import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from data_preprocessing import preprocess_test_data

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def predict_image_segmentation(model, image_path, threshold=0.9):
    # Preprocess the image
    X_test = preprocess_test_data([image_path])

    # Make predictions on the test image
    predicted_segmentation = model.predict(X_test)

    # Apply threshold to convert probabilities to binary predictions
    predicted_segmentation = (predicted_segmentation > threshold).astype(np.uint8)
    return predicted_segmentation

def plot_predictions(predicted_segmentation, original_image):
    # Plot original image and predicted mask
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_segmentation[0, ..., 0], cmap='gray')  
    plt.title('Predicted Segmentation')
    plt.show()

if __name__ == "__main__":
    # Define the path to your test images directory
    # test_img_dir = '/kaggle/input/airbus-ship-detection/test_v2/'

    test_img_dir = ''
    model_path = '/models/model_best_checkpoint.h5' # the path to the model

    # Load the model
    model = load_trained_model(model_path)

    # Get the list of image names in the test directory
    test_ids = os.listdir(test_img_dir)

    # Select a random image name from the test dataset
    img_name = np.random.choice(test_ids)

    # Preprocess the image
    img_path = os.path.join(test_img_dir, img_name)
    X_test = preprocess_test_data([img_path])
    original_image = X_test[0]

    # Make predictions on the test image
    predicted_segmentation = predict_image_segmentation(model, img_path)

    # Plot the predictions
    plot_predictions(predicted_segmentation, original_image)
