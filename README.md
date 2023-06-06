# Airbus Ship Detection - Segmentation Model

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
    - [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
    - [Model Architecture](#model-architecture)
    - [Model Development](#model-development)
    - [Training the Model](#training-the-model)
- [Running Inference on Test Images](#running-inference-on-test-images)
- [Configuring Your Paths](#configuring-your-paths)
- [Results](#results)
- [Contact](#contact)


## Introduction

This project aims to build a semantic segmentation model using the U-Net architecture to address the task of ship detection in satellite images. The goal is to accurately identify and delineate ships within the images, which can have significant applications in maritime surveillance, naval navigation, and environmental monitoring.

### Problem Statement
The task involves segmenting ships from satellite images, which is a challenging problem in computer vision. Semantic segmentation requires assigning a specific class label to each pixel in an image, making it an essential technique for object recognition and scene understanding.

### Work Done
In this project, I trained a U-Net model using tf.keras, a high-level neural networks API in Python. The U-Net architecture is well-suited for image segmentation tasks due to its ability to classify pixels efficiently and handle complex data patterns.

The model was trained and evaluated using the Dice score, which measures the overlap between predicted and ground truth segmentations. A higher Dice score indicates a better match between the predicted and actual ship boundaries.

### Notebooks
The development and experimentation process are documented in Jupyter notebooks, providing detailed insights into our methodology, thought processes, and key findings. The notebooks included in this repository are as follows:

1. `airbus-ship-segment-everything.ipynb`: This notebook incorporates the recent Segment-Everything model from Meta AI and adapts it for binary classification in the context of ship segmentation.

2. `airbus-ship-segmentation.ipynb`: This notebook contains the primary codebase for the U-Net model implementation using tf.keras. It serves as the core task implementation for ship segmentation and has been adapted specifically for a Kaggle notebook.

3. `airbus-ship-segmentation_models.ipynb`: This notebook explores an alternative approach by utilizing a pre-trained encoder based on the ResNet34 architecture. The segmentation_models library is used to experiment with this approach, leveraging the benefits of transfer learning and high-level feature extraction.

4. `exploratory-analysis.ipynb`: This notebook includes some exploratory data analyisis.

These notebooks provide a comprehensive overview of our work, including code, documentation, and insights, ensuring transparency, understanding, and reproducibility of our research.


## Prerequisites 

Before you can run the code in this repository, there are a few steps that you need to follow:

1. **Download the preprocessed data**: To save time on preprocessing, you can download the preprocessed `X_train`, `y_train`, `X_valid`, and `y_valid` datasets directly from this [link](https://drive.google.com/file/d/1bEcS5kJSgKq3-Xg4FpWQAfjKK3z1S3sR/view?usp=sharing). Once downloaded, please locate these files in the root directory of this project.

2. **Download the original dataset**: The original dataset, along with the images, can be downloaded from the following Kaggle competition - [Airbus Ship Detection](https://www.kaggle.com/c/airbus-ship-detection/data). Please download and change the corresponding paths to the images folders in the code. 

Here is the folder structure for the dataset that you will need:

- 📁 airbus-ship-detection
  - 📁 test_v2
  - 📁 train_v2
  - train_ship_segmentations_v2.csv


## Folder Structure

This project has a modular structure and is divided into several Python scripts and Jupyter notebooks for various tasks. The organization is as follows:

- 📁 root
  - data_preprocessing.py
  - main.py
  - model_creation.py
  - model_inference.py
  - model_training.py
  - README.md
  - requirements.txt
  - train_df.csv
  - valid_df.csv
  - 📁 .ipynb_checkpoints
    - Airbus Case study -checkpoint.ipynb
    - airbus-case-study (1)-checkpoint.ipynb
  - 📁 models
    - model_best_checkpoint.h5
  - 📁 Notebooks
    - airbus-ship-segment-everything.ipynb
    - airbus-ship-segmentation.ipynb
    - airbus-ship-segmentation_models.ipynb
    - exploratory-analysis.ipynb
  - 📁 __pycache__
    - data_preprocessing.cpython-310.pyc
    - model_creation.cpython-310.pyc
    - model_inference.cpython-310.pyc
    - model_training.cpython-310.pyc


- `data_preprocessing.py`: This script contains all the necessary steps for preprocessing the Airbus Ship Detection Dataset. 

- `model_creation.py`: This script is used for defining and creating the segmentation model.

- `model_training.py`: This script is used for training the model on the preprocessed dataset.

- `model_inference.py`: This script is used for making predictions with the trained model.

- `main.py`: This is the main driver script that coordinates the running of the scripts mentioned above.

- `train_df.csv` and `valid_df.csv`: These are CSV files that contain the training and validation dataframes respectively.

- `requirements.txt`: This file lists all the Python dependencies required to run the project.

The repository also contains the following directories:

- `models`: This directory contains the saved model weights and architectures. Currently, it contains `model_best_checkpoint.h5`, which are the weights of the best model checkpoint during training. This will help you save time and not train the model. Just load the model and predict. See the instructions.

- `Notebooks`: This directory contains Jupyter notebooks some additional workflows that I did except for the task which was to train the U-Net model. It currently includes:
    - `airbus-ship-segment-everything.ipynb`: This notebook includes the recent Segment-Everything model from Meta AI. In integrated the model and adjusted it to do binary classification in our case of ship-segmentation.
    - `airbus-ship-segmentation.ipynb`: This notebook is the same model version that I wrote in this repository only in the version of kaggle notebook.
    - `airbus-ship-segmentation_models.ipynb`: This notebook presents another aproach that I tried which is to use a pre-trained encoder based on resnet34 also using and experimenting with the **segmentation_model** library.
    - `exploratory-analysis.ipynb`: This notebook includes some exploratory data analyisis.
  
- `.ipynb_checkpoints`: This directory contains checkpoint files from Jupyter notebooks, which are created while the notebook is open. 

- `__pycache__`: This directory contains compiled Python scripts, generated by Python interpreter for performance optimization.

- `README.md`: This is the file you're reading now. It provides an overview of the project and explains how to use it.


## Dataset

The data used in this project comes from the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/data) hosted on Kaggle. The challenge of this competition is to detect ships in satellite images as quickly as possible.

The dataset is structured as follows:

- The `train_v2` folder contains a set of jpg images, each of which can contain multiple ships. 
- The `test_v2` folder contains jpg images in the public test set.
- The `train_ship_segmentations_v2.csv` file provides the run-length encoded pixel locations of the ship for the training images. 

The dataframe has two columns. The first column is the `ImageId` which basically has the image names as strings and the second column is the `EncodedPixels` which is the rle-encoded coordinates of the ship present in the images. RLE is run-length encoding. It is used to encode the location of foreground objects in segmentation. Instead of outputting a mask image, you give a list of start pixels and how many pixels after each of those starts is included in the mask.

The images are of resultion 768x768 pixels.

The images were acquired by Airbus' Pleiades satellites, which are capable of detailed imaging with a resolution of up to 0.5 meters.

Each image can contain multiple ships or no ships at all, and the task is to identify the presence of ships in these images and also locate them. For more detailed information about the dataset, please refer to the competition's [official webpage](https://www.kaggle.com/c/airbus-ship-detection/data).


## Methodology

Here step-by-step approach to detect ships in satellite images, consisting of data cleaning and preprocessing, model development, training, and evaluation.

### Data Cleaning & Preprocessing

The data preprocessing includes the loading and filtering of the initial dataset, which is located in the `train_ship_segmentations_v2.csv` file. The first 5000 images are kept for processing because the RAM of the kaggle notebook was crushing (it is limited). This dataset is then divided into two parts: images with ships (`ships_df`) and images without ships (`no_ships_df`). Each part is further split into training and validation datasets, with 80% for training and 20% for validation.

The script also includes a `preprocess_data` function for resizing images and masks and converting the masks to categorical format. A `preprocess_test_data` function is included for resizing the test images.

The resulting datasets are then saved as `train_df.csv` and `valid_df.csv` for further use. These are the files that you can see in the repository root.

## Model Architecture

The model used in this project is based on the U-Net architecture, a type of convolutional neural network that was originally developed for biomedical image segmentation. It's known for its "U" shape, which comes from a series of down-sampling layers , followed by an up-sampling path  which restores the original resolution of the image. The encoder and decoder parts are connected by skip connections that allow low-level feature maps to be directly fed into the decoder part of the network.

In this script, the U-Net model is implemented using TensorFlow's Keras API. The model is defined using the functional API, which allows for more flexibility in connecting layers and defining complex models.

The U-Net model implemented in this script consists of:

1. **Encoder Part**: The encoder part is composed of three blocks, each containing two convolutional layers followed by a max pooling layer. The number of filters in the convolutional layers doubles with each block, starting with 64 in the first block.

2. **Bottom Part**: The bottom of the U-Net (between the encoder and decoder parts) consists of two convolutional layers with 512 filters each.

3. **Decoder Part**: The decoder part is also composed of three blocks. Each block starts with an up-sampling layer, followed by a concatenation with the corresponding block in the encoder part, and then two convolutional layers. The number of filters in the convolutional layers halves with each block, starting with 256 in the first block.

4. **Output Layer**: The final layer is a convolutional layer with a number of filters equal to the number of classes in the target (i.e., `n_classes`). This layer uses the softmax activation function to output probabilities for each class.

In addition to the U-Net model, this script also defines a custom loss function called `weighted_binary_crossentropy`. This loss function computes the binary cross-entropy loss, but applies a weight factor to the predictions for the ship class. This can be useful if the classes in the target are imbalanced, as it gives more importance to the minority class.

To build and compile the model, simply call the `build_unet` function with the desired input shape and number of classes.

### Model Development

The model used in this project is a custom TensorFlow model defined in `model_creation.py`. This model uses a custom metric function `dice_coef`, defined in `model_training.py`.

## Training the Model

The `train_model.py` script is responsible for training the U-Net model on the image segmentation task. It uses the training and validation data prepared by the `data_preprocessing.py` script and the model defined by the `model_creation.py` script. This script also contains two additional performance metrics: IoU coefficient and Dice coefficient.

### Metrics

- **Intersection over Union (IoU):** This metric measures the overlap between the true and predicted segmentations. A higher IoU score means a better match.

- **Dice Coefficient:** This metric is similar to IoU but is twice the area of overlap divided by the total number of pixels in both images. It ranges between 0 (no overlap) and 1 (perfect match). It is good for imbalanced dataset.

### Training Procedure

The model is trained using the Adam optimizer and binary cross entropy loss. The training procedure includes the following steps:

1. Loading the training and validation data. If preprocessed data is already saved as `.npy` files, the script loads this data. If not, we the `preprocess_data` function from `data_preprocessing.py` to preprocess the data and save it as `.npy` files.

2. Building and compiling the U-Net model. If a pre-existing model is provided, the script uses this model. If not, the script builds a new model using the `build_unet` function from `model_creation.py`.

3. Training the model on the training data for a specified number of epochs, using the Adam optimizer and binary cross entropy loss. The model's performance is evaluated on the validation data at the end of each epoch. 

4. Saving the best model based on the validation performance during training.

To train the model, simply call the `train_model` function. This function returns the trained model and the history of training (including the training and validation loss and metrics at each epoch).

## Running Inference on Test Images

The `run_inference.py` script loads a trained model and uses it to make predictions on a set of test images. These images are randomly selected from a specified directory. The script also includes a function to plot the original images alongside the predicted segmentations.

### Inference Procedure

The script follows these steps:

1. **Load the trained model:** The `load_trained_model` function is used to load the trained model saved as a `.h5` file. The model is compiled during this loading process.

2. **Prepare the test images:** A list of image names is obtained from the test image directory. The script then randomly selects a certain number of images from this list (currently set to 20 images) for prediction.

3. **Preprocess the test images:** Each test image is preprocessed using the `preprocess_test_data` function from `data_preprocessing.py`.

4. **Make predictions:** The trained model is used to make predictions on the preprocessed test images.

5. **Plot the predictions:** The `plot_predictions` function is used to plot the original image and the predicted segmentation side by side.

To run the inference, simply call the `run_inference` function with the path to your test images directory and the path to the trained model as arguments.

### Example Usage

```python
test_img_dir = 'd:/Profils/myeghiazaryan/Downloads/test_v2/'
model_path = 'd:/Profils/myeghiazaryan/Desktop/airbus_case_study/models/model_best_checkpoint.h5'
run_inference(test_img_dir, model_path)
```
If you want to see segmentations of the ships you can simply run the ``model_inference.py`` file by running the following command:

```python
python model_inference.py
```

## Configuring Your Paths

In order to utilize this repository with your own data, you will need to adjust various file and directory paths within the scripts to match your own environment. Here's a summary of what might need to be changed:

1. **Image Directory:** This is the directory where your image data resides. For both the training and testing scripts, you need to set this to point to your own image directories. These are specified in the scripts `data_preprocessing.py` and `run_inference.py` with the variable `img_dir`.

Example:
```python
img_dir = 'd:/Profils/myeghiazaryan/Downloads/train_v2/'  # Change this to your directory
```
2. **Model Path:** This is the location where the trained model .h5 file is saved. The path is specified in train_model.py for saving the trained model and ``run_inference.py`` for loading the trained model.

Example:

``model_path = 'd:/Profils/myeghiazaryan/Desktop/airbus_case_study/models/model_best_checkpoint.h5'``  # Change this to your path

3. **Dataframe CSV Files:** In ``train_model.py``, the paths to ``train_df.csv`` and ``valid_df.csv`` should be updated to the location where these CSV files are saved in your environment.

4. **Numpy Data Files:** In ``train_model.py``, the paths to ``X_train.npy``, ``y_train.npy``, ``X_valid.npy``, and ``y_valid.npy`` need to be set to the desired save/load location for these numpy data files.

Remember to adjust these paths to fit your specific directory structure and naming conventions.

## Results

Although the primary focus was the development and training of the U-Net model, due to constraints such as limited RAM, computational power, and time, I was note able to optimally train this model. 

However, despite these limitations, I managed to successfully leverage the segmentation_models library, and also the very recent open source Segment-Everything model from Meta.AI. This model was adapted for binary classification, facilitating the segmentation of ships within the satellite imagery.

I evaluated the performance using the Dice score, getting not very bad results for my trained U-Net model promising results with the segmentation_models' pretrained encoder and comparingly good segmentation results with the adapted model from Meta.AI. For a detailed breakdown of the results and performance metrics, please refer to the respective Jupyter notebooks.

![alt text](https://github.com/[martineghiazaryan]/[airbus_case_study]/blob/[branch]/image.jpg?raw=true)



## Contact
[Back to top](#table-of-contents)

For any questions or feedback, feel free to reach out to me:

- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/martin-yeghiazaryan/)
- **Email**: [martineghiazaryan@gmail.com](mailto:martineghiazaryan@gmail.com)



