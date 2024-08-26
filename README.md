# Deep Facial Emotion Detection (CNN)
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Overview

- This project is an emotion classification application built using the FER2013 dataset.
- It employs a convolutional nerual network to predict emotions from facial images.
- The application is built using TensorFlow and Streamlit, making it a full-stack deep learning project.


## Features

- **Emotion Prediction**: Classify the emotion of a given facial image (e.g., happy, sad, angry).
- **Webcam Capture**: Capture images directly using your webcam.
- **File Upload**: Upload an image file for emotion classification.

## Performance
- The image below showcases the performance numbers achieved for the different models within this project (current world record = 75%)
  
<img width="407" alt="Screen Shot 2024-08-06 at 7 49 03 PM" src="https://github.com/user-attachments/assets/df5e363b-d81a-44a8-89f1-a703644f8d73">=

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```
    git clone https://github.com/mohammed-majid/CNN_emotion_classification.git
    ```

2. **Install the required packages**:
    ```
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model** and place it in the project directory:
    - `custom_model_v2.h5`

4. **Run the Streamlit application**:
    ```
    streamlit run app.py
    ```
    **or**
    ```
    python3 -m streamlit run app.py
    ```

## Usage

1. **Open the Streamlit application** in your web browser.

2. **Choose between using the webcam or uploading an image file**:
   - **Webcam**: Click the "Capture Image" button to take a picture.
   - **File Upload**: Click the "Upload Image" button to upload a file from your computer.

3. **Click the "Predict Emotion" button** to get the emotion prediction.


## Acknowledgements

This project was developed using the following libraries and tools:
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)

### Side Note
- Considering the size of the dataset used for this project, I was unable to commit it to this repository. In case you want to check it out, [Press here.](https://www.kaggle.com/datasets/msambare/fer2013)

