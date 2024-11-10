# Real-Time Facial Expression Recognition

This project is a Real-Time Facial Expression Recognition (FER) system that utilizes Convolutional Neural Networks (CNNs) to detect and classify emotions from facial expressions in images and live video streams. The project supports seven basic emotions: **angry, disgust, fear, happy, sad, surprise, and neutral**.

To explore more interesting Computer Vision projects, please check [here](https://github.com/HoDangCao/openCV_projEx).

## Demo

<img src='./real-time_fer_demo.gif' width=500>

## Features
- **Real-Time Emotion Detection**: Detects emotions from facial expressions in real-time using a webcam.
- **Data Processing and Normalization**: Pre-processes images for optimal model performance.
- **Model Persistence**: Saves the trained model to disk for easy reuse.
- **Performance Evaluation**: Includes metrics such as accuracy, F1-score, and a confusion matrix for in-depth evaluation.
- **Visualization**: Displays bar charts and heatmaps to analyze and understand predictions.

## Technologies Used
- **Python**: Core programming language.
- **Keras & TensorFlow**: For building, training, and saving the CNN model.
- **OpenCV**: For real-time face detection.
- **Matplotlib & Seaborn**: For data visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Real-Time-FER.git
   cd Real-Time-FER
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [here](https://drive.google.com/file/d/18yZyj5qqg8-UHlheqmCMM4wf9KOdV42T/view) and place it in the `./data` directory.

## Model Architecture

The model is a CNN with multiple convolutional, pooling, and fully connected layers:
1. **Input Reshaping**: Converts flattened input into a 48x48 grayscale image format.
2. **Convolution Layers**: Extracts features from images with ReLU activation.
3. **Pooling Layers**: Reduces spatial dimensions to focus on essential features.
4. **Dropout Layers**: Reduces overfitting by randomly disabling neurons during training.
5. **Fully Connected Layers**: Processes the features into emotion predictions.

## Training & Evaluation

Run the `notebook` to see the full building, training and evaluation process.

During training, the model uses **categorical cross-entropy** loss and **Adam optimizer** to improve accuracy. Results will be displayed, showing metrics such as training loss, accuracy, and F1-score.

This script provides performance metrics on the test data, including confusion matrices for analyzing prediction accuracy across different emotions.

## Real-Time Emotion Detection

The system can detect emotions in real-time via a webcam. To start live emotion detection, run:
```python
python real-time_reg.py
```

Press the `q` key to exit the detection window.

## Saved Model

You can use the pretrained model is saved as:
- `fer.json`: Model architecture
- `fer.h5`: Model weights

To reload and use the saved model:
```python
model = model_from_json(open("fer.json", "r").read())
model.load_weights("fer.h5")
```
