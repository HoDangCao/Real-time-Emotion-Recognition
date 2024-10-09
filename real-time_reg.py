import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Model, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Normalization
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Reshape
from keras.regularizers import l2
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.preprocessing import image
from sklearn.metrics import f1_score
import cv2

pic_height, pic_width = 48, 48
num_labels = 7

class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.reshape = Reshape((pic_height, pic_width, 1)) # Reshaping flattened input
        
        # 1st Convolution Layer
        self.conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop1 = Dropout(0.5)
        
        # 2nd Convolution Layer
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.conv4 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = Dropout(0.5)

        # 3rd Convolution Layer (adjusted to control spatial size reduction)
        self.conv5 = Conv2D(128, (3, 3), activation='relu')
        self.conv6 = Conv2D(128, (3, 3), activation='relu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))  # Final pooling layer

        # Flatten and fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(1024, activation='relu')
        self.drop3 = Dropout(0.2)
        self.fc2 = Dense(1024, activation='relu')
        self.drop4 = Dropout(0.2)
        self.output_layer = Dense(num_labels, activation='softmax')

    def call(self, inputs):
        x = self.reshape(inputs)
        
        # 1st Convolution Layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # 2nd Convolution Layer
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # 3rd Convolution Layer (adjusted pool size)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        x = self.drop4(x)

        return self.output_layer(x)

model = CNN()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = model_from_json(open("fer.json", "r").read())
# model.load_weights('fer.h5')

# model = load_model('models/cnn.keras')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

def predict_img(x, model=model, size=(pic_height, pic_width), verbose=False):
    x /= 255
    x = x.reshape(-1, size[0]*size[1])
    pred_prob = model.predict(x, verbose=0)
    return emotions[np.argmax(pred_prob, axis=1)[0]]

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret: break

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h].astype('float')
        roi_gray = cv2.resize(roi_gray, (48, 48))
        predicted_emotion = predict_img(roi_gray)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()