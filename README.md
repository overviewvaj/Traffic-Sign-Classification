# Traffic-Sign-Classification

# Introduction
This project aims to classify traffic signs using deep learning techniques. The German Traffic Sign Recognition Benchmark dataset is used for this project.

# Dataset
The dataset is available for download at https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html. It consists of two files:

GTSRB_Final_Test_Images.zip - contains 12,630 .ppm images for testing.
GTSRB_Final_Training_Images.zip - contains 39,209 .ppm images for training.
Each image is 32x32 pixels and there are a total of 43 categories (classes) in the training dataset.

# Accessing the Dataset
To access the dataset, please download the files and extract them into a folder. You will need to create two empty folders in the same folder which now has the training images, testing images and a .csv file:

• Tmp_Data
• pictureOutput
Please download the GT_final_test.csv file from this repository and place it in the same folder as the training and testing images and the above two folders.

Model
The following model architecture was used for this project:

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

The model was compiled with the following parameters:
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

# Usage
1. Clone the repository and navigate to the project directory.
2. Download and extract the dataset from the link provided above.
3. Create two empty folders in the same folder as the dataset: Tmp_Data and pictureOutput.
4. Download GT_final_test.csv and place it in the same folder as the dataset and the two folders created above.
5. Open the Jupyter notebook file Traffic_Sign_Classification.ipynb and run the cells in the notebook.

# Dependencies
• tensorflow==2.6.0
• keras==2.6.0
• pandas==1.1.5
• numpy==1.19.5
• matplotlib==3.2.2
• opencv-python==4.5.3.56
