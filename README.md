# Deep-Learning---Exp-7

## **Implement an Autoencoder in TensorFlow/Keras**

## **AIM**

To develop a convolutional autoencoder for image denoising application.

## **PROBLEM STATEMENT AND DATASET**

Autoencoder is an unsupervised artificial neural network that is trained to copy its input to output. An autoencoder will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.The goal of an autoencoder is to get an output that is identical to the input. Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image. We are using MNIST Dataset for this experiment. The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.


## **Neural Network Model**

<img width="890" height="471" alt="image" src="https://github.com/user-attachments/assets/1c42bdb1-effe-49e3-8461-4c96b8897007" />


## **DESIGN STEPS**

**STEP 1:** Load and Preprocess Data

  - Load the MNIST dataset of handwritten digits.

  - Normalize pixel values between 0 and 1 for faster convergence.

  - Reshape images to (28, 28, 1) for CNN input compatibility.


**STEP 2:** Add Noise to Images

  - Generate Gaussian noise and add it to both training and testing images.

  - Use np.clip() to keep pixel values within valid range [0, 1].

  - The noisy images act as inputs, and the clean images as targets.

**STEP 3**: Build the Encoder Network

  - Stack multiple Conv2D and MaxPooling2D layers.

  - Gradually reduce spatial dimensions to compress the input image into a latent feature representation.

**STEP 4:** Build the Decoder Network

   - Stack Conv2D and UpSampling2D layers.

  - Gradually reconstruct the original image size from the compressed encoding.

  - Use sigmoid activation in the final layer to generate pixel values between 0 and 1.

**STEP 5:** Compile and Train the Model

  - Compile using Adam optimizer and binary crossentropy loss.

  - Train the model with noisy images as input and clean images as output for a few epochs.

  - Validate performance using test data.

**STEP 6:** Evaluate and Visualize Results

  - Predict denoised images using the trained model.

  - Plot loss vs. validation loss to monitor performance.

  - Display side-by-side comparison of original, noisy, and denoised images using Matplotlib.

## **PROGRAM**

**Name:** SOWMIYA G

**Register Number:** 2305002023

``` PYTHON
from tensorflow.keras import layers, models, Input, datasets
import numpy as np, matplotlib.pyplot as plt, pandas as pd

(x_train,_),(x_test,_) = datasets.mnist.load_data()
x_train,x_test=[x.astype('float32')/255. for x in (x_train,x_test)]
x_train,x_test=[x.reshape(-1,28,28,1) for x in (x_train,x_test)]
noise=0.5
x_train_n=np.clip(x_train+noise*np.random.normal(0,1,x_train.shape),0,1)
x_test_n=np.clip(x_test+noise*np.random.normal(0,1,x_test.shape),0,1)

inp=Input((28,28,1))
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoded=layers.MaxPooling2D((2,2),padding='same')(x)

x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoded=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

autoencoder=models.Model(inp,decoded)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(x_train_n,x_train,epochs=3,batch_size=256,validation_data=(x_test_n,x_test))

pd.DataFrame(autoencoder.history.history)[['loss','val_loss']].plot()
decoded_imgs=autoencoder.predict(x_test_n)
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    for j,data in enumerate([x_test,x_test_n,decoded_imgs]):
        ax=plt.subplot(3,n,i+1+j*n);plt.imshow(data[i].reshape(28,28),cmap='gray');ax.axis('off')
plt.show()
````

## **OUTPUT**

**Training Loss, Validation Loss Vs Iteration Plot:**

<img width="556" height="413" alt="image" src="https://github.com/user-attachments/assets/64117728-eaf1-42c2-8e13-556b0a140027" />


**Training loss**

**Original vs Noisy Vs Reconstructed Image**

<img width="1529" height="328" alt="image" src="https://github.com/user-attachments/assets/db3f3465-66a9-47d9-a01f-14f78302cbde" />


## **RESULT**

Thus we have successfully developed a convolutional autoencoder for image denoising application.
