# Training models for 3D images
- Based on Tensorflow and its Dataset library
- Kaggle Passenger Screening Algorithm Challenge:
Improve the accuracy of the Department of Homeland Security's threat recognition algorithms 
(https://www.kaggle.com/c/passenger-screening-algorithm-challenge)

Since original dataset is not publicly available, the training pipeline based on TF Dataset API can be used and adapted to
read binary data.
The models operate on 3D volumetric image data in different ways, including weight-shared 2D-CNNs for sliced-input, and 3D-CNNs
for a more end-to-end approach.
