# HandwrittenDigitClassifier
# This will be a handwritten digit classifier using a 
# Feedforward neural network built from scratch using only NumPy and mathematics.
# The dataset consists of 28x28 images, each pixel being a number between 0 (white) and 255 (black).
# Thus, each image can be represented as a matrix with 784 numbers between 0-255 to represent each pixel value.
# We will have one matrix, with m (m being the number of images in the dataset) columns, 
# each column representing one image in the dataset. We will call this matrix A. 
# Within each column, we get another matrix with 784 rows to represent each pixel in the image it represents.
# We will have 10 classes of digits, 0-9, that the neural network will try to classify the image data into.
# The first layer of the neural network--the input layer--will consist of 784 nodes for each pixel of the given image.
# The second layer is a hidden layer with 10 nodes, and, finally, 
# the output layer is going to consist of 10 nodes representing the digits.
# FORWARD PROPAGATION--a recognition-inference architechture--in the first layer will be done by 
# taking the dot product of a new 10x784 WEIGHT (W) matrix 
# by the A matrix to make a 10xm Z matrix. The values within the W matrix will represent the 
# 7840 connections between the input layer nodes and hidden layer. 
# The Z matrix will also have a BIAS (B) matrix added to the dot product of A and W 
# to indicate patterns beyond those that pass through the origin (think y = mx+b!).
# Next we need an ACTIVATION FUNCTION to take the weighted sum of inputs and apply
# a transformation to produce an output. We will use a ReLU--rectified linear unit. 
# ReLU outputs the input directly if it's positive, or 0 otherwise. ReLU is advantageous because it
# avoids saturation, alleviating the VANISHING GRADIENT PROBLEM--
# when the gradient of the error/loss function become so small that it cannot be used to adjust the weights and bias. 
# ReLU applied to the Z will comprise the hidden layer. The output layer will be equal to a second W matrix, 
# comprised of values representing the connection between the hidden layer 
# and output layer's nodes, dot producted by the hidden layer matrix--ReLU(Z)--added to another bias vector. 
# The final output layer matrix will have the SOFTMAX function--which takes a tuple of K real numbers 
# and turns them into a probability distribution of K possible outcomes
# --as its activation function to take the output matrix and produce probabilities from it, describing how likely each
# final value is to be a class of digit. We will use BACKPROPAGATION to find the error, 
# and see how the previous weights and biases can be adjusted to minimize the error. 
# This will allow the model to "learn" which images correspond to which digits. 
# Once it adjusts, it will be passed back into the forward propagation, 
# then back into backpropagation to minimize the error further. Repeat until fully accurate!
