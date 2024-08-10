/*
    Saugat Malla
*/

/*
    This file contains code to define the neural network and train the NN
*/

#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;

// Function to create and configure a Multilayer Perceptron (MLP) model
cv::Ptr<cv::ml::ANN_MLP> modelMLP(int inputLayer, int hiddenLayer, int outputLayer){

    // Create a pointer to an instance of ANN_MLP
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

    // Define the layers of the MLP
    cv::Mat layers = cv::Mat(3, 1, CV_32SC1);
    layers.row(0) = cv::Scalar(inputLayer);    // Input layer size
    layers.row(1) = cv::Scalar(hiddenLayer);   // Hidden layer size
    layers.row(2) = cv::Scalar(outputLayer);   // Output layer size
    mlp->setLayerSizes(layers);

    // Set the activation function for each neuron in the MLP
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);

    // Set the training method for the MLP (Backpropagation)
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);

    // Set termination criteria for the training process
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 0.01));

    return mlp; // Return the configured MLP model
}

// Function to train the MLP model
void mlpTrain(cv::Ptr<cv::ml::ANN_MLP> mlp, cv::Mat trainingData, cv::Mat labelData){

    // Train the MLP model using the provided training data and labels
    mlp->train(trainingData, cv::ml::ROW_SAMPLE, labelData);

    // Save the trained MLP model to a file
    mlp->save("./models/mlp_mnist_model.xml");

}
