/*
    Saugat Malla
*/

/*
    This file contains code to load the dataset, train the model and save the trained model 
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>

#include "datasetMnist.h" // Header file for functions related to reading MNIST dataset
#include "model.h" // Header file for functions related to creating and training models

#include<opencv2/dnn.hpp>
#include<opencv2/dnn/all_layers.hpp>

using namespace std;

int main(){

    // File paths for MNIST dataset
    string fileName = "./Dataset/Mnist/train-images.idx3-ubyte";
    string lfName = "./Dataset/Mnist/train-labels.idx1-ubyte";

    // Loading dataset
    vector<vector<unsigned char>> imageFile = readbyteImages(fileName); // Read byte images from file
    vector<vector<unsigned char>> labelFile = readbyteLabels(lfName); // Read byte labels from file

    // Vectors to store images and labels
    vector<cv::Mat> imagesValue;
    vector<int> labelsValue;

    // Load dataset into OpenCV Mats
    loadDataset(imageFile, labelFile, imagesValue, labelsValue);

    // Output the number of loaded images and labels
    cout<<"Loaded Data (Images and Labels): "<<imagesValue.size()<<" "<<labelsValue.size()<<endl;

    // Preparing Dataset
    int inputLayer = imagesValue[0].total(); // Total number of pixels in an image (input layer size)
    int hiddenLayer = 100; // Size of the hidden layer
    int outputLayer = 10; // Number of classes (output layer size)
    int numSamples = imagesValue.size(); // Number of samples
    cv::Mat trainingData(numSamples, inputLayer, CV_32F); // Matrix to store training data
    cv::Mat labelData(numSamples, outputLayer, CV_32F); // Matrix to store labels

    // Prepare dataset for training
    prepareDataset(trainingData, labelData, imagesValue, labelsValue, outputLayer);
    cout<<"Training Data size:"<<trainingData.size()<<" "<<labelData.size()<<endl;

    // Training 
    // ANN_MLP
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    mlp = modelMLP(inputLayer, hiddenLayer, outputLayer);
    mlpTrain(mlp, trainingData, labelData);

    return 0;
}
