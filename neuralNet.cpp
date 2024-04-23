#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>

#include "datasetMnist.h"
#include "model.h"

using namespace std;

int main(){

    string fileName = "/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/Dataset/Mnist/train-images.idx3-ubyte";
    string lfName = "/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/Dataset/Mnist/train-labels.idx1-ubyte";

    //  Loading dataset
    vector<vector<unsigned char>> imageFile = readbyteImages(fileName);
    vector<vector<unsigned char>> labelFile = readbyteLabels(lfName);

    vector<cv::Mat> imagesValue;
    vector<int> labelsValue;
    loadDataset(imageFile, labelFile, imagesValue, labelsValue);

    cout<<"Loaded Data (Images and Labels): "<<imagesValue.size()<<" "<<labelsValue.size()<<endl;


    // Preparing Dataset
    int inputLayer = imagesValue[0].total();
    int hiddenLayer = 100;
    int outputLayer = 10;
    int numSamples = imagesValue.size();
    cv::Mat trainingData(numSamples, inputLayer, CV_32F);
    cv::Mat labelData(numSamples, outputLayer, CV_32F);

    prepareDataset(trainingData, labelData, imagesValue, labelsValue, outputLayer);
    cout<<"Training Data size:"<<trainingData.size()<<" "<<labelData.size()<<endl;

    // Training
    cv::Ptr<cv::ml::ANN_MLP> mlp;
    mlp = modelMLP(inputLayer, hiddenLayer, outputLayer);
    mlpTrain(mlp, trainingData, labelData);


    return 0;
}