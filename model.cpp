#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;

cv::Ptr<cv::ml::ANN_MLP> modelMLP(int inputLayer, int hiddenLayer, int outputLayer){

    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
    cv::Mat layers = cv::Mat(3,1, CV_32SC1);
    layers.row(0) = cv::Scalar(inputLayer);
    layers.row(1) = cv::Scalar(hiddenLayer);
    layers.row(2) = cv::Scalar(outputLayer);
    mlp->setLayerSizes(layers);
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001);
    mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 0.01));

    return mlp;

}

void mlpTrain(cv::Ptr<cv::ml::ANN_MLP> mlp, cv::Mat trainingData, cv::Mat labelData){

    mlp->train(trainingData, cv::ml::ROW_SAMPLE, labelData);

    mlp->save("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/models/mlp_mnist_model.xml");

}

