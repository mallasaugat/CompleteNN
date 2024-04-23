#ifndef MODEL_H
#define MODEL_H

cv::Ptr<cv::ml::ANN_MLP> modelMLP(int inputLayer, int hiddenLayer, int outputLayer);

void mlpTrain(cv::Ptr<cv::ml::ANN_MLP> mlp, cv::Mat trainingData, cv::Mat labelData);

#endif