#ifndef DATASET_H
#define DATASET_H

std::vector<std::vector<unsigned char>> readbyteImages(const std::string& fName);

std::vector<std::vector<unsigned char>> readbyteLabels(const std::string& fName);

void loadDataset(std::vector<std::vector<unsigned char>> imageFile, std::vector<std::vector<unsigned char>> labelFile, std::vector<cv::Mat> &imagesValue, std::vector<int> &labelsValue);

void prepareDataset(cv::Mat &trainingData, cv::Mat &labelData,std::vector<cv::Mat> imagesValue, std::vector<int> labelsValue, int outputLayer);

#endif