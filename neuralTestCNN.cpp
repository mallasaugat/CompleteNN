/*
    Saugat Malla
*/

/*
    This file contains code to load the pretrained CNN model and use it for prediction
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

using namespace std;

int main(){

    // Load the ONNX model
    cv::dnn::Net model = cv::dnn::readNetFromONNX("model.onnx");

    // Read and preprocess the test image
    cv::Mat testImage = cv::imread("testImages/1.jpeg", cv::IMREAD_GRAYSCALE);
    cv::resize(testImage, testImage, cv::Size(28,28));

    // Convert the test image to a format suitable for input to the model
    cv::Mat inputBlob = cv::dnn::blobFromImage(testImage, 1.0 / 255, cv::Size(28, 28), cv::Scalar(0), false, false);

    // Set the input blob as the input to the model
    model.setInput(inputBlob);

    // Perform forward pass to get the output
    cv::Mat output = model.forward();

    // Get the predicted class and its probability
    cv::Point classifiedProb;
    double prob;
    cv::minMaxLoc(output, nullptr, &prob, nullptr, &classifiedProb);

    int predictedClass = classifiedProb.x;

    // Output the predicted class and its probability
    cout << "Predicted Class: " << predictedClass << " with probability: " << prob << endl;

    return 0;
}
