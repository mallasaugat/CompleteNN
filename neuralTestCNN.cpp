#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

using namespace std;

int main(){

    // Load the ONNX model
    cv::dnn::Net model = cv::dnn::readNetFromONNX("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6_test/model.onnx");

    // Read and preprocess the test image
    cv::Mat testImage = cv::imread("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6_test/testImages/1.jpeg", cv::IMREAD_GRAYSCALE);
    cv::resize(testImage, testImage, cv::Size(28,28));
    cv::Mat inputBlob = cv::dnn::blobFromImage(testImage, 1.0 / 255, cv::Size(28, 28), cv::Scalar(0), false, false);
    // Set the input to the model
    model.setInput(inputBlob);

    // Perform forward pass to get the output
    cv::Mat output = model.forward();

    // Get the predicted class
    cv::Point classifiedProb;
    double prob;
    cv::minMaxLoc(output, nullptr, &prob, nullptr, &classifiedProb);

    int predictedClass = classifiedProb.x;

    cout << "Predicted Class: " << predictedClass << " with probability: " << prob << endl;

    return 0;
}
