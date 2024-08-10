/*
    Saugat Malla
*/

/*
    This file contains code to load the pretrained cnn model and use it for prediction on an image not present in the dataset
*/


#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;

int main(){

    // Load the trained neural network model from file
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("./models/trained_digit_model.xml");

    // Read the test image
    cv::Mat testImage = cv::imread("./testImages/7.png", cv::IMREAD_GRAYSCALE);

    // Resize the test image to match the input size of the model
    cv::resize(testImage, testImage, cv::Size(28,28));

    // Reshape the test image to a single row matrix
    cv::Mat flattenedImage = testImage.reshape(1,1);

    // Convert the image to the data type expected by the model
    cv::Mat input;
    flattenedImage.convertTo(input, CV_32F);

    // Perform prediction using the loaded model
    cv::Mat output;
    mlp->predict(input, output);

    // Display the predicted output
    cout<<output<<endl;

    // Extract the predicted class and its probability
    cv::Point classifiedProb;
    double prob;
    cv::minMaxLoc(output, nullptr, &prob, nullptr, &classifiedProb);

    int predictedClass = classifiedProb.x;

    // Output the predicted class and its probability
    cout<<"Predicted Class: "<<predictedClass<<" "<<" with probability: "<< prob <<endl;

    return 0;
}
