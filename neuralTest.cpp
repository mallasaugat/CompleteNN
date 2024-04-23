#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;

int main(){

    // cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/models/trained_digit_model.xml");
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::load("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/models/mlp_mnist_model.xml");

    cv::Mat testImage = cv::imread("/Users/saugatmalla/Documents/NEU-Courses/Sem 2/PRCV/Projects/P6/testImages/5.png", cv::IMREAD_GRAYSCALE);

    cv::resize(testImage, testImage, cv::Size(28,28));

    cv::Mat flattenedImage = testImage.reshape(1,1);

    cv::Mat input;

    flattenedImage.convertTo(input, CV_32F);

    cv::Mat output;
    mlp->predict(input, output);

    cout<<output<<endl;

    cv::Point classifiedProb;
    double prob;
    cv::minMaxLoc(output, nullptr, &prob, nullptr, &classifiedProb);

    int predictedClass = classifiedProb.x;


    cout<<"Predicted Class: "<<predictedClass<<" "<<" with probability: "<< prob <<endl;


    return 0;
}