/*
    Saugat Malla
*/

/*
    This file contains code to load the trained model and check the accuracy of the model
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/dnn.hpp>

#include "datasetMnist.h" // Header file for reading MNIST dataset


// Function to read IDX1-UBYTE label files (MNIST label files)
std::vector<unsigned char> readIDX1UByteLabelFile(const std::string& filename) {
    // Open the IDX1-UBYTE label file in binary mode
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX1-UBYTE label file." << std::endl;
        return {}; // Return an empty vector if file opening fails
    }

    // Read the IDX1-UBYTE label file header
    char magicNumber[4];
    char numLabelsBytes[4];

    file.read(magicNumber, 4);
    file.read(numLabelsBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numLabels = (static_cast<unsigned char>(numLabelsBytes[0]) << 24) | (static_cast<unsigned char>(numLabelsBytes[1]) << 16) | (static_cast<unsigned char>(numLabelsBytes[2]) << 8) | static_cast<unsigned char>(numLabelsBytes[3]);

    // Initialize a vector to store the labels
    std::vector<unsigned char> labels(numLabels);

    // Read labels
    file.read(reinterpret_cast<char*>(labels.data()), numLabels);

    file.close();

    return labels; // Return the vector of labels
}

int main() {
    // Load the trained model
    cv::Ptr<cv::ml::ANN_MLP> model = cv::ml::StatModel::load<cv::ml::ANN_MLP>("./models/trained_digit_model.xml");
    if (model.empty()) {
        std::cerr << "Failed to load the trained model." << std::endl;
        return -1;
    }

    // Load test data
    std::string test_filename = "./Dataset/Mnist/t10k-images.idx3-ubyte";
    std::string test_label_filename = "./Dataset/Mnist/t10k-labels.idx1-ubyte";

    std::vector<std::vector<unsigned char>> test_imagesFile = readbyteImages(test_filename); // Read test images
    std::vector<unsigned char> test_labelsFile = readIDX1UByteLabelFile(test_label_filename); // Read test labels

    // Prepare test data
    std::vector<cv::Mat> test_imagesData;
    std::vector<int> test_labelsData;

    for (size_t i = 0; i < test_imagesFile.size(); ++i) {
        // Reshape each image and convert to OpenCV Mat
        cv::Mat image = cv::Mat(test_imagesFile[i]).reshape(1, 28);
        test_imagesData.push_back(image);
        test_labelsData.push_back(static_cast<int>(test_labelsFile[i])); // Store test labels as integers
    }

    // Evaluate the model
    int correct_predictions = 0;
    int total_predictions = test_imagesData.size();

    for (size_t i = 0; i < test_imagesData.size(); ++i) {
        // Preprocess the test image
        cv::Mat testImage = test_imagesData[i];
        cv::resize(testImage, testImage, cv::Size(28, 28)); // Resize to match model input size
        cv::Mat flattenedImage = testImage.reshape(1, 1);
        cv::Mat input;
        flattenedImage.convertTo(input, CV_32F);

        // Predict the label using the trained model
        cv::Mat predicted;
        model->predict(input, predicted);

        // Find the predicted label
        cv::Point maxLoc;
        cv::minMaxLoc(predicted, nullptr, nullptr, nullptr, &maxLoc);
        int predictedLabel = maxLoc.x;

        // Compare with ground truth label
        int trueLabel = test_labelsData[i];
        if (predictedLabel == trueLabel)
            correct_predictions++;
    }

    // Calculate accuracy
    double accuracy = static_cast<double>(correct_predictions) / total_predictions;
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}
