/*
    Saugat Malla
*/

/*
    This file contains code to load the dataset and preprocess the data for training
*/

#include<iostream>
#include<opencv2/opencv.hpp> // OpenCV library for image processing
#include<fstream> // For file input/output operations

using namespace std;

// Function to read byte images from a file
vector<vector<unsigned char>> readbyteImages(const string& fName){

    // Open the file in binary mode
    ifstream file(fName, ios::binary);

    // Check if the file opened successfully
    if(!file){
        cerr<<"Failed to open the file: "<<fName<<endl;
        return {}; // Return an empty vector if file opening fails
    }

    // Arrays to store metadata information from the file
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    // Read metadata from the file
    file.read(magicNumber,4);
    file.read(numImagesBytes,4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);

    // Convert byte arrays to integers
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | 
                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) | 
                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) | 
                    (static_cast<unsigned char>(numImagesBytes[3]));
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | 
                    (static_cast<unsigned char>(numRowsBytes[1]) << 16) | 
                    (static_cast<unsigned char>(numRowsBytes[2]) << 8) | 
                    (static_cast<unsigned char>(numRowsBytes[3]));
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | 
                    (static_cast<unsigned char>(numColsBytes[1]) << 16) | 
                    (static_cast<unsigned char>(numColsBytes[2]) << 8) | 
                    (static_cast<unsigned char>(numColsBytes[3]));

    // Vector to store images
    vector<vector<unsigned char>> images;

    // Loop through each image in the file
    for(int i=0;i<numImages;i++){

        // Vector to store pixel values of the current image
        vector<unsigned char> image(numRows * numCols);
        
        // Read pixel values from the file
        file.read(reinterpret_cast<char*>(image.data()), numRows * numCols);

        // Add the image to the vector of images
        images.push_back(image);
    }

    // Close the file
    file.close();

    // Return the vector of images
    return images;
}

// Function to read byte labels from a file
vector<vector<unsigned char>> readbyteLabels(const string& fName){

    // Open the file in binary mode
    ifstream file(fName, ios::binary);

    // Check if the file opened successfully
    if(!file){
        cerr<<"Failed to open the file: "<<fName<<endl;
        return {}; // Return an empty vector if file opening fails
    }

    // Arrays to store metadata information from the file
    char magicNumber[4];
    char numImagesBytes[4];

    // Read metadata from the file
    file.read(magicNumber, 4);
    file.read(numImagesBytes,4);

    // Convert byte array to integer
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | 
                    (static_cast<unsigned char>(numImagesBytes[1]) << 16) | 
                    (static_cast<unsigned char>(numImagesBytes[2]) << 8) | 
                    (static_cast<unsigned char>(numImagesBytes[3]));

    // Vector to store labels
    vector<vector<unsigned char>> labels;

    // Loop through each label in the file
    for(int i=0;i<numImages;i++){

        // Vector to store label of the current image
        vector<unsigned char> image(1);
        
        // Read label from the file
        file.read((char*)(image.data()), 1);

        // Add the label to the vector of labels
        labels.push_back(image);
    }

    // Close the file
    file.close();

    // Return the vector of labels
    return labels;
}

// Function to load dataset into OpenCV Mats
void loadDataset(vector<vector<unsigned char>> imageFile, vector<vector<unsigned char>> labelFile, vector<cv::Mat> &imagesValue, vector<int> &labelsValue){

    // Loop through each image and label pair
    for(int curImg=0; curImg<(int)imageFile.size(); curImg++){

        int curRow = 0;
        int curCol = 0;

        // Create a black image of size 28x28
        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28,28), CV_8UC1);

        // Loop through each pixel value of the current image
        for(int imgValue=0;imgValue<(int)imageFile[curImg].size();imgValue++){

            // Set pixel value in the temporary image
            tempImg.at<uchar>(cv::Point(curCol++, curRow)) = (int)imageFile[curImg][imgValue];

            // Check if the row is filled, move to the next row
            if( (imgValue) % 28 == 0 ){
                curRow++;
                curCol = 0;
            }
        }

        // Add the temporary image and label to the vectors
        imagesValue.push_back(tempImg);
        labelsValue.push_back((int)labelFile[curImg][0]);
    }
}

// Function to prepare dataset for training
void prepareDataset(cv::Mat &trainingData, cv::Mat &labelData, vector<cv::Mat> imagesValue, vector<int> labelsValue, int outputLayer){
    
    // Loop through each image
    for(int i=0;i<imagesValue.size(); i++){

        // Flatten the image and convert to float
        cv::Mat image = imagesValue[i].reshape(1, 1);
        image.convertTo(trainingData.row(i), CV_32F);

        // Create a one-hot encoded label
        cv::Mat label = cv::Mat::zeros(1, outputLayer, CV_32F);
        label.at<float>(0, labelsValue[i]) = 1.0;
        label.copyTo(labelData.row(i));
    }
}
