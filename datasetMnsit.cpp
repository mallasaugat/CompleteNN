#include<iostream>
#include<opencv2/opencv.hpp>
#include<fstream>

using namespace std;

vector<vector<unsigned char>> readbyteImages(const string& fName){
    
    ifstream file(fName, ios::binary);

    if(!file){
        cerr<<"Failed to open the file: "<<fName<<endl;
        return {};
    }

    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber,4);
    file.read(numImagesBytes,4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);

    cout<<static_cast<int>(numImagesBytes[0])<<" "<<static_cast<int>(numImagesBytes[1])<<" "<<(int)static_cast<unsigned char>(numImagesBytes[2])<<" "<<static_cast<int>(numImagesBytes[3])<<" "<<endl;

    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | (static_cast<unsigned char>(numImagesBytes[3]));
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | (static_cast<unsigned char>(numRowsBytes[1]) << 16) | (static_cast<unsigned char>(numRowsBytes[2]) << 8) | (static_cast<unsigned char>(numRowsBytes[3]));
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | (static_cast<unsigned char>(numColsBytes[1]) << 16) | (static_cast<unsigned char>(numColsBytes[2]) << 8) | (static_cast<unsigned char>(numColsBytes[3]));

    cout<<numImages<<" "<<numRows<<" "<<numCols;

    vector<vector<unsigned char>> images;

    for(int i=0;i<numImages;i++){

        vector<unsigned char> image(numRows * numCols);
        file.read((char*)(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}


vector<vector<unsigned char>> readbyteLabels(const string& fName){

    ifstream file(fName, ios::binary);

    if(!file){
        cerr<<"Failed to open the file: "<<fName<<endl;
        return {};
    }

    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes,4);

    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | (static_cast<unsigned char>(numImagesBytes[3]));

    vector<vector<unsigned char>> labels;

    for(int i=0;i<numImages;i++){

        vector<unsigned char> image(1);
        file.read((char*)(image.data()), 1);

        labels.push_back(image);

    }

    file.close();

    return labels;
}


void loadDataset(vector<vector<unsigned char>> imageFile, vector<vector<unsigned char>> labelFile, vector<cv::Mat> &imagesValue, vector<int> &labelsValue){

    for(int curImg=0; curImg<(int)imageFile.size(); curImg++){

        int curRow = 0;
        int curCol = 0;

        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28,28), CV_8UC1);

        for(int imgValue=0;imgValue<(int)imageFile[curImg].size();imgValue++){

            // cout<<(int)imageFile[curImg][imgValue]<<" ";
            if(curRow == 28){
                break;
            }

            tempImg.at<uchar>(cv::Point(curCol++, curRow)) = (int)imageFile[curImg][imgValue];

            if( (imgValue) % 28 == 0 ){

                curRow++;
                curCol = 0;

                // cout<<endl;

            }

        }

        cout<<(int)labelFile[curImg][0]<<endl;

        imagesValue.push_back(tempImg);
        labelsValue.push_back((int)labelFile[curImg][0]);

        // cv::imshow("Cur Image",tempImg);
        // cv::waitKey(0);

    }

}


void prepareDataset(cv::Mat &trainingData, cv::Mat &labelData, vector<cv::Mat> imagesValue, vector<int> labelsValue, int outputLayer){
    
   
    for(int i=0;i<imagesValue.size(); i++){

        cv::Mat image = imagesValue[i].reshape(1, 1);
        image.convertTo(trainingData.row(i), CV_32F);

        cv::Mat label = cv::Mat::zeros(1, outputLayer, CV_32F);
        label.at<float>(0, labelsValue[i]) = 1.0;
        label.copyTo(labelData.row(i));

    }


}