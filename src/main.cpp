#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include "../include/functions.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){

//    TODO Set working directory

    String folder_images("data/training_dataset/images/");
    String folder_labels("data/training_dataset/labels_txt/");
    vector<String> path_images;

    glob(folder_images+"*.png", path_images);

    int i, j;
    for (i=0; i< path_images.size(); i++){

        String path_image = path_images[i];
        Mat img = imread(path_image);
        resize(img, img, Size(512, 256));
        cvtColor(img, img, COLOR_BGR2HSV);

        vector<string> split_results = split(path_image, '.');
        split_results = split(split_results[0], '/');
        string name_image = split_results[3];

        String path_label = folder_labels + name_image + ".txt";
        cout << path_label << endl;

        bool file_exists = access(path_label.c_str(), 0) == 0;
        String x;
        Mat mask = Mat(img.size(), CV_8UC1, Scalar(0));

        if (file_exists) {
            ifstream inFile;
            inFile.open(path_label);

            if (!inFile) {
                cerr << "Unable to open file datafile.txt";
                continue;
            }

            while (inFile >> x) {
                vector<String> coordinates = split(x, ':');
                coordinates = split(coordinates[1], ';');

                Point p1(stoi(coordinates[0]), stoi(coordinates[2]));
                Point p2(stoi(coordinates[0]), stoi(coordinates[3]));
                Point p3(stoi(coordinates[1]), stoi(coordinates[3]));
                Point p4(stoi(coordinates[1]), stoi(coordinates[2]));

                vector<vector<Point>> pts = {{p1, p2, p3, p4}};
                fillPoly(mask, pts, Scalar(255));
            }

            inFile.close();

//            // Visualization
//            Mat masked = Mat(img.size(), img.type());
//            vector<Mat> channels;
//            split(img, channels);
//
//            for (j=0; j<3; j++) cv::bitwise_and(channels[j], mask, channels[j]);
//
//            merge(channels, masked);
//
//            imshow("Mask", mask);
//            imshow("Masked", masked);
//            waitKey();
        }

        Mat filtered, edges;
        GaussianBlur(img, filtered, Size(3, 3), 1);
        Canny(filtered, edges, 200, 50);

//        imshow("Not Filtered", img);
//        imshow("Filtered", filtered);
//        imshow("Canny", edges);
//        waitKey();

        Mat output(img.size(), CV_8UC4);
        vector<Mat> channels;
        channels.push_back(img);
        channels.push_back(edges);
        merge(channels, output);

//        imshow("total", output);
//        waitKey();

        String output_path("data/training_dataset/processed_images/");
        String mask_path("data/training_dataset/masks/");

        imwrite(output_path+name_image+".png", output);
        imwrite(mask_path+name_image+".png", mask);

    }

    return 0;
}
