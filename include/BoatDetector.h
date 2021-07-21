#ifndef BOAT_DETECTION_BOATDETECTOR_H
#define BOAT_DETECTION_BOATDETECTOR_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include <opencv2/dnn/dnn.hpp>

using namespace std;
using namespace cv;

class BoatDetector {

public:
    BoatDetector(Mat input_img, String img_name, Size dim);
    Mat image_preprocessing();
    Mat mask_preprocessing(const String& path_label);
    void make_prediction(dnn::Net& net);
    void prediction_processing();
    void apply_prediction_to_input();
    float prediction_evaluation();

protected:
    Mat img;
    Mat processed_img;
    String name;
    Size init_dim;
    int init_dim_max;
    Size net_dim;
    Mat mask;
    Mat predicted_mask;
    Mat net_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
};

#endif
