#ifndef BOAT_DETECTION_BOATDETECTOR_H
#define BOAT_DETECTION_BOATDETECTOR_H

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class BoatDetector {

public:
    BoatDetector(Mat input_img, String img_name);
    Mat image_preprocessing();

protected:
    Mat img;
    String name;

};


#endif
