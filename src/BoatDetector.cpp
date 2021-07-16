#include "../include/BoatDetector.h"

#include <utility>

BoatDetector::BoatDetector(Mat input_img, String img_name){
    img = std::move(input_img);
    name = img_name;
}

Mat BoatDetector::image_preprocessing() {

    int dim_max = max(img.rows, img.cols);
    Mat square_img = Mat(Size(dim_max, dim_max), CV_8UC3);
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
    if (img.rows < dim_max){
        bottom = ceil((double)(dim_max-img.rows)/2);
        top = floor((double)(dim_max-img.rows)/2);
    }
    else if (img.cols < dim_max) {
        right = ceil((double)(dim_max-img.cols)/2);
        left = floor((double)(dim_max-img.cols)/2);
    }

    copyMakeBorder(img, square_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    cvtColor(square_img, square_img, COLOR_BGR2HSV);

    vector<Mat> channels, dest;
    split(square_img, channels);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    for(int j=0; j<channels.size(); j++) {
        Mat out;
        clahe->apply(channels[j], out);
        dest.push_back(out);
    }
    merge(dest, square_img);

    Mat filtered, edges;
    GaussianBlur(square_img, filtered, Size(3, 3), 1);
    Canny(filtered, edges, 200, 50);

    Mat output(square_img.size(), CV_8UC4);
    vector<Mat> chan;
    chan.push_back(square_img);
    chan.push_back(edges);
    merge(chan, output);

    resize(output, output, Size(200, 200));

    img = output;

    return img;
}