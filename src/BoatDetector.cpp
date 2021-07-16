#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include "../include/BoatDetector.h"
#include "../include/functions.h"
#include <utility>

BoatDetector::BoatDetector(Mat input_img, String img_name, Size dim){
    img = std::move(input_img);
    name = std::move(img_name);
    init_dim = img.size();
    dim_max = max(init_dim.height, init_dim.width);
    new_dim = std::move(dim);
    mask = Mat(init_dim, CV_8UC1, Scalar(0));
}

Mat BoatDetector::image_preprocessing() {

    Mat square_img = Mat(Size(dim_max, dim_max), CV_8UC3);
    int top, bottom, left, right;

    extract_squared_padding(init_dim, dim_max, top, bottom, left, right);

    copyMakeBorder(img, square_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    cvtColor(square_img, square_img, COLOR_BGR2HSV);

    vector<Mat> channels, dest;
    split(square_img, channels);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    for(auto & channel : channels) {
        Mat out;
        clahe->apply(channel, out);
        dest.push_back(out);
    }
    merge(dest, square_img);

    Mat filtered, edges;
    GaussianBlur(square_img, filtered, Size(3, 3), 1);

//    Canny(filtered, edges, 200, 50);
//    Mat output(square_img.size(), CV_8UC3);
//    vector<Mat> chan;
//    chan.push_back(square_img);
//    chan.push_back(edges);
//    merge(chan, output);

    resize(filtered, filtered, new_dim);

    img = filtered;
    return img;
}

Mat BoatDetector::mask_preprocessing(const String& path_label){
    bool file_exists = access(path_label.c_str(), 0) == 0;

    if (!file_exists){
        cerr << "Mask file not existing for mask: " << path_label;
        exit(1);
    }

    String x;
    ifstream inFile;
    inFile.open(path_label);

    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);
    }

    while (inFile >> x) {
        vector<String> coordinates = split_line(x, ':');
        coordinates = split_line(coordinates[1], ';');

        Point p1(stoi(coordinates[0]), stoi(coordinates[2]));
        Point p2(stoi(coordinates[0]), stoi(coordinates[3]));
        Point p3(stoi(coordinates[1]), stoi(coordinates[3]));
        Point p4(stoi(coordinates[1]), stoi(coordinates[2]));

        vector<vector<Point>> pts = {{p1, p2, p3, p4}};
        fillPoly(mask, pts, Scalar(255));
    }

    inFile.close();

    int top, bottom, left, right;

    extract_squared_padding(init_dim, dim_max, top, bottom, left, right);

    Mat square_mask = Mat(Size(dim_max, dim_max), CV_8UC1);

    copyMakeBorder(mask, square_mask, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    resize(square_mask, mask, new_dim);

    return mask;
}

void BoatDetector::make_prediction(dnn::Net& net){

//    vector<Mat> output;
//    dnn::Model model ("model/vgg.pb");
//    model.setInputSize(Size(new_dim));
//    model.predict(img, output);

    Mat blob;
    dnn::blobFromImage(img, blob);
    net.setInput(blob);
    auto output = net.forward();

    vector<float> out;

    for (int j=0; j<output.size().width; j++){
//        for (int k=0; j<output.size().height; k++) {
            out.push_back(output.at<float>(0, j));
        }

    Mat out_matrix (Size(out.size(), 1), CV_32FC1, out.data());
    cout << out_matrix.size() << endl;
    imshow("Predicted", out_matrix);
    waitKey(0);

}