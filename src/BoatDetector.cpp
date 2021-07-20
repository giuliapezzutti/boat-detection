#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include "../include/BoatDetector.h"
#include "../include/functions.h"
#include <utility>

RNG rng(12345);

BoatDetector::BoatDetector(Mat input_img, String img_name, Size dim){
    img = std::move(input_img);
    name = std::move(img_name);
    init_dim = img.size();
    init_dim_max = max(init_dim.height, init_dim.width);
    net_dim = std::move(dim);
    processed_img = Mat(net_dim, CV_32FC4, Scalar(0));
    mask = Mat(init_dim, CV_8UC1, Scalar(0));
    predicted_mask = Mat::zeros(net_dim, CV_8UC1);
    net_output = Mat(net_dim, CV_32FC1, Scalar(0));
    contours = vector<vector<Point>>();
    hierarchy = vector<Vec4i>();
}

Mat BoatDetector::image_preprocessing() {

    Mat square_img = Mat(Size(init_dim_max, init_dim_max), CV_32FC4);
    int top, bottom, left, right;

    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);

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
//    GaussianBlur(square_img, filtered, Size(3, 3), 1);

//    Canny(filtered, edges, 200, 50);
//    Mat output(square_img.size(), CV_8UC3);
//    vector<Mat> chan;
//    chan.push_back(square_img);
//    chan.push_back(edges);
//    merge(chan, output);

    resize(square_img, square_img, net_dim);

    processed_img = square_img;
    return processed_img;
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
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);

    Mat square_mask = Mat(Size(init_dim_max, init_dim_max), CV_8UC1);

    copyMakeBorder(mask, square_mask, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    resize(square_mask, mask, net_dim);

    return mask;
}

void BoatDetector::make_prediction(dnn::Net& net) {

    auto layers = net.getLayerNames();
//    for (auto layer : layers)
//        cout << layer << endl;

    int sz[] = {1, net_dim.height, net_dim.width, 3};
    vector<float> mean;
    mean.push_back(1.0/127.5);
    mean.push_back(1.0/127.5);
    mean.push_back(1.0/127.5);
    float scale = 1/127.5;
    Mat blob(4, sz, CV_32F, processed_img.data);
    vector<Mat> output_img;

    dnn::blobFromImage(processed_img, blob, scale, net_dim, 1.0/127.5);
    net.setInput(blob);
    auto output = net.forward();
    dnn::imagesFromBlob(output, output_img);

    output_img[0].copyTo(net_output);

    imshow("Prediction", net_output);
    waitKey(0);
}

void BoatDetector::prediction_processing() {

    Mat final_mask (net_dim, CV_8UC1);

    for (int j=0; j<net_output.size().width; j++)
        for (int k=0; k<net_output.size().height; k++)
            if (net_output.at<float>(k, j) < 0.98)
                final_mask.at<u_char>(k, j) = 0;
            else
                final_mask.at<u_char>(k, j) = 255;

    imshow("thresholded mask", final_mask);
    waitKey(0);

    Mat canny_output;
    Canny(final_mask, canny_output, 200, 400);

    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point> > contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i],contours_poly[i],3,true);
        boundRect[i] = boundingRect(contours_poly[i]);
    }

    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
    for(size_t i = 0; i< contours.size(); i++){
        drawContours(predicted_mask, contours_poly, (int)i, color);
        rectangle(predicted_mask, boundRect[i].tl(), boundRect[i].br(), color, 1);
    }

    imshow("Prediction", predicted_mask);
    waitKey(0);
}

void BoatDetector::apply_prediction_to_input(){

    int top, bottom, left, right;
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);

    Mat initial_dim_predicted_mask;
    resize(predicted_mask, initial_dim_predicted_mask, Size(init_dim_max, init_dim_max));
    initial_dim_predicted_mask = initial_dim_predicted_mask(Range(top, init_dim_max-bottom), Range(left, init_dim_max-right));

    imshow("initial_dim_predicted_mask", initial_dim_predicted_mask);
    waitKey(0);

    vector<Mat> three;
    three.push_back(initial_dim_predicted_mask);
    three.push_back(initial_dim_predicted_mask);
    three.push_back(initial_dim_predicted_mask);
    Mat three_channels_initial_dim_predicted_mask;
    merge(three, three_channels_initial_dim_predicted_mask);

    Mat img_masked = img.clone();
    bitwise_and(img, three_channels_initial_dim_predicted_mask, img_masked);

    imshow("Masked image", img_masked);
    waitKey(0);

}