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
    new_dim = std::move(dim);
    processed_img = Mat(new_dim, CV_32FC3, Scalar(0));
    mask = Mat(init_dim, CV_8UC1, Scalar(0));
    predicted_mask = Mat::zeros(new_dim, CV_8UC1);
    contours = vector<vector<Point>>();
    hierarchy = vector<Vec4i>();
}

Mat BoatDetector::image_preprocessing() {

    Mat square_img = Mat(Size(init_dim_max, init_dim_max), CV_8UC3);
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
    GaussianBlur(square_img, filtered, Size(3, 3), 1);

//    Canny(filtered, edges, 200, 50);
//    Mat output(square_img.size(), CV_8UC3);
//    vector<Mat> chan;
//    chan.push_back(square_img);
//    chan.push_back(edges);
//    merge(chan, output);

    resize(filtered, filtered, new_dim);

    processed_img = filtered;
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
    resize(square_mask, mask, new_dim);

    return mask;
}

Mat BoatDetector::make_prediction(dnn::Net& net){

    Mat blob;
    vector<float> out;
    Mat out_matrix (Size(new_dim), CV_16FC1);

    auto layers = net.getLayerNames();

    dnn::blobFromImage(processed_img, blob, 1.0, new_dim, 0);
    net.setInput(blob);
    auto output = net.forward();

    for (int j=0; j<output.size().width; j++){
        out.push_back(output.at<float>(0, j));
    }

    resize(out, out_matrix, new_dim);
    cout << out_matrix.size() << endl;

//    for (int j=0; j<out_matrix.size().width; j++)
//        for (int k=0; k<out_matrix.size().height; k++) {
//            cout << out_matrix.at<float>(k, j) << endl;
//    }
//
//    imshow("Predicted", out_matrix);
//    waitKey(0);

    return out_matrix;
}

void BoatDetector::prediction_processing(Mat pred_mask){

    for (int j=0; j<pred_mask.size().width; j++)
        for (int k=0; k<pred_mask.size().height; k++)
            if (pred_mask.at<float>(k, j) < 0.5)
                pred_mask.at<float>(k, j) = 0;
            else
                pred_mask.at<float>(k, j) = 1;

    pred_mask = pred_mask * 255;

    findContours(mask, contours, RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);

//    predicted_mask = processed_img.clone();

    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(predicted_mask, contours, -1, color, 2, LINE_8, hierarchy, 0);
//    drawContours(predicted_detection, contours, -1, color, 2, LINE_8, hierarchy, 0);

}

void BoatDetector::apply_prediction_to_input(){

    int top, bottom, left, right;
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);

    Mat initial_dim_predicted_mask;
    resize(predicted_mask, initial_dim_predicted_mask, Size(init_dim_max, init_dim_max));
    initial_dim_predicted_mask = initial_dim_predicted_mask(Range(top, init_dim_max-bottom), Range(left, init_dim_max-right));

    cout << initial_dim_predicted_mask.size() << endl;
    cout << img.size() << endl;

    // --- da aggiungere questa maschera all'immagine iniziale

    Mat square_img;
    copyMakeBorder(img, square_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    resize(square_img, square_img, new_dim);

    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(square_img, contours, -1, color, 2, LINE_8, hierarchy, 0);

    resize(square_img, square_img, Size(init_dim_max, init_dim_max));
    Mat correct_size_prediction = square_img(Range(top, init_dim_max-bottom), Range(left, init_dim_max-right));

    imshow("Contours", correct_size_prediction);
    waitKey(0);
}