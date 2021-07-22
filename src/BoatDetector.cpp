#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>
#include "../include/BoatDetector.h"
#include "../include/utilities-functions.h"
#include <utility>

RNG rng(12345);

BoatDetector::BoatDetector(Mat input_img, String img_name, Size dim){
    /// Class constructor
    img = std::move(input_img);
    name = std::move(img_name);
    
    init_dim = img.size();
    init_dim_max = max(init_dim.height, init_dim.width);
    net_dim = std::move(dim);
    
    processed_img = Mat(net_dim, CV_32FC4, Scalar(0));
    mask = Mat(init_dim, CV_8UC1, Scalar(0));
    net_output = Mat(net_dim, CV_32FC1, Scalar(0));
    predicted_mask = Mat(net_dim, CV_8UC1, Scalar(0));

    contours = vector<vector<Point>>();
}

Mat BoatDetector::image_preprocessing() {
    /// Image preprocessing done at different steps

    Mat square_img = Mat(Size(init_dim_max, init_dim_max), CV_32FC4);
    Mat filtered, edges, laplacian;
    int top, bottom, left, right;

    // Padding to create a square matrix
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);
    copyMakeBorder(img, square_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));

    // Converto to HSV colour space
    cvtColor(square_img, square_img, COLOR_BGR2GRAY);

    // Equalize histograms for each channels thanks to CLAHE
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

    // Noise removal through Gaussian filtering
    GaussianBlur(square_img, filtered, Size(3, 3), 0.1);

    // Canny edge detector
    Canny(filtered, edges, 200, 50);

    // Creation of the new image with channels: S, V, Canny
    Mat output(square_img.size(), CV_8UC3);
    vector<Mat> chan;
    split(filtered, channels);
    chan.push_back(channels.at(0));
    chan.push_back(channels.at(1));
    chan.push_back(edges);
    merge(chan, output);

//    erode(output, output, Mat());

    // Resize of the image according to the network input shape
    resize(filtered, processed_img, net_dim);

    return processed_img;
}

Mat BoatDetector::mask_preprocessing(const String& path_label){
    /// Mask preprocessing: from the txt file path containing the vertex indexes, 
    ///it determines the 0-1 associated mask 
    
    // Check that the file exists, otherwise send an error
    bool file_exists = access(path_label.c_str(), 0) == 0;
    if (!file_exists){
        cerr << "Mask file not existing for mask: " << path_label;
        exit(1);
    }

    // Open the corresponding txt file
    String x;
    ifstream inFile;
    inFile.open(path_label);
    if (!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);
    }

    // Read each file line independently and extract the coordinates
    while (inFile >> x) {
        vector<String> coordinates = split_line(x, ':');
        coordinates = split_line(coordinates[1], ';');
        
        // Determine the associated points
        Point p1(stoi(coordinates[0]), stoi(coordinates[2]));
        Point p2(stoi(coordinates[0]), stoi(coordinates[3]));
        Point p3(stoi(coordinates[1]), stoi(coordinates[3]));
        Point p4(stoi(coordinates[1]), stoi(coordinates[2]));
        vector<vector<Point>> pts = {{p1, p2, p3, p4}};
    
        // Create the rectangle on the mask
        fillPoly(mask, pts, Scalar(255));
    }

    inFile.close();

    // Pad the mask to make it square
    Mat square_mask = Mat(Size(init_dim_max, init_dim_max), CV_8UC1);
    int top, bottom, left, right;
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);
    copyMakeBorder(mask, square_mask, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
    
    // Resize of the image according to the network input shape
    resize(square_mask, mask, net_dim);

    return mask;
}

void BoatDetector::make_prediction(dnn::Net& net) {
    /// Making of the mask prediction thanks to the input neural network for the current image

    // Parameters for the blob extraction (during training, the pixel values have been divided by 255
    int sz[] = {1, net_dim.height, net_dim.width, 3};
    vector<float> mean;
    mean.push_back(1.0/127.5);
    mean.push_back(1.0/127.5);
    mean.push_back(1.0/127.5);
    float scale = 1/127.5;
    Mat blob(4, sz, CV_32F, processed_img.data);
    vector<Mat> output_img;

    // Blob extraction
    dnn::blobFromImage(processed_img, blob, scale, net_dim, 1.0/127.5);
    
    // Set input to the network and forward it to obtain the prediction  
    net.setInput(blob);
    auto output = net.forward();
    
    // Extract the image correspondent to the prediction (one for each input image)
    dnn::imagesFromBlob(output, output_img);
    output_img[0].copyTo(net_output);

    imshow("Prediction", net_output);
}

void BoatDetector::prediction_processing() {
    /// Processing of the predicted mask: 0-255 mask is obtained through thresholding
    /// and since rectangular boxes are present in the ground-truth masks, correspondent
    /// contours and rectangular shapes are extracted

    Mat final_mask (net_dim, CV_8UC1);

    // Prediction thresholding (and back to 0-255 mask)
    for (int j=0; j<net_output.size().width; j++)
        for (int k=0; k<net_output.size().height; k++)
            if (net_output.at<float>(k, j) < 0.99)
                final_mask.at<u_char>(k, j) = 0;
            else
                final_mask.at<u_char>(k, j) = 255;

    // Canny edge extraction 
    Mat canny_output;
    Canny(final_mask, canny_output, 200, 400);

    // Extraction of the contours in the mask
    findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    
    // Analysis of each contour and determination of its rectangular bound
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i],contours_poly[i],3,true);
        boundRect[i] = boundingRect(contours_poly[i]);
    }

    // Drawing of the rectangular bound on the mask (it is completely filled)
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
    for(size_t i = 0; i< contours.size(); i++){
        drawContours(predicted_mask, contours_poly, (int)i, color);
        rectangle(predicted_mask, boundRect[i].tl(), boundRect[i].br(), color, -1);
    }
}

void BoatDetector::apply_prediction_to_input(){
    /// Application of the prediction to the input image to make it visualizable by the user

    // Inverse padding operation to come to the original shape and size
    Mat initial_dim_predicted_mask;
    int top, bottom, left, right;
    extract_squared_padding(init_dim, init_dim_max, top, bottom, left, right);
    resize(predicted_mask, initial_dim_predicted_mask, Size(init_dim_max, init_dim_max));
    initial_dim_predicted_mask = initial_dim_predicted_mask(Range(top, init_dim_max-bottom), Range(left, init_dim_max-right));

    // Extract the contours from the correct-dimensioned mask and report them in the input image (without processing)
    vector<vector<Point>> cnt;
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256));
    findContours(initial_dim_predicted_mask, cnt, RETR_LIST, CHAIN_APPROX_NONE);

    Mat predicted_img = img.clone();
    drawContours(predicted_img, cnt, -1, color, 3);

    imshow("Predicted image", predicted_img);

//    imwrite("images/"+name+".png", predicted_img);
}

float BoatDetector::prediction_evaluation(){
    /// Evaluation of the prediction with respect to the known ground truth thanks to Intersection over Union metric

    // computation of intersection and union
    Mat intersection_mask, union_mask;
    bitwise_and(mask, predicted_mask, intersection_mask);
    bitwise_or(mask, predicted_mask, union_mask);

    // Count of the number of elements in each computed mask
    int count_int = countNonZero(intersection_mask);
    int count_uni = countNonZero(union_mask);

    // Calculation of the distance by definition
    float iou = (float)count_int/(float)count_uni;
    cout << "Intersection over union for the current prediction: " << iou << endl;

    return iou;
}