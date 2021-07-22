#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/dnn/dnn.hpp>
#include "../include/utilities-functions.h"
#include "../include/BoatDetector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    // Get the inputs from command line
    String input_imgs, folder_images, folder_masks;
    bool train, eval, pred;
    init(argc, argv, input_imgs, train, eval, pred, folder_masks);

    // Check if the passed image paths correspond to a .png file or a folder: in the latter, extract all .png files path
    vector<String> img_paths;
    if (input_imgs.substr(input_imgs.size() - 4) != ".png" or input_imgs.substr(input_imgs.size() - 4) != ".jpg") {
        folder_images = input_imgs;
        vector<String> x;
        glob(folder_images + "/*.png", x);
        for (auto path : x)
            img_paths.push_back(path);
        glob(folder_images + "/*.jpg", x);
        for (auto path : x)
            img_paths.push_back(path);
    }
    else {
        img_paths.push_back(input_imgs);
        folder_images = extract_folder_from_path(input_imgs);
    }

    // Check that at least one image is passed as input
    if (img_paths.empty()){
        cerr << "The folder must contain at least one image!";
        exit(1);
    }

    // If training task, create the directories to save the image and mask datasets
    String save_path = folder_images + "/training_images";
    String mask_save_path = folder_masks + "/image_masks";
    if (train){
        if (opendir(save_path.c_str()) == nullptr) {
            mkdir(save_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
        if (opendir(mask_save_path.c_str()) == nullptr)
            mkdir(mask_save_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    // Load of the pre-trained model thanks to dnn class
    dnn::Net net;
    if (eval or pred){
        String model_path = "models/model.pb";
        net = dnn::readNetFromTensorflow(model_path);

        // Extraction of the layers name (to check that the loading has been done correctly)
        auto layers = net.getLayerNames();
//        for (auto layer : layers)
//            cout << layer << endl;

        cout << "To go to the subsequent image analysis, press any key after the prediction is shown!" << endl;
    }

    vector<float> iou_vector;

    // Cycle over each image path found (eventually only one)
    for (const auto& path : img_paths){

        // Extract the correspondent image
        string name_image = extract_name_from_path(path);
        Mat img = imread(path);
        cout << "\nAnalyzing image " << name_image << "..." << endl;

        // Create an instance of BoatDetector class
        BoatDetector bd = BoatDetector(img, name_image, Size(512, 512));

        // Perform image preprocessing and, if needed, mask preprocessing
        Mat processed_img = bd.image_preprocessing();
        Mat mask;
        if (train or eval) {
            String path_label = folder_masks + "/" + name_image + ".txt";
            mask = bd.mask_preprocessing(path_label);
        }

        // If the training dataset is required, save image and mask in the correspondent folder
        if (train) {
            String img_save = save_path + "/" + name_image + "_training.png";
            imwrite(img_save, processed_img);
            String mask_save = mask_save_path + "/" + name_image + "_mask.png";
            imwrite(mask_save, mask);
            cout << "Pre-processed image and mask saved!" << endl;
        }

        // If network analysis is required, perform it with the functions of BoatDetector
        if (eval or pred) {
            bd.make_prediction(net);
            bd.prediction_processing();
            bd.apply_prediction_to_input();
            iou_vector.push_back(bd.prediction_evaluation());
        }

        waitKey(0);
        destroyAllWindows();
    }

    // Compute the mean value of IoU metric and print it
    if (eval) {
        float mean_iou = 0;
        for (auto iou : iou_vector)
            mean_iou += iou;
        mean_iou = mean_iou / iou_vector.size();

        cout << "\nMean IoU value for the entire set of images: " << mean_iou << endl;
    }
    return 0;
}
