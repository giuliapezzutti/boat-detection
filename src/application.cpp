#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/dnn/dnn.hpp>
#include "../include/functions.h"
#include "../include/BoatDetector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    String input_imgs, folder_images, folder_masks;
    bool train, eval, pred;
    init(argc, argv, input_imgs, train, eval, pred, folder_masks);

    vector<String> img_paths;
    if (input_imgs.substr(input_imgs.size() - 4) != ".png") {
        folder_images = input_imgs;
        glob(folder_images + "/*.png", img_paths);
    }
    else {
        img_paths.push_back(input_imgs);
        folder_images = extract_folder_from_path(input_imgs);
    }

    if (img_paths.empty()){
        cerr << "The folder must contain at least one image!";
        exit(1);
    }

    String save_path = folder_images + "/training_images";
    if (train){
        if (opendir(save_path.c_str()) == nullptr)
            mkdir(save_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    String mask_save_path = folder_masks + "/image_masks";
    if (train or eval){
        if (opendir(mask_save_path.c_str()) == nullptr)
            mkdir(mask_save_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    dnn::Net net;
    if (eval or pred){
        String model_path = "model/vggboat.pb";
        net = dnn::readNetFromTensorflow(model_path);
    }

    for (const auto& path : img_paths){

        string name_image = extract_name_from_path(path);
        Mat img = imread(path);

        BoatDetector bd = BoatDetector(img, name_image, Size(100, 100));
        Mat processed_img = bd.image_preprocessing();

        if (train or eval) {

            String path_label = folder_masks + "/" + name_image + ".txt";
            Mat mask = bd.mask_preprocessing(path_label);

            if (train) {
                String img_save = save_path + "/" + name_image + "_training.png";
                imwrite(img_save, processed_img);
                String mask_save = mask_save_path + "/" + name_image + "_mask.png";
                imwrite(mask_save, mask);
            }
        }

        bd.make_prediction(net);

    }
}
