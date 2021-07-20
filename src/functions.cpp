#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstddef>

using namespace std;
using namespace cv;

void init(int argc, char *argv[], String &images_path, bool &train, bool &eval, bool &pred, String &masks_path) {

    const String keys =
            "{help h usage ? |<none>| Print help message}"
            "{training t     |      | Generate a set of pre-processed images for the neural network training}"
            "{evaluation e   |      | Evaluate the neural network with respect to one or more images with known mask}"
            "{predict p      |      | Predict one or more images thanks to the neural network}"
            "{@images        |      | Input images path}"
            "{masks folder   |      | Input masks path}";

    CommandLineParser parser(argc, argv, keys);
    parser.about("\nBOAT DETECTION\n");
    if (parser.has("help")) {
        parser.printMessage();
        exit(1);
    }

    if ((parser.has("training") and parser.has("evaluation"))
        or (parser.has("training") and parser.has("predict"))
        or (parser.has("evaluation") and parser.has("predict"))){
        cerr << "Only one task can be performed at time!";
        exit(1);
    }
   
    if (!parser.has("training") and !parser.has("evaluation") and !parser.has("predict")){
        cerr << "You must provide one task to be performed!";
        exit(1);
    }

    images_path = parser.get<String>("@images");
    if (images_path.empty()){
        cerr << "You must provide the path to the images folder!";
        exit(1);
    }

    parser.has("training") ? train = true : train = false;
    parser.has("evaluation") ? eval = true : eval = false;
    parser.has("predict") ? pred = true : pred = false;

    if (parser.has("training") or (parser.has("evaluation"))) {
        if (parser.has("masks")) {
            masks_path = parser.get<String>("masks");
        }
        else{
            cerr << "You must provide the masks folder path for the current task!";
            exit(-1);
        }
    }

    if (!parser.check()) {
        parser.printErrors();
        exit(-1);
    }

}

vector<string> split_line (const string &s, char delim){
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

string extract_name_from_path (const string& path){
    vector<string> split_results = split_line(path, '.');
    string without_extension = split_results[0];
    split_results = split_line(without_extension, '/');
    string name_image = split_results[split_results.size()-1];
    split_results = split_line(name_image, '_');
    return split_results[0];
}

string extract_folder_from_path (const string& path){
    vector<string> split_results = split_line(path, '.');
    string without_extension = split_results[0];
    size_t last = without_extension.find_last_of('/');
    string folder = string(&without_extension[0], &without_extension[last+1]);
    return folder;
}


void extract_squared_padding(Size current_dim, int desired_dim, int& top, int& bottom , int& left, int& right){
    top = 0;
    bottom = 0;
    left = 0;
    right = 0;

    if (current_dim.height < desired_dim){
        bottom = ceil((double)(desired_dim-current_dim.height)/2);
        top = floor((double)(desired_dim-current_dim.height)/2);
    }
    else if (current_dim.width < desired_dim) {
        right = ceil((double)(desired_dim-current_dim.width)/2);
        left = floor((double)(desired_dim-current_dim.width)/2);
    }
}

