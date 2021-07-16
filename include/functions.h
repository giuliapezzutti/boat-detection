#ifndef BOAT_DETECTION_FUNCTIONS_H
#define BOAT_DETECTION_FUNCTIONS_H

#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<string> split (const string &s, char delim);
string extract_name_from_path (const string& path);
string extract_folder_from_path (const string& path);
void init(int argc, char *argv[], String &images_path, bool &train, bool &eval, bool &pred, String &masks_path);

#endif
