#ifndef BOAT_DETECTION_FUNCTIONS_H
#define BOAT_DETECTION_FUNCTIONS_H

#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void init(int argc, char *argv[], String &images_path, bool &train, bool &eval, bool &pred, String &masks_path);
vector<string> split_line (const string &s, char delim);
string extract_name_from_path (const string& path);
string extract_folder_from_path (const string& path);
void extract_squared_padding(Size current_dim, int desired_dim, int& top, int& bottom , int& left, int& right);

#endif
