#include <opencv2/core/core.hpp>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include "RobustFeatureMatching.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "number of input parameters are not sufficient." << std::endl;
        return 1;
    }
    cv::Mat first_image = cv::imread(argv[1], 0);
    cv::Mat second_image = cv::imread(argv[2], 0);
    RobustFeatureMatching matcher;
    std::vector<cv::DMatch> result = matcher.run(first_image, second_image);
    std::cout << "Result: " << result.size() << std::endl;
}
