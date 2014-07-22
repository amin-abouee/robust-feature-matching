#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    // load both images
    cv::Mat srcImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat descImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    // define keypoints for both images
    std::vector <cv::KeyPoint> srcKeypoints;
    std::vector <cv::KeyPoint> descKeypoints;

    // pointer to the feature point detector object
    cv::Ptr <cv::FeatureDetector> detector = cv::FeatureDetector::create("AKAZE");
    // cv::Ptr<cv::FeatureDetector> detector = new cv::MSER();
    detector->detect(srcImage, srcKeypoints);
    detector->detect(descImage, descKeypoints);

    std::cout << "detector: " << srcKeypoints.size() << " , " << descKeypoints.size() << std::endl;

    cv::Mat srcDescriptors;
    cv::Mat descDescriptors;
    // pointer to the feature descriptor extractor object
    cv::Ptr <cv::DescriptorExtractor> extractor = cv::DescriptorExtractor::create("AKAZE");
    // cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor();
    extractor->compute(srcImage, srcKeypoints, srcDescriptors);
    extractor->compute(descImage, descKeypoints, descDescriptors);
    std::cout << "extractor: " << srcDescriptors.rows << " , " << descDescriptors.rows << std::endl;

    cv::Ptr <cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming(2)");
    // cv::BFMatcher matcher(cv::NORM_HAMMING2);
    std::vector< cv::DMatch > matches;
    matcher->match( srcDescriptors, descDescriptors, matches );
    // Ubitrack::Vision::RobustFetureMatchingBitVecFeatureBase matcher(srcFeaturepoints, descFeaturePoints, 0.65, 0.99, 1.0);
    // matcher.run();
    // matches = matcher.getMatches();

    // std::vector<cv::DMatch> cvMatches;

    // std::cout << "final match points: " << matches.size() << std::endl;

    // for (std::size_t i(0); i < matches.size(); i++)
    // {
    // cvMatches.push_back(cv::DMatch(matches[i].first, matches[i].second, 0));
    // }

    cv::Mat matchImage;
    cv::drawMatches(srcImage, srcKeypoints, descImage, descKeypoints, matches, matchImage);

    cv::namedWindow("Robust Matcher", 1);
    cv::imshow("Robust Matcher", matchImage);

    cv::waitKey(0);
    return 0;
}