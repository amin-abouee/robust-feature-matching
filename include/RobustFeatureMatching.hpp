#ifndef Robust_Feature_Macthing_H
#define Robust_Feature_Macthing_H

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

class RobustFeatureMatching
{
   public:
    RobustFeatureMatching(float inputRatioNearestNeighborParam = 0.8,
                          float inputFundamentalReprojectionErrorParam = 1.0,
                          float inputFundamentalConfidenceProbablyParam = 0.99,
                          std::size_t inputThresholdNumPointsParam = 6);

    virtual ~RobustFeatureMatching();
    std::vector<cv::DMatch> run(cv::Mat inputReferenceImage, cv::Mat inputFollowingImage);

    void featureExtraction();
    void bruteForceMatching();
    void nearestNeighbor();
    void symmetricMatching();
    void epipolarConstraint();

    std::vector<cv::DMatch> getBestBruteForceMatchesVector();
    std::vector<cv::DMatch> getBestNearestNeighborMatchesVector();
    std::vector<cv::DMatch> getBestSymmetricMatchesVector();
    std::vector<cv::DMatch> getBestEpipolarConstraintMatchesVector();

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > getInlierPoints();
    std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint> > getKeyPoints();

   private:
    cv::Mat referenceImage;
    cv::Mat followingImage;
    std::vector<cv::KeyPoint> referenceKeypoints;
    std::vector<cv::KeyPoint> followingKeypoints;
    cv::Mat referenceDescriptors;
    cv::Mat followingDescriptors;

    std::vector<std::vector<cv::DMatch> > matches1;
    std::vector<std::vector<cv::DMatch> > matches2;
    std::vector<cv::DMatch> NNMatches1;
    std::vector<cv::DMatch> NNMatches2;
    std::vector<cv::DMatch> bestBruteForceMatchesVector;
    std::vector<cv::DMatch> bestNearestNeighborMatchesVector;
    std::vector<cv::DMatch> bestSymmetricMatchesVector;
    std::vector<cv::DMatch> bestEpipolarConstraintMatchesVector;

    std::vector<cv::Point2f> inlierPoints1;
    std::vector<cv::Point2f> inlierPoints2;

    float ratioNearestNeighborParam;
    float fundamentalReprojectionErrorParam;
    float fundamentalConfidenceProbablyParam;
    std::size_t thresholdNumPointsParam;
};

#endif /* Robust_Feature_Macthing_H */
