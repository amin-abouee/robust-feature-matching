#include <iostream>
#include "RobustFeatureMatching.hpp"

RobustFeatureMatching::RobustFeatureMatching(float inputRatioNearestNeighborParam,
                                             float inputFundamentalReprojectionErrorParam,
                                             float inputFundamentalConfidenceProbablyParam,
                                             std::size_t inputThresholdNumPointsParam)
    : ratioNearestNeighborParam(inputRatioNearestNeighborParam),
      fundamentalReprojectionErrorParam(inputFundamentalReprojectionErrorParam),
      fundamentalConfidenceProbablyParam(inputFundamentalConfidenceProbablyParam),
      thresholdNumPointsParam(inputThresholdNumPointsParam)
{
}

RobustFeatureMatching::~RobustFeatureMatching()
{
}
std::vector<cv::DMatch> RobustFeatureMatching::run(cv::Mat inputReferenceImage, cv::Mat inputFollowingImage)
{
    referenceImage = inputReferenceImage;
    followingImage = inputFollowingImage;
    featureExtraction();
    bruteForceMatching();
    nearestNeighbor();
    symmetricMatching();
    epipolarConstraint();

    if (bestEpipolarConstraintMatchesVector.size() >= thresholdNumPointsParam)
        return bestEpipolarConstraintMatchesVector;
    else
    {
        if (bestSymmetricMatchesVector.size() >= thresholdNumPointsParam)
            return bestSymmetricMatchesVector;
        else
        {
            if (bestNearestNeighborMatchesVector.size() >= thresholdNumPointsParam)
                return bestNearestNeighborMatchesVector;
            else
                return bestBruteForceMatchesVector;
        }
    }
}

void RobustFeatureMatching::featureExtraction()
{
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(referenceImage, cv::noArray(), referenceKeypoints, referenceDescriptors);
    std::cout << referenceKeypoints.size() << std::endl;
    akaze->detectAndCompute(followingImage, cv::noArray(), followingKeypoints, followingDescriptors);
    std::cout << followingKeypoints.size() << std::endl;
}

void RobustFeatureMatching::bruteForceMatching()
{
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.knnMatch(referenceDescriptors, followingDescriptors, matches1, 2);
    matcher.knnMatch(followingDescriptors, referenceDescriptors, matches2, 2);
    for (size_t i(0); i < matches1.size(); i++)
    {
        bestBruteForceMatchesVector.push_back(matches1[i][0]);
    }
}

void RobustFeatureMatching::nearestNeighbor()
{
    for (size_t i(0); i < matches1.size(); i++)
    {
        float dist1 = matches1[i][0].distance;
        float dist2 = matches1[i][1].distance;

        if (dist1 < ratioNearestNeighborParam * dist2)
        {
            NNMatches1.push_back(matches1[i][0]);
        }
    }

    for (size_t i(0); i < matches2.size(); i++)
    {
        float dist1 = matches2[i][0].distance;
        float dist2 = matches2[i][1].distance;

        if (dist1 < ratioNearestNeighborParam * dist2)
        {
            NNMatches2.push_back(matches2[i][0]);
        }
    }
    bestNearestNeighborMatchesVector = NNMatches1;
}

void RobustFeatureMatching::symmetricMatching()
{
    for (size_t i(0); i < NNMatches1.size(); i++)
    {
        for (size_t j(0); j < NNMatches2.size(); j++)
        {
            if (NNMatches1[i].queryIdx == NNMatches2[j].trainIdx && NNMatches1[i].trainIdx == NNMatches2[j].queryIdx)
            {
                bestSymmetricMatchesVector.push_back(
                    cv::DMatch(NNMatches1[i].queryIdx, NNMatches1[i].trainIdx, NNMatches1[i].distance));
                break;
            }
        }
    }
}

void RobustFeatureMatching::epipolarConstraint()
{
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for (size_t i(0); i < bestSymmetricMatchesVector.size(); i++)
    {
        int queryInx = bestSymmetricMatchesVector[i].queryIdx;
        int trainInx = bestSymmetricMatchesVector[i].trainIdx;
        points1.push_back(cv::Point2f(referenceKeypoints[queryInx].pt.x, referenceKeypoints[queryInx].pt.y));
        points2.push_back(cv::Point2f(followingKeypoints[trainInx].pt.x, followingKeypoints[trainInx].pt.y));
    }
    std::vector<uchar> inliers(points1.size(), 0);
    cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2),  // matching points
                                                 cv::FM_RANSAC,                       // RANSAC method
                                                 fundamentalReprojectionErrorParam,   // distance to epipolar line
                                                 fundamentalConfidenceProbablyParam,  // confidence probability
                                                 inliers);  // match status (inlier or outlier)

    int inxMatcher = 0;
    for (std::vector<uchar>::const_iterator inlierIterator = inliers.begin(); inlierIterator != inliers.end();
         inlierIterator++)
    {
        if (*inlierIterator)
        {
            int first = bestSymmetricMatchesVector[inxMatcher].queryIdx;
            int second = bestSymmetricMatchesVector[inxMatcher].trainIdx;
            bestEpipolarConstraintMatchesVector.push_back(cv::DMatch(first, second, 0));
            inlierPoints1.push_back(points1[inxMatcher]);
            inlierPoints2.push_back(points2[inxMatcher]);
        }
        inxMatcher++;
    }
}

std::vector<cv::DMatch> RobustFeatureMatching::getBestBruteForceMatchesVector()
{
    return bestBruteForceMatchesVector;
}
std::vector<cv::DMatch> RobustFeatureMatching::getBestNearestNeighborMatchesVector()
{
    return bestNearestNeighborMatchesVector;
}

std::vector<cv::DMatch> RobustFeatureMatching::getBestSymmetricMatchesVector()
{
    return bestSymmetricMatchesVector;
}
std::vector<cv::DMatch> RobustFeatureMatching::getBestEpipolarConstraintMatchesVector()
{
    return bestEpipolarConstraintMatchesVector;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f> > RobustFeatureMatching::getInlierPoints()
{
    return make_pair(inlierPoints1, inlierPoints2);
}

std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint> > RobustFeatureMatching::getKeyPoints()
{
    return make_pair(referenceKeypoints, followingKeypoints);
}
