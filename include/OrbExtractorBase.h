#pragma once

#include <vector>


namespace ORB_SLAM2
{

class OrbExtractorBase {
public:
	virtual void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) = 0;

	virtual int GetLevels() = 0;

	virtual float GetScaleFactor() = 0;

	virtual std::vector<float> GetScaleFactors() = 0;

	virtual std::vector<float> GetInverseScaleFactors() = 0;

	virtual std::vector<float> GetScaleSigmaSquares() = 0;

	virtual std::vector<float> GetInverseScaleSigmaSquares() = 0;
};
}