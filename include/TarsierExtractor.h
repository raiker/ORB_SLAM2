#pragma once

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <tarsier_common.h>
#include <soft_tarsier.h>

#include "OrbExtractorBase.h"

namespace ORB_SLAM2
{

class TarsierExtractorBase : public OrbExtractorBase {
public:
	virtual int GetLevels() {
		return 9;
	}

	virtual float GetScaleFactor() {
		return 1.25f;
	}

	virtual std::vector<float> GetScaleFactors() {
		return std::vector<float> {1.0f, 1.25f, 1.5625f, 2.0f, 2.5f, 3.125f, 4.0f, 5.0f, 6.25f};
	}

	virtual std::vector<float> GetInverseScaleFactors() {
		return std::vector<float> {1.0f, 0.8f, 0.64f, 0.5f, 0.4f, 0.32f, 0.25f, 0.2f, 0.16f};
	}

	virtual std::vector<float> GetScaleSigmaSquares() {
		return std::vector<float> {1.0f, 1.5625f, 2.44140625f, 4.0f, 6.25f, 9.765625f, 16.0f, 25.0f, 39.0625f};
	}

	virtual std::vector<float> GetInverseScaleSigmaSquares() {
		return std::vector<float> {1.0f, 0.64f, 0.4096f, 0.25f, 0.16f, 0.1024f, 0.0625f, 0.04f, 0.0256f};
	}
};

class HardTarsierExtractor : public TarsierExtractorBase {
protected:
	Tarsier dev;
public:
	HardTarsierExtractor(const char * dev_node);

	// Compute the ORB features and descriptors on an image using a Tarsier device.
    // Mask is ignored in the current implementation.
	virtual void ProcessImage(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors);
};

class SoftTarsierExtractor : public TarsierExtractorBase {
public:
	SoftTarsierExtractor() {}

	// Compute the ORB features and descriptors on an image using a software emulated Tarsier
    // Mask is ignored in the current implementation.
	virtual void ProcessImage(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors);
};

}