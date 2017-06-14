#pragma once

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <tarsier/tarsier_common.h>

namespace ORB_SLAM2
{

class TarsierExtractor {
protected:
	Tarsier dev;
public:
	TarsierExtractor(const char * dev_node);

	// Compute the ORB features and descriptors on an image using a Tarsier device.
    // Mask is ignored in the current implementation.
	void operator()(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors);
};

}