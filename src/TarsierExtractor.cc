#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "TarsierExtractor.h"

namespace ORB_SLAM2 {
	TarsierExtractor::TarsierExtractor(const char * dev_node) :
		dev(dev_node)
	{}

    // Mask is ignored
	void operator()(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors)
	{
		//transmute the pixels into an ImgData struct
		if(image.empty()){
			return;
		}

		Mat image_matrix = image.getMat();
		assert(image.type() == CV_8UC1);

		cv::Size size = image.size();
		if (image.isContinuous()){
			size.width *= size.height();
			size.height = 1;
		}

		vector<uint8_t> pixels;

		for (int y = 0; y < size.height; y++){
			uint8_t * src_ptr = image.ptr(y);

			pixels.insert(pixels.end(), src_ptr, src_ptr + size.width);
		}

		assert(pixels.size() == image.size().width * image.size().height);

		ImgData frame(image.size().width, image.size().height, std::move(pixels));

		std::vector<feature_descriptor> output_features = dev.get_image_features(frame);

		keypoints.clear();
		descriptors = Mat::zeros((int)output_features.size(), 32, CV_8UC1);

		for (int i = 0; i < output_features.size(); i++) {
			auto feature = output_features[i];

			keypoints.emplace_back(
				feature.x,
				feature.y,
				18, //size
				-1, //angle
				0 //response
				feature.level,
				-1 //class id
			);

			//copy descriptor bytes
			std::copy(feature.orb_descriptor_chunks, feature.orb_descriptor_chunks + 32, descriptors.ptr(i));
		}
	}
}