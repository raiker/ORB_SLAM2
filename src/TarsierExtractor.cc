#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "TarsierExtractor.h"

namespace ORB_SLAM2 {
	static std::vector<uint8_t> vec_from_mat(cv::Mat &image_matrix) {
		assert(image_matrix.type() == CV_8UC1);

		cv::Size size = image_matrix.size();
		if (image_matrix.isContinuous()){
			size.width *= size.height;
			size.height = 1;
		}

		std::vector<uint8_t> pixels;

		for (int y = 0; y < size.height; y++){
			uint8_t * src_ptr = image_matrix.ptr(y);

			pixels.insert(pixels.end(), src_ptr, src_ptr + size.width);
		}

		return pixels;
	}

	HardTarsierExtractor::HardTarsierExtractor(const char * dev_node) :
		dev(dev_node)
	{}

    // Mask is ignored
	void HardTarsierExtractor::ProcessImage(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors)
	{
		//transmute the pixels into an ImgData struct
		if(image.empty()){
			return;
		}

		cv::Mat image_matrix = image.getMat();
		
		std::vector<uint8_t> pixels = vec_from_mat(image_matrix);

		assert(pixels.size() == image_matrix.size().width * image_matrix.size().height);

		ImgData frame(image_matrix.size().width, image_matrix.size().height, std::move(pixels));

		std::vector<feature_descriptor> output_features = dev.get_image_features(frame);

		keypoints.clear();
		descriptors.create(output_features.size(), 32, CV_8U);

		for (uint32_t i = 0; i < output_features.size(); i++) {
			auto feature = output_features[i];

			keypoints.emplace_back(
				feature.x,
				feature.y,
				18, //size
				-1, //angle
				0, //response
				feature.level,
				-1 //class id
			);

			//copy descriptor bytes
			std::copy(feature.orb_descriptor_chunks, feature.orb_descriptor_chunks + 32, descriptors.getMat().ptr(i));
		}
	}

	void SoftTarsierExtractor::ProcessImage(cv::InputArray image, cv::InputArray mask,
		std::vector<cv::KeyPoint>& keypoints,
		cv::OutputArray descriptors)
	{
		if(image.empty()){
			return;
		}

		cv::Mat image_matrix = image.getMat();
		std::vector<uint8_t> pixels = vec_from_mat(image_matrix);
		assert(pixels.size() == image_matrix.size().width * image_matrix.size().height);

		soft_tarsier::ResultSet * results = soft_tarsier::get_features(&pixels[0], image_matrix.size().width, image_matrix.size().height);

		uint32_t num_features = soft_tarsier::get_num_features(results);
		//std::cout << num_features << std::endl;
		
		keypoints.clear();
		keypoints.resize(num_features);
		descriptors.create(num_features, 32, CV_8U);

		assert(sizeof(cv::KeyPoint) == 28);
		assert(sizeof(soft_tarsier::ORBDescriptor) == 32);

		soft_tarsier::fill_arrays_and_drop(results, &keypoints[0], reinterpret_cast<soft_tarsier::ORBDescriptor*>(descriptors.getMat().ptr(0)));

		/*for (int i = 0; i < 24; i++){
			auto &kp = keypoints[i];

			std::cout << kp.pt.x << " " << kp.pt.y << std::endl;
		}*/
	}
}