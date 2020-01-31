/*
 * configs.h
 *
 *  Created on: Dec 15, 2019
 *      Author: yinon
 */

#ifndef CONFIGS_H_
#define CONFIGS_H_

#include <opencv2/highgui.hpp>

using namespace cv;


class PoseEstimationConfig
{
public:
	// number of part types
	int num_part_types;

	// number of linkage types
	int num_link_types;

	// network input dimensions
	Size input_size;

	// network input number of channels
	int input_nchannels;

	// network output confidence map and part affinity fields output size for each part/link
	Size output_map_size;

	// maximum parts/links to be detected in an for each part type
	int max_parts;

	// maximum objects to be detected in an image
	int max_objects;

	// confidence threshold for checking an element in confidence map to be a peak
	float peak_confidence_threshold;

	// window size to check if a candidate confidence is a peak
	int peak_window_size;

	// PAF vector score for determining that two parts should be linked
	float link_threshold;

	// number of samples between source and destination part to check for part affinity score
	int line_integral_samples;

	// path for the TensorRT engine
	std::string engine_path;

	// topology
	std::vector<std::array<int, 4>> topology;

	PoseEstimationConfig();

	PoseEstimationConfig(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
						 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7);
};

/*
 * Config for the Resnet 18 224x224 model
 */
class Resnet18Size224Config : public PoseEstimationConfig
{
public:
	Resnet18Size224Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7);
private:
	// part inds to meaning
	const std::vector<std::string> keypoints{"nose", "left_eye", "right_eye", "left_ear", "right_ear",
		"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
		"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"};

	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(16, 14),
				std::make_pair(14, 12),
				std::make_pair(17, 15),
				std::make_pair(15, 13),
				std::make_pair(12, 13),
				std::make_pair(6, 8),
				std::make_pair(7, 9),
				std::make_pair(8, 10),
				std::make_pair(9, 11),
				std::make_pair(2, 3),
				std::make_pair(1, 2),
				std::make_pair(1, 3),
				std::make_pair(2, 4),
				std::make_pair(3, 5),
				std::make_pair(4, 6),
				std::make_pair(5, 7),
				std::make_pair(18, 1),
				std::make_pair(18, 6),
				std::make_pair(18, 7),
				std::make_pair(18, 12),
				std::make_pair(18, 13)
	};
};

/*
 * Config for the Resnet 18 224x224 model
 */
class Densenet121Size256Config : public PoseEstimationConfig
{
public:
	Densenet121Size256Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7);
private:
	// part inds to meaning
	const std::vector<std::string> keypoints{"nose", "left_eye", "right_eye", "left_ear", "right_ear",
		"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
		"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"};

	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(16, 14),
				std::make_pair(14, 12),
				std::make_pair(17, 15),
				std::make_pair(15, 13),
				std::make_pair(12, 13),
				std::make_pair(6, 8),
				std::make_pair(7, 9),
				std::make_pair(8, 10),
				std::make_pair(9, 11),
				std::make_pair(2, 3),
				std::make_pair(1, 2),
				std::make_pair(1, 3),
				std::make_pair(2, 4),
				std::make_pair(3, 5),
				std::make_pair(4, 6),
				std::make_pair(5, 7),
				std::make_pair(18, 1),
				std::make_pair(18, 6),
				std::make_pair(18, 7),
				std::make_pair(18, 12),
				std::make_pair(18, 13)
	};
};

/*
 * Config for my fabulous Resnet 18 224x224 based foot pose model
 */
class FootPose224Config : public PoseEstimationConfig
{
public:
	FootPose224Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7);
private:
	// part inds to meaning
	const std::vector<std::string> keypoints{"left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
		"right_small_toe", "right_heel"};

	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(1, 2),
				std::make_pair(2, 3),
				std::make_pair(3, 1),
				std::make_pair(4, 5),
				std::make_pair(5, 6),
				std::make_pair(6, 4)
	};
};

/*
 * Config for my fabulous Resnet 18 224x224 based body and foot pose model
 */
class BodyAndFeetPose224Config : public PoseEstimationConfig
{
public:
	BodyAndFeetPose224Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7, int output_width = 56, int output_height = 56);
private:
	// part inds to meaning
//	const std::vector<std::string> keypoints{"left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
//		"right_small_toe", "right_heel"};

	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(15, 13),
				std::make_pair(13, 11),
				std::make_pair(16, 14),
				std::make_pair(14, 12),
				std::make_pair(11, 12),
				std::make_pair(5, 11),
				std::make_pair(6, 12),
				std::make_pair(5, 6),
				std::make_pair(5, 7),
				std::make_pair(6, 8),
				std::make_pair(7, 9),
				std::make_pair(8, 10),
				std::make_pair(1, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(1, 3),
				std::make_pair(2, 4),
				std::make_pair(3, 5),
				std::make_pair(4, 6),
				std::make_pair(17, 15),
				std::make_pair(18, 15),
				std::make_pair(19, 15),
				std::make_pair(20, 16),
				std::make_pair(21, 16),
				std::make_pair(22, 16)
	};
};

/*
 * Config for my fabulous custom-trained Resnet 18 224x224 based body only pose model
 */
class BodyOnlyPose224Config : public PoseEstimationConfig
{
public:
	BodyOnlyPose224Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7, int output_width = 56, int output_height = 56);
private:
	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(15, 13),
				std::make_pair(13, 11),
				std::make_pair(16, 14),
				std::make_pair(14, 12),
				std::make_pair(11, 12),
				std::make_pair(5, 11),
				std::make_pair(6, 12),
				std::make_pair(5, 6),
				std::make_pair(5, 7),
				std::make_pair(6, 8),
				std::make_pair(7, 9),
				std::make_pair(8, 10),
				std::make_pair(1, 2),
				std::make_pair(0, 1),
				std::make_pair(0, 2),
				std::make_pair(1, 3),
				std::make_pair(2, 4),
				std::make_pair(3, 5),
				std::make_pair(4, 6)
	};
};


/*
 * Config for my fabulous Resnet 18 224x224 based hand pose model
 */
class HandPose224Config : public PoseEstimationConfig
{
public:
	HandPose224Config(std::string engine_path, int max_parts=100, int max_objects=100, float peak_confidence_threshold=0.1,
			 float link_threshold=0.1, int peak_window_size=5, int line_integral_samples=7);
private:
	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(0, 1),
				std::make_pair(1, 2),
				std::make_pair(2, 3),
				std::make_pair(3, 4),
				std::make_pair(0, 5),
				std::make_pair(5, 6),
				std::make_pair(6, 7),
				std::make_pair(7, 8),
				std::make_pair(0, 9),
				std::make_pair(9, 10),
				std::make_pair(10, 11),
				std::make_pair(11, 12),
				std::make_pair(0, 13),
				std::make_pair(13, 14),
				std::make_pair(14, 15),
				std::make_pair(15, 16),
				std::make_pair(0, 17),
				std::make_pair(17, 18),
				std::make_pair(18, 19),
				std::make_pair(19, 20)
	};
};

#endif /* CONFIGS_H_ */
