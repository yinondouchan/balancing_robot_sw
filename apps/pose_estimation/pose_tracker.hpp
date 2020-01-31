/*
 * pose_tracker.hpp
 *
 *  Created on: Dec 17, 2019
 *      Author: yinon
 */

#ifndef POSE_TRACKER_HPP_
#define POSE_TRACKER_HPP_

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;


class PoseTracker
{
public:
	PoseTracker();

	// track pose given previous frame and pose data
	void track_pose(const Mat &frame, const int object_count, const Mat &objects, const std::vector<std::vector<Point2f>> &normalized_peaks);

	// draw extracted keypoint bounding boxes on frame
	void draw_keypoint_bboxes_on_frame(const Mat &frame, const std::vector<Rect2f> &bboxes);

	// find bounding boxes around keypoints
	void get_keypoint_bboxes(std::vector<Rect2f> &bboxes, const Mat &frame, const int object_count, const Mat &objects, const std::vector<std::vector<Point2f>> &normalized_peaks);
private:
	int prev_object_count;
	Mat prev_frame;
	Mat prev_objects;
	std::vector<std::vector<Point2f>> prev_peaks;
};


#endif /* POSE_TRACKER_HPP_ */
