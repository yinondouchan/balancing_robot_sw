#include "pose_tracker.hpp"
#include <iostream>

PoseTracker::PoseTracker()
{
	prev_object_count = 0;
}

void PoseTracker::track_pose(const Mat &frame, const int object_count, const Mat &objects, const std::vector<std::vector<Point2f>> &normalized_peaks)
{
	// match new keypoints with old keypoints

	// check if any of the keypoints have disappeared

	// for each keypoint that disappeared find new location by tracking

	// update objects' missing keypoints

	// store previous result
	prev_frame = frame;
	prev_object_count = object_count;
	prev_objects = objects;
}

void PoseTracker::get_keypoint_bboxes(std::vector<Rect2f> &bboxes, const Mat &frame, const int object_count, const Mat &objects, const std::vector<std::vector<Point2f>> &normalized_peaks)
{
	for (int obj_ind = 0; obj_ind < object_count; obj_ind++)
	{
		for (int part_type_ind = 0; part_type_ind < objects.cols; part_type_ind++)
		{
			int part_peak_ind = objects.at<int>(obj_ind, part_type_ind);
			if (part_peak_ind >= 0)
			{
				float peak_x = normalized_peaks[part_type_ind][part_peak_ind].x * frame.cols;
				float peak_y = normalized_peaks[part_type_ind][part_peak_ind].y * frame.rows;

				bboxes.push_back(Rect2f(peak_x - 10, peak_y - 10, 30, 30));
			}
		}
	}
}

void PoseTracker::draw_keypoint_bboxes_on_frame(const Mat &frame, const std::vector<Rect2f> &bboxes)
{
	for (Rect2f bbox : bboxes)
	{
		rectangle(frame, bbox, Scalar(255, 0, 0), 2);
	}
}
