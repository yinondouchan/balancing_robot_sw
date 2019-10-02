#include "detector_tracker_fusion.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

DetectorTrackerFusion::DetectorTrackerFusion(DetectorBase &detector, TrackerBase &tracker)
: _detector(detector), _tracker(tracker)
{
	tracker_init_once = false;
	prev_centroid.x = -1;
	prev_centroid.y = -1;
}

// do the fusion and output the ROI
bool DetectorTrackerFusion::output_roi(Mat &current_frame, Rect2d &out_roi, bool draw_fusion_on_frame)
{
	// detect objects
	std::vector<Rect2d> object_bboxes;
	std::vector<std::string> object_labels;
	_detector.detect(current_frame, object_bboxes, object_labels);

	// find target in detections
	int target_index = find_target_index_in_detections(object_bboxes, object_labels);
	bool found_target = target_index != -1;

	// draw object detections if stated to do so
	if (draw_fusion_on_frame) _detector.draw_bboxes_on_image(current_frame, object_bboxes, object_labels);

	if (found_target)
	{
		// initialize the tracker to this ROI
		_tracker.init(current_frame, object_bboxes[target_index]);
		tracker_init_once = true;

		// output this ROI
		out_roi = object_bboxes[target_index];

		Point2d centroid = get_centroid_from_rect2d(out_roi);

		// draw a small circle in the target's centroid if stated to do so
		if (draw_fusion_on_frame) circle(current_frame, centroid, 5, Scalar(0, 255, 0), 3);

		// save previously located centroid
		prev_centroid = centroid;

		return true;
	}
	else
	{
		// track according to previous frame and ROI (if there exists one)
		bool success = tracker_init_once && _tracker.update(current_frame, out_roi);


		// draw tracker ROI and centroid if stated to do so
		if (success) {
			Point2d centroid = get_centroid_from_rect2d(out_roi);

			if (draw_fusion_on_frame)
			{
				_tracker.draw_roi_on_frame(current_frame, out_roi, Scalar(255, 0, 0));
				circle(current_frame, centroid, 5, Scalar(255, 0, 0), 3);
			}

			// save previously located centroid
			prev_centroid = centroid;
		}

		return success;
	}
}

// given detections, find the target index in them. Return -1 if target was not found
int DetectorTrackerFusion::find_target_index_in_detections(std::vector<Rect2d> &bboxes, std::vector<std::string> &labels)
{
	// nothing to do if detector did not find any objects
	if (bboxes.size() == 0) return -1;

	int person_index = -1;

	bool locked_on_target = (prev_centroid.x != -1) && (prev_centroid.y != -1);
	double min_distance = DBL_MAX;
	double max_bbox_area = 0;

	// if did not lock on target before: find the person with the highest bounding box area
	// if locked on target before: find the person with the closest centroid to the previously locked target
	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i].compare("person") == 0)
		{
			// detected a person
			if (locked_on_target)
			{
				// find detection with closest centroid
				Point2d new_target_centroid = get_centroid_from_rect2d(bboxes[i]);

				double x_diff = new_target_centroid.x - prev_centroid.x;
				double y_diff = new_target_centroid.y - prev_centroid.y;
				double distance_from_prev_centroid = sqrt(x_diff*x_diff + y_diff*y_diff);
				if (min_distance > distance_from_prev_centroid)
				{
					min_distance = distance_from_prev_centroid;
					person_index = i;
				}
			}
			else
			{
				// find detection with largest bounding box area
				double bbox_area = bboxes[i].width * bboxes[i].height;
				if (max_bbox_area < bbox_area)
				{
					max_bbox_area = bbox_area;
					person_index = i;
				}
			}
		}
	}

	return person_index;
}

Point2d DetectorTrackerFusion::get_centroid_from_rect2d(Rect2d &rect)
{
	return Point2d(rect.x + rect.width/2, rect.y + rect.height/2);
}
