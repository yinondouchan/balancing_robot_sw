#include "detector_tracker_fusion.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

DetectorTrackerFusion::DetectorTrackerFusion(DetectorBase &detector, TrackerBase &tracker)
: _detector(detector), _tracker(tracker)
{
	tracker_init_once = false;
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

		// draw a small circle in the target's centroid if stated to do so
		if (draw_fusion_on_frame) circle(current_frame, get_centroid_from_rect2d(out_roi), 5, Scalar(0, 255, 0), 3);

		return true;
	}
	else
	{
		// track according to previous frame and ROI (if there exists one)
		bool success = tracker_init_once && _tracker.update(current_frame, out_roi);

		// draw tracker ROI and centroid if stated to do so
		if (draw_fusion_on_frame && success) {
			_tracker.draw_roi_on_frame(current_frame, out_roi, Scalar(255, 0, 0));
			circle(current_frame, get_centroid_from_rect2d(out_roi), 5, Scalar(255, 0, 0), 3);
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
	// for now assume one person in frame and get the first index from it
	for (int i = 0; i < labels.size(); i++)
	{
		if (labels[i].compare("person") == 0)
		{
			person_index = i;
		}
	}

	return person_index;
}

Point2d DetectorTrackerFusion::get_centroid_from_rect2d(Rect2d &rect)
{
	return Point2d(rect.x + rect.width/2, rect.y + rect.height/2);
}
