#include "detector_tracker_fusion.h"
#include "../utils/math_funcs.h"

#include <opencv2/imgproc.hpp>

#include <iostream>

DetectorTrackerFusion::DetectorTrackerFusion(DetectorBase &detector, TrackerBase &tracker)
: _detector(detector), _tracker(tracker)
{
	tracker_init_once = false;
	fusion_done_once = false;
	prev_centroid.x = -1;
	prev_centroid.y = -1;
}

// do the fusion and output the ROI
bool DetectorTrackerFusion::output_roi(Mat &current_frame, Rect2d &out_roi, bool draw_fusion_on_frame)
{
	// cropped frame to fusion - sometimes the current frame and sometimes a crop of it
	/*Mat cropped_frame;

	// crop frame
	if (fusion_done_once) cropped_frame = current_frame(frame_roi);
	else
	{
		// init frame ROI and don't crop frame
		frame_roi = Rect2d(0, 0, current_frame.cols, current_frame.rows);
		cropped_frame = current_frame;
	}*/

	// get ROI from fusion
	bool success = get_roi_from_fusion(current_frame, current_frame, out_roi, draw_fusion_on_frame);

	// update the current frame ROI
	/*update_frame_roi(current_frame, out_roi, success, frame_roi);

	// draw frame ROI if told to
	if (draw_fusion_on_frame) rectangle(current_frame, frame_roi, Scalar( 0, 0, 255 ), 2, 1 );

	fusion_done_once = true;*/

	return success;
}

// do the fusion and output the ROI
bool DetectorTrackerFusion::get_roi_from_fusion(Mat &current_frame, Mat &cropped_frame, Rect2d &out_roi, bool draw_fusion_on_frame)
{
	// detect objects
	std::vector<Rect2d> object_bboxes;
	std::vector<std::string> object_labels;
	_detector.detect(cropped_frame, object_bboxes, object_labels);

	// offset detection bbox to be relative to current frame rather than cropped frame
	for (int i = 0; i < object_bboxes.size(); i++)
	{
		object_bboxes[i].x += frame_roi.x;
		object_bboxes[i].y += frame_roi.y;
	}

	// find target in detections
	int target_index = find_target_index_in_detections(object_bboxes, object_labels);
	bool found_target = target_index != -1;

	// draw object detections if stated to do so
	if (draw_fusion_on_frame) _detector.draw_bboxes_on_image(current_frame, object_bboxes, object_labels);

	// true if fusion succeeded
	bool success;

	// update tracker (if initialized)
	bool tracker_success = tracker_init_once && _tracker.update(current_frame, out_roi);
	success = success && tracker_success;

	if (found_target)
	{
		// calculate IOU between detector ROI and tracker ROI
		if (tracker_success)
		{
			double bbox_iou = calculate_iou_rect2d(object_bboxes[target_index], out_roi);

			// if the IOU is large enough (boxes overlap) we take the tracker output since it is significantly less noisy
			if (bbox_iou > 0.8)
			{
				Point2d centroid = get_centroid_from_rect2d(out_roi);

				if (draw_fusion_on_frame)
				{
					_tracker.draw_roi_on_frame(current_frame, out_roi, Scalar(255, 0, 0));
					circle(current_frame, centroid, 5, Scalar(255, 0, 0), 3);
				}

				// save previously located centroid
				prev_centroid = centroid;
				return true;
			}
		}

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

		success = true;
	}
	else
	{
		// track according to previous frame and ROI (if there exists one)
		//success = tracker_init_once && _tracker.update(current_frame, out_roi);


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
	}

	return success;
}

// update the frame ROI
void DetectorTrackerFusion::update_frame_roi(Mat &input_frame, Rect2d &fusion_result_roi, bool fusion_success, Rect2d &new_frame_roi)
{
	int min_frame_dimension = input_frame.cols < input_frame.rows ? input_frame.cols : input_frame.rows;
	int max_bbox_dimension = fusion_result_roi.height > fusion_result_roi.width ? fusion_result_roi.height : fusion_result_roi.width;

	Point2d fusion_result_centroid = get_centroid_from_rect2d(fusion_result_roi);

	if (!fusion_success || max_bbox_dimension > (0.9 * min_frame_dimension))
	{
		// if fusion did not succeed or current output ROI is too big to fit in crop look at the whole frame
		new_frame_roi.x = 0;
		new_frame_roi.y = 0;
		new_frame_roi.width = input_frame.cols;
		new_frame_roi.height = input_frame.rows;
	}
	else
	{
		Point2d frame_roi_centroid = get_centroid_from_rect2d(frame_roi);
		if (abs(frame_roi_centroid.x - fusion_result_centroid.x) < 20) return;
		if (abs(frame_roi_centroid.y - fusion_result_centroid.y) < 20) return;

		// look around the frame
		new_frame_roi.width = min_frame_dimension;
		new_frame_roi.height = min_frame_dimension;
		new_frame_roi.x = fusion_result_centroid.x - min_frame_dimension / 2;
		new_frame_roi.y = fusion_result_centroid.y - min_frame_dimension / 2;

		// make sure frame is within image boundaries
		int bottom_right_x = new_frame_roi.x + new_frame_roi.width;
		int bottom_right_y = new_frame_roi.y + new_frame_roi.height;

		if (new_frame_roi.x < 0) new_frame_roi.x = 0;
		if (new_frame_roi.y < 0) new_frame_roi.y = 0;
		if (bottom_right_x >= input_frame.cols) new_frame_roi.x = input_frame.cols - new_frame_roi.width;
		if (bottom_right_y >= input_frame.rows) new_frame_roi.y = input_frame.rows - new_frame_roi.height;
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
