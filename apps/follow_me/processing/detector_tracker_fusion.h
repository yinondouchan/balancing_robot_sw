/*
 * detector_tracker_fusion.h
 *
 *  Created on: Sep 29, 2019
 *      Author: yinon
 */

#ifndef DETECTOR_TRACKER_FUSION_H_
#define DETECTOR_TRACKER_FUSION_H_

#include "../detectors/detector_base.h"
#include "../trackers/tracker_base.h"
#include "../utils/filters.h"

#include <string>

/*
 * This class fuses between a detector and a tracker to create a more robust ROI of an object.
 * It either outputs the detector or the tracker based on the following logic:
 * - If the desired object is detected, output the detector.
 * - If the desired object is not detected and tracking was successful output the tracker.
 * - If both were unsuccessful there's nothing we can do.
 */
class DetectorTrackerFusion
{
public:
	DetectorTrackerFusion(DetectorBase &detector, TrackerBase &tracker);

	// do the fusion and output the ROI
	bool output_roi(Mat &current_frame, Rect2d &out_roi, bool draw_fusion_on_frame=false);
private:
	// find the centroid of a Rect2d object
	Point2d get_centroid_from_rect2d(Rect2d &rect);

	// given detections, find the target index in them. Return -1 if target was not found
	int find_target_index_in_detections(std::vector<Rect2d> &bboxes, std::vector<std::string> &labels);

	// update the frame ROI
	void update_frame_roi(Mat &input_frame, Rect2d &fusion_result_roi, bool fusion_success, Rect2d &new_frame_roi);

	// do the fusion and output the ROI
	bool get_roi_from_fusion(Mat &current_frame, Mat &cropped_frame, Rect2d &out_roi, bool draw_fusion_on_frame=false);

	// detector and tracker
	DetectorBase &_detector;
	TrackerBase &_tracker;

	// true if tracker was initialized at least once
	bool tracker_init_once;

	bool fusion_done_once;

	// previous detected/tracked centroid
	Point2d prev_centroid;

	// region of interest of input frame. Before fusing the detector and the tracker we want to activate the detector and tracker
	// only around the detected object. This can substantially decrease input size to the detector and the tracker
	// and may even increase accuracy.
	Rect2d frame_roi;

	// complementary filter for output ROI
	RectComplementaryFilter complementary_filter;
	RectLowPassFilter rect_lpf;
};



#endif /* DETECTOR_TRACKER_FUSION_H_ */
