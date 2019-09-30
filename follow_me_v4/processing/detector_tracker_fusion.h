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

	DetectorBase &_detector;
	TrackerBase &_tracker;

	bool tracker_init_once;
};



#endif /* DETECTOR_TRACKER_FUSION_H_ */
