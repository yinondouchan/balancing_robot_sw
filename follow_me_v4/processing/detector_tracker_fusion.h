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
	DetectorTrackerFusion(DetectorBase detector, TrackerBase tracker);
	Rect2d output_roi();
private:
	DetectorBase _detector;
	TrackerBase _tracker;
};



#endif /* DETECTOR_TRACKER_FUSION_H_ */
