/*
 * median_flow_tracker.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef MEDIAN_FLOW_TRACKER_H_
#define MEDIAN_FLOW_TRACKER_H_

#include "tracker_base.h"

#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

class MedianFlowTracker : public TrackerBase
{
public:
	MedianFlowTracker();

	// initialize the tracker - set initial frame
	void init(Mat &frame, Rect2d ROI) override;

	// update the tracker - given previous frame and ROI, find the ROI in the next frame
	bool update(Mat &frame, Rect2d &out_bbox) override;

private:
	// previous frame and ROI
	Rect2d prev_roi;

	// the tracker's openCV implementation
	Ptr<TrackerMedianFlow> tracker;
};


#endif /* MEDIAN_FLOW_TRACKER_H_ */
