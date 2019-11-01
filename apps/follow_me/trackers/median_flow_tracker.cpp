#include "median_flow_tracker.h"

MedianFlowTracker::MedianFlowTracker()
{
	// create the tracker
	tracker = TrackerMedianFlow::create();
}

// initialize the tracker - set initial frame
void MedianFlowTracker::init(Mat &frame, Rect2d roi)
{
	// yes, yes, I clear and create the tracker from scratch. I know this is ugly but it seems that
	// clearing and initializing it doesn't work.
	tracker->clear();
	tracker = TrackerMedianFlow::create();
	tracker->init(frame, roi);
	prev_roi = roi;
}

// update the tracker - given previous frame and ROI, find the ROI in the next frame
bool MedianFlowTracker::update(Mat &frame, Rect2d &out_bbox)
{
	return tracker->update(frame, out_bbox);
}
