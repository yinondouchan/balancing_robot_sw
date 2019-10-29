#include "location_estimation.h"

// estimate location: given an ROI and a frame return the perpendicular distance (meters)
// and centroid (pixels in frame, relative to frame center)
void ROILocationEstimation::estimate_location(Rect2d &roi, Mat &frame, double &distance, Point2d &centroid)
{
	// calculate distance - not implemented yet :(
	distance = 0;

	// calculate centroid
	centroid.x = (roi.x + roi.width / 2) - (frame.cols / 2);
	centroid.y = (roi.y + roi.height / 2) - (frame.rows / 2);
}
