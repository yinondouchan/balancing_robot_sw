#include "detector_tracker_fusion.h"

DetectorTrackerFusion::DetectorTrackerFusion(DetectorBase detector, TrackerBase tracker)
: _detector(detector), _tracker(tracker)
{

}

Rect2d DetectorTrackerFusion::output_roi()
{

}
