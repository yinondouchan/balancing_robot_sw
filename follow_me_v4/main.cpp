#include "detectors/yolo_detector.h"
#include "detectors/ssd_detector.h"
#include "detectors/aruco_marker_detector.h"
#include "trackers/median_flow_tracker.h"
#include "video/gstreamer_video_source.h"
#include "video/gstreamer_video_output.h"
#include "processing/detector_tracker_fusion.h"
#include "processing/location_estimation.h"
#include "robot_control/robot_controller.h"
#include "utils/camera_calibration.h"
#include "processing/exposure_factor_estimation.h"

#include <unistd.h>
#include <signal.h>

using namespace cv;

#define VIDEO_WIDTH 800
#define VIDEO_HEIGHT 450
#define VIDEO_FRAMERATE 60

// program must exit cleanly (free memory, close video, etc.) therefore we catch SIGINT
volatile bool sigint_received = false;
void on_signal(int signo)
{
	if (signo == SIGINT)
	{
		printf("\nreceived SIGINT. Leaving so early?\n");
		sigint_received = true;
	}
}

void run_follow_me()
{
	GstreamerVideoSource video_input;
	GstreamerVideoOutput video_output;
	YoloDetector object_detector;
	MedianFlowTracker medianflow_tracker;
	DetectorTrackerFusion fusion(object_detector, medianflow_tracker);
	ROILocationEstimation location_estimation;
	RobotController robot_controller;
	//ExposureFactorEstimator exposure_factor_estimator;

	object_detector.init();

	video_input.init(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);
	video_output.init(VIDEO_OUT_TCP_SERVER, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);

	// attach signal handler for SIGINT
	if( signal(SIGINT, on_signal) == SIG_ERR ) std::cout << "can't catch SIGINT" << std::endl;

	while (!sigint_received)
	{
		// grab a frame from video input
		Mat frame;
		video_input.read(frame);

		// fuse detector and tracker to find target ROI in current frame
		Rect2d roi;
		bool fusion_success = fusion.output_roi(frame, roi, true);

		if (fusion_success)
		{
			//double height_factor, width_factor;
			//exposure_factor_estimator.estimate_exposure_factors(frame, roi, height_factor, width_factor);
			//std::cout << height_factor << " " << width_factor << std::endl;

			// estimate target location given current ROI and frame
			double perp_distance;
			Point2d centroid;
			location_estimation.estimate_location(roi, frame, perp_distance, centroid);

			// control robot given target location
			robot_controller.control_by_area_and_centroid(roi.width * roi.height, centroid.x, centroid.y);
		}
		else
		{
			// if no ROI received stay put
			robot_controller.control(0, 0);
		}

		// output frame
		video_output.output_frame(frame);
	}
}

int main()
{
	run_follow_me();

//	GstreamerVideoSource video_input;
//	video_input.init(VIDEO_WIDTH, VIDEO_HEIGHT, 20);
//
//	GstreamerVideoOutput video_output;
//	video_output.init(VIDEO_OUT_TCP_SERVER, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);
//
//	ExposureFactorEstimator estimator;
//
//	// attach signal handler for SIGINT
//	if( signal(SIGINT, on_signal) == SIG_ERR ) std::cout << "can't catch SIGINT" << std::endl;
//
//	while (!sigint_received)
//	{
//		// grab a frame from video input
//		Mat frame;
//		video_input.read(frame);
//
//		Rect2d roi(300, 100, 200, 200);
//		double height_factor, width_factor;
//		estimator.estimate_exposure_factors(frame, roi, height_factor, width_factor);
//
//	    std::cout << height_factor << " " << width_factor << std::endl;
//
//	    video_output.output_frame(frame);
//	}


    return 0;
}
