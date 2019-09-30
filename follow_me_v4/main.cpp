#include "detectors/yolo_detector.h"
#include "detectors/ssd_detector.h"
#include "detectors/aruco_marker_detector.h"
#include "trackers/median_flow_tracker.h"
#include "video/gstreamer_video_source.h"
#include "video/gstreamer_video_output.h"
#include "processing/detector_tracker_fusion.h"
#include "processing/location_estimation.h"
#include "robot_control/robot_controller.h"

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
			// estimate target location given current ROI and frame
			double perp_distance;
			Point2d centroid;
			location_estimation.estimate_location(roi, frame, perp_distance, centroid);

			// control robot given target location
			robot_controller.control(perp_distance, centroid.x, centroid.y);
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
	GstreamerVideoSource video_input;
	GstreamerVideoOutput video_output;

	SSDDetector detector;
	detector.init();

	video_input.init(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);
	video_output.init(VIDEO_OUT_TCP_SERVER, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);

	if( signal(SIGINT, on_signal) == SIG_ERR ) std::cout << "can't catch SIGINT" << std::endl;

	while (!sigint_received)
	{
		Mat frame;
		video_input.read(frame);

		std::vector<Rect2d> out_bboxes;
		std::vector<std::string> out_labels;
		detector.detect(frame, out_bboxes, out_labels);

		// output frame
		video_output.output_frame(frame);
	}

    return 0;
}
