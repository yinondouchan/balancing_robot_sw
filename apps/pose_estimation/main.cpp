#include "trt_pose.h"
#include "pose_tracker.hpp"

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <signal.h>
#include <chrono>

using namespace cv;

// program must exit cleanly (free memory, close video, etc.) therefore we catch SIGINT
volatile bool sigint_received = false;
void on_signal(int signo)
{
	if (signo == SIGINT)
	{
		printf("\nreceived SIGINT. Where the hell do you think you're going?\n");
		sigint_received = true;
	}
}

bool init_input_video(VideoCapture &video, int res_width, int res_height, int argus_width, int argus_height, int framerate)
{
	std::stringstream input_stream;
	input_stream << "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" << argus_width << ", height=" << argus_height << ","
			" framerate=" << framerate << "/1, format=NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx, width=" << res_width << ", height=" << res_height <<
			" ! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true";
	return video.open(input_stream.str(), CAP_GSTREAMER);
}

bool init_output_video(VideoWriter &writer, int res_width, int res_height, int framerate)
{
	std::stringstream output_stream;
	output_stream << "appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=" << res_width
	<< ",height=" << res_height << " ! jpegenc quality=85 ! tcpserversink host=0.0.0.0 port=5000";
	return writer.open(output_stream.str(), CAP_GSTREAMER, 0, (double)0, cv::Size(res_width, res_height), true);
}

int main(int argc, char** argv)
{
	int capture_width = 1280;
	int capture_height = 720;
	int input_width = 800;
	int input_height = 450;
	int input_framerate = 60;

	VideoCapture input_video;
	bool success = init_input_video(input_video, input_width, input_height, capture_width, capture_height, input_framerate);
	if (!success)
	{
		std::cout << "Failed to open input video stream. You suck." << std::endl;
		return 0;
	}

	VideoWriter writer;
	success = init_output_video(writer, input_width, input_height, input_framerate);
	if (!success)
	{
		std::cout << "Failed to open output video stream. You're a failure." << std::endl;
		return 0;
	}

//	Resnet18Size224Config pose_estimation_config("../engines/resnet18_baseline_att_224x224_A_epoch_249_fp16.engine");
//	Densenet121Size256Config pose_estimation_config("/home/yinon/dev/trt_pose/tasks/hum-an_pose/densenet121_baseline_att_256x256_B_epoch_160_fp16.engine");
	BodyAndFeetPose224Config pose_estimation_config("../engines/body_foot_crop_25ep.engine");
//	BodyOnlyPose224Config pose_estimation_config("../engines/body_only_25ep.engine");
//	HandPose224Config pose_estimation_config("../engines/hand_pose_wider.engine");
//	pose_estimation_config.link_threshold = 0.2;
//	pose_estimation_config.peak_confidence_threshold = 0.2;
	pose_estimation_config.output_map_size = Size(56, 56);

	PoseEstimation pose_estimation;
	PoseTracker pose_tracker;
	pose_estimation.init(pose_estimation_config);

    Mat frame;

	// attach signal handler for SIGINT
	if( signal(SIGINT, on_signal) == SIG_ERR ) std::cout << "can't catch SIGINT" << std::endl;

    while (!sigint_received)
    {
    	input_video.grab();
    	input_video.retrieve(frame);

    	Mat inference_input;
    	resize(frame, inference_input, pose_estimation_config.input_size);

    	pose_estimation.run_inference(inference_input);

    	std::vector<Rect2f> keypoint_bboxes;
    	//pose_tracker.get_keypoint_bboxes(keypoint_bboxes, frame, pose_estimation.object_count, pose_estimation.objects, pose_estimation.refined_peaks);
    	//pose_tracker.draw_keypoint_bboxes_on_frame(frame, keypoint_bboxes);

    	pose_estimation.draw_output_on_frame(frame);


    	writer << frame;
    }
}
