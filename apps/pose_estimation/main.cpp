#include "trt_pose.h"

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <signal.h>

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
	VideoCapture input_video;
	bool success = init_input_video(input_video, 800, 450, 800, 450, 60);
	if (!success)
	{
		std::cout << "Failed to open input video. You suck." << std::endl;
		return 0;
	}

	VideoWriter writer;
	success = init_output_video(writer, 800, 450, 60);
	if (!success)
	{
		std::cout << "Failed to open output video. You're a failure." << std::endl;
		return 0;
	}

	PoseEstimation pose_estimation;
	pose_estimation.init("../engines/resnet18_baseline_att_224x224_A_epoch_249.engine");

    Mat frame;

	// attach signal handler for SIGINT
	if( signal(SIGINT, on_signal) == SIG_ERR ) std::cout << "can't catch SIGINT" << std::endl;

    while (!sigint_received)
    {
    	input_video.grab();
    	input_video.retrieve(frame);

    	Mat inference_input;
    	resize(frame, inference_input, Size(NET_INPUT_HEIGHT, NET_INPUT_WIDTH));

    	pose_estimation.run_inference(inference_input);
    	pose_estimation.draw_output_on_frame(frame);

    	writer << frame;
    }

	/*float dummy_input[INPUT_NELEM];

    for (int i = 0; i < 500; i++)
    {
		std::cout << "Running inference on dummy input #" << i << std::endl;
		run_inference(dummy_input, executionContext, cuda_stream);
    }*/
}
