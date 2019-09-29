#include "gstreamer_video_output.h"

void GstreamerVideoOutput::init(int res_width, int res_height, int framerate)
{
	init(VIDEO_OUT_SHARED_MEMORY, res_width, res_height, framerate);
}

void GstreamerVideoOutput::init(video_output_t output, int res_width, int res_height, int framerate)
{
	switch(output)
	{
	case VIDEO_OUT_SHARED_MEMORY:
		writer.open("appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=800,height=450"
					    " ! shmsink socket-path=/tmp/follow_me wait-for-connection=false", CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	case VIDEO_OUT_SCREEN:
		writer.open("appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=800,height=450"
							    " ! autovideosink", CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	case VIDEO_OUT_TCP_SERVER:
		writer.open("appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=800,height=450"
								" ! jpegenc quality=85 ! tcpserversink host=0.0.0.0 port=5000", CAP_GSTREAMER, 0, (double)0, cv::Size(800, 450), true);
		break;
	}

}

void GstreamerVideoOutput::output_frame(Mat &frame)
{
	writer << frame;
}
