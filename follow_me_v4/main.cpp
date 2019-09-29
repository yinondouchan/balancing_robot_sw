#include "detectors/yolo_detector.h"
#include "detectors/ssd_detector.h"
#include "detectors/aruco_marker_detector.h"
#include "trackers/median_flow_tracker.h"
#include "video/gstreamer_video_source.h"
#include "video/gstreamer_video_output.h"
#include "robot_control/robot_controller.h"

#include <unistd.h>

using namespace cv;

#define VIDEO_WIDTH 800
#define VIDEO_HEIGHT 450
#define VIDEO_FRAMERATE 60

void test_yolo_detector()
{
    YoloDetector detector;
    detector.init();

    Mat test_image = imread("../test_images/2.jpg");

    std::vector<Rect2d> bboxes;
    std::vector<std::string> labels;
    detector.detect(test_image, bboxes, labels);

    for (int i = 0; i < bboxes.size(); i++)
    {
        auto bbox = bboxes[i];
        auto label = labels[i];
        rectangle(test_image, bbox, Scalar( 0, 255, 0 ), 2, 1 );
        putText(test_image, label, Point(bbox.x, bbox.y), FONT_HERSHEY_DUPLEX, 1, Scalar( 0, 0, 255 )/*, int thickness=1, int lineType=8, bool bottomLeftOrigin=false*/);
    }

    imwrite("out_image.png", test_image);
}

Point2d get_centroid_from_rect2d(Rect2d &rect)
{
	return Point2d(rect.x + rect.width/2, rect.y + rect.height/2);
}

void detect_and_track_person(DetectorBase &detector, TrackerBase &tracker, VideoSourceBase &video_input, VideoOutputBase &video_output, RobotController &robot_controller)
{

	// true if tracker was initialized at least once
	bool tracker_init_once = false;

	int frame_counter = 0;

	Point2d centroid;

	while(true)
	{
		Mat frame;
		video_input.read(frame);

		if ((frame_counter % 1) == 0)
		{
			// detect objects
			std::vector<Rect2d> object_bboxes;
			std::vector<std::string> object_labels;
			detector.detect(frame, object_bboxes, object_labels);
			detector.draw_bboxes_on_image(frame, object_bboxes, object_labels);

			if (object_bboxes.size())
			{
				int person_index = -1;
				// assume one person in frame
				for (int i = 0; i < object_labels.size(); i++)
				{
					if (object_labels[i].compare("person") == 0)
					{
						person_index = i;
					}
				}

				if (person_index != -1)
				{
					tracker.init(frame, object_bboxes[0]);
					tracker_init_once = true;

					Point2d centroid = get_centroid_from_rect2d(object_bboxes[0]);

					circle(frame, centroid, 5, Scalar(0, 255, 0), 3);
					robot_controller.control(0, 450 - centroid.x, 225 - centroid.y);
				}
				else
				{
					robot_controller.control(0, 0);
				}
			}
			else
			{
				Rect2d out_bbox;
				bool track_ok = tracker.update(frame, out_bbox);
				if (track_ok)
				{
					tracker.draw_roi_on_frame(frame, out_bbox, Scalar(255, 0, 0));
					circle(frame, get_centroid_from_rect2d(out_bbox), 5, Scalar(255, 0, 0), 3);

					Point2d centroid = get_centroid_from_rect2d(out_bbox);
					robot_controller.control(0, 450 - centroid.x, 225 - centroid.y);
				}
				else
				{
					robot_controller.control(0, 0);
				}
			}
		}
		else if (tracker_init_once)
		{
			Rect2d out_bbox;
			tracker.update(frame, out_bbox);
			tracker.draw_roi_on_frame(frame, out_bbox, Scalar(255, 0, 0));
		}

		// detect markers
	    //std::vector<Rect2d> marker_bboxes;
	    //std::vector<std::string> marker_labels;
		//marker_detector.detect(frame, marker_bboxes, marker_labels);
		//marker_detector.draw_bboxes_on_image(frame, marker_bboxes, marker_labels);


		video_output.output_frame(frame);

		frame_counter++;
	}
}

int main()
{
	YoloDetector object_detector;
	MedianFlowTracker medianflow_tracker;
	ArucoMarkerDetector marker_detector;
	GstreamerVideoSource video_input;
	GstreamerVideoOutput video_output;
	RobotController robot_controller;

	object_detector.init();

	video_input.init(VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);
	video_output.init(VIDEO_OUT_TCP_SERVER, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FRAMERATE);

	detect_and_track_person(object_detector, medianflow_tracker, video_input, video_output, robot_controller);

//	marker_detector.init();
//	object_detector.init();
//
//	video_input.init();
//	video_output.init(VIDEO_OUT_TCP_SERVER);
//
//	int ctr = 0;
//
//	while(true)
//	{
//		Mat frame;
//		video_input.read(frame);
//
//		// detect objects
//		std::vector<Rect2d> object_bboxes;
//		std::vector<std::string> object_labels;
//		object_detector.detect(frame, object_bboxes, object_labels);
//		object_detector.draw_bboxes_on_image(frame, object_bboxes, object_labels);
//
//		// detect markers
//		/*std::vector<std::vector<Point2f>> polygons;
//		std::vector<std::string> marker_labels;
//		marker_detector.detect(frame, polygons, marker_labels);
//		marker_detector.draw_polygons_on_image(frame, polygons, marker_labels);*/
//
//		video_output.output_frame(frame);
//
//		ctr++;
//	}

    return 0;
}
