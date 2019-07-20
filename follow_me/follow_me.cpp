#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <string>
#include <time.h>

#include "arduino_serial_comm.h"
#include "detect_markers.h"
#include "follow_me.h"


#define CAMERA_FOV_DEG 75.7
#define BBOX_WIDTH 75
#define BBOX_HEIGHT 75
#define POS_DISCREPANCY_DIST_THRESHOLD 10
#define BBOX_DISCREPANCY_THRESHOLD 0.2
#define MARKER_LENGTH_CM 0.07
#define BBOX_LPF_COEFF 0.8

using namespace std;
using namespace cv;


// PID controller
PIDController::PIDController(double p_coeff, double i_coeff, double d_coeff)
							: _p_coeff(p_coeff), _i_coeff(i_coeff), _d_coeff(d_coeff)
{
	_i = 0;
	_prev_error_d = 0;
	_p_lpf_tc = 0;
	_i_lpf_tc = 0;
	_d_lpf_tc = 0;
	_error_lpf_p = 0;
	_error_lpf_i = 0;
	_error_lpf_d = 0;
}

void PIDController::set_p_lpf(double time_constant)
{
	_p_lpf_tc = time_constant;
}

void PIDController::set_i_lpf(double time_constant)
{
	_i_lpf_tc = time_constant;
}

void PIDController::set_d_lpf(double time_constant)
{
	_d_lpf_tc = time_constant;
}

double PIDController::control(double error)
{
	clock_t now = clock();
	double dt = (double)(now - _prev_time) / CLOCKS_PER_SEC;
	_prev_time = now;
	
	_error_lpf_p = _p_lpf_tc / (_p_lpf_tc + dt) * _error_lpf_p + dt / (dt + _p_lpf_tc) * error;
	_error_lpf_i = _i_lpf_tc / (_i_lpf_tc + dt) * _error_lpf_i + dt / (dt + _i_lpf_tc) * error;
	_error_lpf_d = _d_lpf_tc / (_d_lpf_tc + dt) * _error_lpf_d + dt / (dt + _d_lpf_tc) * error;
	
	double _p = _p_coeff * _error_lpf_p;
	_i += _i_coeff * _error_lpf_i * dt;
	_d = _d_coeff * (_error_lpf_d - _prev_error_d) / dt;
	
	_prev_error_d = _error_lpf_d;
	return _p + _i + _d;
}

Point get_centroid_from_marker_detection(MarkerDetector &detector, Mat &image, bool &found_marker, Vec3d &out_tvecs)
{
	Point centroid;
	
	vector< int > ids;
	vector< vector< Point2f > > corners, rejected;
	vector< Vec3d > rvecs, tvecs;
	
	detector.detect_markers(image, ids, corners, rejected, rvecs, tvecs);
	
	if (tvecs.size() > 0) out_tvecs = tvecs[0];
	
	found_marker = ids.size() > 0;
	
	if(ids.size() > 0) {
		// find centroid of first corner
		centroid.x = 0.25 * (corners[0][0].x + corners[0][1].x + corners[0][2].x + corners[0][3].x);
		centroid.y = 0.25 * (corners[0][0].y + corners[0][1].y + corners[0][2].y + corners[0][3].y);
	}
	
	return centroid;
}

Point get_centroid_from_object_tracking(Ptr<Tracker> tracker, Mat &frame, Rect2d &bbox, bool &track_ok)
{
	Point centroid;
	track_ok = tracker->update(frame, bbox);
	
	if (track_ok)
	{
		centroid.x = 0.5 * (bbox.tl().x + bbox.br().x);
		centroid.y = 0.5 * (bbox.tl().y + bbox.br().y);
	}
	
	return centroid;
}

KalmanFilter* init_bbox_kf()
{
	KalmanFilter *kf = new KalmanFilter(6, 3, 0);
	kf->transitionMatrix = (Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	kf->statePre.at<float>(0) = 0;
	kf->statePre.at<float>(1) = 0;
	kf->statePre.at<float>(2) = 0;
	kf->statePre.at<float>(3) = 0;
	kf->statePre.at<float>(4) = 0;
	kf->statePre.at<float>(5) = 0;
	setIdentity(kf->measurementMatrix);
	setIdentity(kf->processNoiseCov, Scalar::all(1e-4));
	setIdentity(kf->measurementNoiseCov, Scalar::all(10));
	setIdentity(kf->errorCovPost, Scalar::all(.1));
	
	return kf;
}

void apply_lpf_on_bbox(Rect2d &bbox_lpf, Rect2d &bbox, double lpf_const)
{
	bbox_lpf.x = lpf_const * bbox.x + (1 - lpf_const) * bbox_lpf.x;
	bbox_lpf.y = lpf_const * bbox.y + (1 - lpf_const) * bbox_lpf.y;
	bbox_lpf.width = lpf_const * bbox.width + (1 - lpf_const) * bbox_lpf.width;
	bbox_lpf.height = lpf_const * bbox.height + (1 - lpf_const) * bbox_lpf.height;
}

void move_robot_according_to_centroid_and_distance(Point &centroid, double distance, int frame_width, ArduinoSerialComm &serial_comm, PIDController &vel_pid, PIDController &ang_vel_pid)
{
	// given the camera's FOV convert centroid to angle
	int width_res = frame_width;
	double angle_x = CAMERA_FOV_DEG * (centroid.x - width_res / 2) / width_res;
	int vel = vel_pid.control(distance - 2.0);
	
	// move robot accordingly
	int ang_vel = ang_vel_pid.control(angle_x);
	serial_comm.write_velocity(vel, ang_vel);
}

string gstreamer_input_pipeline(int width, int height)
{
    stringstream input_pipeline;
    input_pipeline << "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=" << width << ", height=" << height
                      << ", framerate=60/1, format=NV12 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink emit-signals=true sync=false max-buffers=2 drop=true";
    return input_pipeline.str();
}

string gstreamer_output_pipeline(int width, int height)
{
    stringstream output_pipeline;
    output_pipeline << "appsrc ! video/x-raw,format=BGR ! videoconvert ! video/x-raw,format=I420,width=" << width
                        << ",height=" << height << " ! shmsink socket-path=/tmp/follow_me";
    return output_pipeline.str();
}

int main(int argc, char *argv[]) {
    ArduinoSerialComm serial_comm;
    
    String video;
    VideoCapture input_video;
    int cam_id = 0;
    int frame_counter = 0;
    int wait_time;
    bool found_marker_first_time = false;

    // angular and linear velocity PID controllers
    PIDController ang_vel_pid(15, 0, 0.4);
    PIDController vel_pid(140, 0, 0.75);
    
    int cam_width = 800;
    int cam_height = 450;

    // start capturing video
    input_video.open(gstreamer_input_pipeline(cam_width, cam_height), CAP_GSTREAMER);
    if (!input_video.isOpened())
    {
        cout << "failed to open camera" << endl;
        return -1;
    }

    //cout << input_video.get(CAP_PROP_FRAME_WIDTH) << endl;
    //cout << input_video.get(CAP_PROP_FRAME_HEIGHT) << endl;

    cv::VideoWriter video_writer;
    cout << gstreamer_input_pipeline(cam_width, cam_height) << endl;
    cout << gstreamer_output_pipeline(cam_width, cam_height) << endl;
    video_writer.open(gstreamer_output_pipeline(cam_width, cam_height), CAP_GSTREAMER, 0, (double)0, cv::Size(cam_width, cam_height), true);
    if (!video_writer.isOpened()) {
        cout << "Failed to open video writer" << endl;
        return -1;
    }

    cout << "camera opened successfully" << endl;

    wait_time = 10;
    
    MarkerDetector detector("", MARKER_LENGTH_CM, true);
    
    // object tracker
    Ptr<Tracker> tracker;
    TrackerMedianFlow::Params tracker_params;
    tracker = TrackerMedianFlow::create(tracker_params);
    Rect2d bbox(150, 150, BBOX_WIDTH, BBOX_HEIGHT);
    Rect2d bbox_lpf(150, 150, BBOX_WIDTH, BBOX_HEIGHT);
    
    // bounding box kalman filter
    KalmanFilter *bbox_kf = init_bbox_kf();
    Mat_<float> bbox_measurement(3,1); bbox_measurement.setTo(Scalar(0));
    Mat_<float> bbox_estimated(3,1); bbox_estimated.setTo(Scalar(0));
    
    Mat frame;
    bool ok = input_video.read(frame);
    //tracker->init(frame, bbox);
    
    Point centroid;
    bool found_marker;
    bool track_ok;
    double dist_over_bbox_width;
    double distance;
    
    int counter = 0;

    while(input_video.grab()) {
		found_marker = false;
		track_ok = false;
		
		// marker translation vector
		Vec3d tvec;
		
		// predict bbox
		//Mat bbox_prediction = bbox_kf->predict();
		
		input_video.retrieve(frame);
		
		if (frame_counter % 10 == 0)
		{
			// try detecting markers
			centroid = get_centroid_from_marker_detection(detector, frame, found_marker, tvec);
		}
		
		if (found_marker)
		{
			double bbox_height, bbox_width;
			double distance_from_marker = sqrt(tvec[0]*tvec[0] + tvec[1]*tvec[1] + tvec[2]*tvec[2]);

			bbox_height = 150.0 / distance_from_marker;
			bbox_width = 150.0 / distance_from_marker;
			
			//cout << "dt" << distance_from_marker << endl << flush;
			
			dist_over_bbox_width = bbox_width * distance_from_marker;		
			bbox = Rect2d(bbox.x, bbox.y, bbox_width, bbox_height);
			
			// update the tracker's bounding box
			bbox.x = centroid.x - bbox_width/2;
			bbox.y = centroid.y - bbox_height/2;
			
			if (!found_marker_first_time)
			{
				// set bbox lpf for the first time as the first bbox received
				bbox_lpf = Rect2d(bbox.x, bbox.y, bbox_width, bbox_height);
				found_marker_first_time = true;
			}
			
			// re-initialize tracker
			tracker->clear();
			tracker = TrackerMedianFlow::create(tracker_params);
			tracker->init(frame, bbox);
			
			//bbox_measurement(0) = bbox.x;
			//bbox_measurement(1) = bbox.y;
			//bbox_measurement(2) = bbox.width;
			//bbox_estimated = bbox_kf->correct(bbox_measurement);
		}
		else
		{
			// if didn't find marker - try tracking object
			track_ok = tracker->update(frame, bbox);
			
			if (track_ok)
			{
				centroid.x = 0.5 * (bbox.tl().x + bbox.br().x);
				centroid.y = 0.5 * (bbox.tl().y + bbox.br().y);
				
				distance = dist_over_bbox_width / bbox.size().width;
														  
				bbox_measurement(0) = bbox.x;
				bbox_measurement(1) = bbox.y;
				bbox_measurement(2) = bbox.width;
				bbox_estimated = bbox_kf->correct(bbox_measurement);
			}
			else
			{
				// both marker detection and tracking failed - stay put
				serial_comm.write_velocity(0, 0);
			}
		}
		
		if(found_marker || track_ok)
		{
			//apply_lpf_on_bbox(bbox_lpf, bbox, 0.2);
			
			//circle(frame, centroid, 1, CV_RGB(255, 255, 255), 3);
			rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
			
			//Rect2d bbox_filtered(bbox_estimated(0), bbox_estimated(1), bbox_estimated(2), bbox_estimated(2));
			
			centroid.x = 0.5 * (bbox.tl().x + bbox.br().x);
			centroid.y = 0.5 * (bbox.tl().y + bbox.br().y);
			distance = dist_over_bbox_width / bbox.size().width;
			move_robot_according_to_centroid_and_distance(centroid, distance, input_video.get(CAP_PROP_FRAME_WIDTH),
														  serial_comm, vel_pid, ang_vel_pid);
		}
		
		//imshow("marker", frame);
        video_writer << frame;
		
		frame_counter++;
		
		char key = (char)waitKey(wait_time);
        if(key == 'q') break;
        
	}
	
    input_video.release();
    video_writer.release();
	delete(bbox_kf);
    return 0;
}
