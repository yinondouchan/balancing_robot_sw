#include "detect_markers.h"

using namespace std;
using namespace cv;

MarkerDetector::MarkerDetector(string cam_file, float marker_length, bool estimate_pose)
							  : _marker_length(marker_length), _estimate_pose(estimate_pose)
{
	
	_camera_matrix = (Mat_<double>(3, 3) << 2268.327499477305, 0.0, 252.93018136571382,
										   0., 2293.328168830158, 231.6528913818712,
										   0., 0., 1.);
	
	_dist_coeffs = (Mat_<double>(5, 1) << -1.931492963868074, -0.03488030536084996, 0.007093034450303904, 0.08102092529685556,
    71.69492121290182);
	
    // read camera calibration file
    if (cam_file.empty())
    {
        // no camera calibration file
        //_estimate_pose = false;
    }
    else
    {
		// read from yaml file
		FileStorage fs(cam_file, FileStorage::READ);
		fs["camera_matrix"] >> _camera_matrix;
		fs["dist_coeff"] >> _dist_coeffs;
		
        //_estimate_pose = true;
    }

    // create the detector params
    init_detector_params();
    
    // default dictionary: DICT_6X6_1000
    _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
}

void MarkerDetector::init_detector_params()
{
    _detector_params = aruco::DetectorParameters::create();

    // set corner refinement method to None
    _detector_params->cornerRefinementMethod = 0;
}

void MarkerDetector::detect_markers(Mat &image, vector<int> &ids, vector<vector<Point2f>> &corners,
									vector<vector<Point2f>> &rejected, vector<Vec3d> &rvecs, vector<Vec3d> &tvecs)
{
	double tick = (double)getTickCount();

	// detect markers and estimate pose
	aruco::detectMarkers(image, _dictionary, corners, ids, _detector_params, rejected);
	if(_estimate_pose && ids.size() > 0)
		aruco::estimatePoseSingleMarkers(corners, _marker_length, _camera_matrix, _dist_coeffs, rvecs,
										 tvecs);
}
