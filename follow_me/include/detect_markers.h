#ifndef DETECT_MARKERS_H
#define DETECT_MARKERS_H

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class MarkerDetector
{
public:
    MarkerDetector(string cam_file, float marker_length, bool estimate_pose);

    void init_detector_params();

    void detect_markers(Mat &image, vector<int> &ids, vector<vector<Point2f>> &corners,
						vector<vector<Point2f>> &rejected, vector<Vec3d> &rvecs, vector<Vec3d> &tvecs);

private:

    bool _estimate_pose;
    float _marker_length;
    bool _show_rejected;
    Mat _camera_matrix;
    Mat _dist_coeffs;
    String _video;
    Ptr<aruco::Dictionary> _dictionary;
    Ptr<aruco::DetectorParameters> _detector_params;
};

#endif // DETECT_MARKERS_H
