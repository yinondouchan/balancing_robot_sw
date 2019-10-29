#include "aruco_marker_detector.h"

#include <sstream>

void ArucoMarkerDetector::init()
{
    _detector_params = aruco::DetectorParameters::create();

    // set corner refinement method to None
    _detector_params->cornerRefinementMethod = 0;

    // default dictionary: DICT_6X6_1000
    _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
}

void ArucoMarkerDetector::detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names)
{
	// detect markers
	std::vector<std::vector<Point2f>> corners;
	std::vector<int> ids;
	std::vector<std::vector<Point2f>> rejected;
	aruco::detectMarkers(image, _dictionary, corners, ids, _detector_params, rejected);

	// convert corners to rectangles
	for (int i = 0; i < corners.size(); i++)
	{


		double min_x = corners[i][0].x;
		double min_y = corners[i][0].y;
		double max_x = corners[i][0].x;
		double max_y = corners[i][0].y;

		// find minimum and maximum coordinate values
		for (Point2f point : corners[i])
		{
			if (point.x < min_x) min_x = point.x;
			if (point.y < min_y) min_y = point.y;
			if (point.x > max_x) max_x = point.x;
			if (point.y > max_y) max_y = point.y;
		}

		// add a bounding box based on those min/max coordinate values
		out_bboxes.push_back(Rect2d(min_x, min_y, max_x - min_x, max_y - min_y));

		// add marker name
		std::stringstream marker_name;
		marker_name << "Marker " << ids[i];
		class_names.push_back(marker_name.str());
	}
}

void ArucoMarkerDetector::detect(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names)
{
	// detect markers
	std::vector<int> ids;
	std::vector<std::vector<Point2f>> rejected;
	aruco::detectMarkers(image, _dictionary, out_polygons, ids, _detector_params, rejected);

	// convert corners to rectangles
	for (int i = 0; i < ids.size(); i++)
	{
		// add marker name
		std::stringstream marker_name;
		marker_name << "Marker " << ids[i];
		class_names.push_back(marker_name.str());
	}
}
