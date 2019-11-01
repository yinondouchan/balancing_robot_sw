#include "detector_base.h"

#include <opencv2/imgproc.hpp>

using namespace cv;

DetectorBase::DetectorBase(){}

DetectorBase::~DetectorBase(){}

// initialization
void DetectorBase::init() {}

// detect objects and return their bounding boxes and class names
void DetectorBase::detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names) {}

// detect objects and return their polygons and class names
void DetectorBase::detect(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names) {}

// draw bounding boxes on an image
void DetectorBase::draw_bboxes_on_image(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names)
{
    for (int i = 0; i < out_bboxes.size(); i++)
    {
        auto bbox = out_bboxes[i];
        auto label = class_names[i];
        rectangle(image, bbox, Scalar( 0, 255, 0 ), 2, 1 );
        putText(image, label, Point(bbox.x, bbox.y), FONT_HERSHEY_DUPLEX, 1, Scalar( 0, 0, 255 ));
    }
}

// draw polygons on an image
void DetectorBase::draw_polygons_on_image(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names)
{
    for (int i = 0; i < out_polygons.size(); i++)
    {
        auto polygon = out_polygons[i];
        auto label = class_names[i];

        // minimal y value and point with minimal y value (for placing text)
        double min_y = INT_MAX;
        int argmin_y = -1;

        // draw lines
        for (int j = 0; j < polygon.size(); j++)
        {
        	// find min and argmin y coordinate
        	if (polygon[j].y < min_y)
        	{
        		min_y = polygon[j].y;
        		argmin_y = j;
        	}

        	// draw a line between two of the polygon's points
        	line(image, polygon[j], polygon[(j+1) % polygon.size()], Scalar( 0, 255, 0 ), 2, 1 );
        }

        putText(image, label, polygon[argmin_y], FONT_HERSHEY_DUPLEX, 1, Scalar( 0, 0, 255 ));
    }
}
