#ifndef DETECTOR_BASE_H
#define DETECTOR_BASE_H

#include <opencv2/highgui.hpp>

using namespace cv;

/*
 * A base class for an objet detector
 */
class DetectorBase
{
public:
    // constructor and destructor
    DetectorBase();
    ~DetectorBase();

    // initialization
    virtual void init();

    // detect objects and return their bounding boxes and class names
    virtual void detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names);

    // detect objects and return their polygons and class names
    virtual void detect(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names);

    // draw bounding boxes on an image
    void draw_bboxes_on_image(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names);

    // draw polygons on an image
    void draw_polygons_on_image(Mat &image, std::vector<std::vector<Point2f>> &out_polygons, std::vector<std::string> &class_names);

private:

};

#endif // DETECTOR_BASE_H
