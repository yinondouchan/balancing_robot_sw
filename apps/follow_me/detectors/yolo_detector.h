#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include "detector_base.h"

#include "yolo/lib/yolo.h"
#include "yolo/lib/yolov2.h"
#include "yolo/lib/yolov3.h"

/*
 * A YoloV2/YoloV3 object detector
 */
class YoloDetector : public DetectorBase
{
public:

    YoloDetector();
    ~YoloDetector();

    // initialization
    void init() override;

    // detect objects and return their bounding boxes and class names
    void detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names) override;

private:
    // convert a BBox object to a Rect2d object
    Rect2d bbox_to_rect2d(BBox bbox);

    // the network
    std::unique_ptr<Yolo> inferNet;

    // container of images of a batch
    std::vector<DsImage> batch;
};

#endif // YOLO_DETECTOR_H
