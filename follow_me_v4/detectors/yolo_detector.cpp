#include "yolo_detector.h"
#include "yolo/lib/yolo_config_parser.h"

YoloDetector::YoloDetector()
{
}

YoloDetector::~YoloDetector()
{
}

// initialization
void YoloDetector::init()
{
    int argc = 2;
    char *argv[2] = {"", "--flagfile=../detectors/yolo/config/yolov2.txt"};

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    std::string testImages = getTestImages();
    std::string testImagesPath = getTestImagesPath();
    bool decode = getDecode();
    bool doBenchmark = getDoBenchmark();
    bool viewDetections = getViewDetections();
    bool saveDetections = getSaveDetections();
    std::string saveDetectionsPath = getSaveDetectionsPath();
    uint batchSize = getBatchSize();
    bool shuffleTestSet = getShuffleTestSet();

    // for now use YoloV2
    inferNet = std::unique_ptr<Yolo>{new YoloV2(batchSize, yoloInfo, yoloInferParams)};
}

// convert a BBox object to a Rect2d object
Rect2d YoloDetector::bbox_to_rect2d(BBox bbox)
{
    return Rect2d(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
}

// detect objects and return their bounding boxes and class names
void YoloDetector::detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names)
{
    // clear batch container
    batch.clear();

    // add image to batch (yes, we use batch size 1 since it is a live video)
    batch.emplace_back(image, inferNet->getInputH(), inferNet->getInputW());

    // preprocess batch (letterbox, rescale image, etc.)
    cv::Mat trtInput = blobFromDsImages(batch, inferNet->getInputH(), inferNet->getInputW());

    // do the inference
    inferNet->doInference(trtInput.data, batch.size());

    // decode the output
    auto curImage = batch.at(0);
    auto binfo = inferNet->decodeDetections(0, curImage.getImageHeight(),
                                            curImage.getImageWidth());

    // do non-max suppression on the output
    auto remaining = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());

    // extract the bounding boxes and their labels
    for (auto b : remaining)
    {
        // get bounding box as a Rect2d object
        out_bboxes.push_back(bbox_to_rect2d(b.box));

        // get the label name of this bounding box
        class_names.push_back(inferNet->getClassName(b.label));
    }
}
