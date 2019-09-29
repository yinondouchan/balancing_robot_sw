/*
 * ssd_detector.h
 *
 *  Created on: Sep 26, 2019
 *      Author: yinon
 */

#ifndef SSD_DETECTOR_H_
#define SSD_DETECTOR_H_

#include "detector_base.h"

#include <sys/types.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

class SSDDetector : public DetectorBase
{
public:
	~SSDDetector();

    // initialization
    void init() override;

    // detect objects and return their bounding boxes and class names
    void detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names) override;
private:

    const char* INPUT_BLOB_NAME = "Input";
    const char* OUTPUT_BLOB_NAME0 = "NMS";
    static constexpr int OUTPUT_CLS_SIZE = 37;

    ICudaEngine *engine;
    IExecutionContext *context;

    nvinfer1::plugin::DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 100, 100, 0.5, 0.6, nvinfer1::plugin::CodeTypeSSD::TF_CENTER, {1, 2, 0}, true, true};

    // host and device buffers
    std::vector<void*> buffers;

    // CUDA stream
    cudaStream_t stream;

    // load TensorRT engine
    ICudaEngine* load_engine();

    // allocate host and device buffers
    void allocate_buffers(IExecutionContext *context);

    // calculate binding buffer sizes
    std::vector<std::pair<int64_t, nvinfer1::DataType>> calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize);

    // do the inference
    void do_inference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize);

    // destroy the detector
    void destroy();
};



#endif /* SSD_DETECTOR_H_ */
