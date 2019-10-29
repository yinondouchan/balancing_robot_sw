/*
 * exposure_factor_estimation.h
 *
 *  Created on: Oct 23, 2019
 *      Author: yinon
 */

#ifndef EXPOSURE_FACTOR_ESTIMATION_H_
#define EXPOSURE_FACTOR_ESTIMATION_H_

#include <opencv2/core.hpp>
#include <experimental/filesystem>
#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <iostream>


#define EXPOSURE_FACTOR_INPUT_HEIGHT 50
#define EXPOSURE_FACTOR_INPUT_WIDTH 50
#define EXPOSURE_FACTOR_INPUT_CHANNELS 3

#define EXPOSURE_FACTOR_OUTPUT_SIZE 2

using namespace cv;
using namespace nvinfer1;

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class ExposureFactorLogger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};


class ExposureFactorEstimator
{
public:
	ExposureFactorEstimator();

	~ExposureFactorEstimator();

	// given a frame and an ROI in that frame return the height and width exposure factors (assuming it is a person in this ROI)
	void estimate_exposure_factors(Mat &frame, Rect2d roi, double &out_height_factor, double &out_width_factor);

	// preprocess image to input dimensions
	void preprocess_image(Mat &frame, Rect2d &roi, Mat &out_frame);

private:

	// check whether a file exists
	bool fileExists(const std::string fileName, bool verbose);

	// create TensorRT engine
	nvinfer1::ICudaEngine* create_trt_engine();

    // allocate host and device buffers
    void allocate_buffers(IExecutionContext *context);

    inline void* safeCudaMalloc(size_t memSize);

    Rect2d clip_roi(Mat &frame, Rect2d roi);

	// tensorRT engine
    ICudaEngine* engine;
    IExecutionContext *context;
    ExposureFactorLogger logger;

    // host and device buffers
    std::vector<void*> buffers;
    void * output_host_buffer;

    // CUDA stream
    cudaStream_t stream;

};

#endif /* EXPOSURE_FACTOR_ESTIMATION_H_ */
