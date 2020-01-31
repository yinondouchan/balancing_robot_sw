/*
 * trt_pose.h
 *
 *  Created on: Dec 11, 2019
 *      Author: yinon
 */

#ifndef TRT_POSE_H_
#define TRT_POSE_H_

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "configs.h"

#include <iostream>
#include <opencv2/highgui.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;

// check that a CUDA operation was successful and notify and abort if it wasn't
#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }


// Logger for TensorRT info/warning/errors
class PoseEstimationLogger : public nvinfer1::ILogger
{
public:
    PoseEstimationLogger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

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

    Severity reportableSeverity;
};

static PoseEstimationLogger gLogger;



class PoseParams
{
public:
	int max_parts;
	int max_objects;
};

class PoseEstimation
{
public:
	PoseEstimation();
	~PoseEstimation();
	void init(PoseEstimationConfig &config);
	void write_engine_to_disk(ICudaEngine *engine, std::string path);
	ICudaEngine* loadTRTEngine(const std::string planFilePath, PoseEstimationLogger& logger);
	void run_inference(Mat &input);
	void draw_output_on_frame(const Mat &frame);
	void draw_peaks_on_frame(const Mat &frame);

    // output part count vector
	std::vector<int> counts;

	// output part peak points vector C x M x 2
	std::vector<std::vector<Point2i>> peaks;

	// output refined peak points vector C x M x 2
	std::vector<std::vector<Point2f>> refined_peaks;

	// score graph K x M x M
	std::vector<Mat> score_graph;

	// connection graph K x 2 x M
	int connection_dims[3];
	Mat connections;

	// object count
	int object_count;

	// object vector OxC
	Mat objects;
private:
	void allocate_buffers();
	void free_buffers();
	void preprocess_input(Mat &input, std::vector<Mat> &output_channels);
	void postprocess_output(float *cmap_raw, float *paf_raw);

	// network and algorithm configutaion
	PoseEstimationConfig config;

	void *device_buffers[3];

	void *output0_host_buffer = nullptr;
	void *output1_host_buffer = nullptr;

	Mat output;

	double pixel_mean[3]{0.485, 0.456, 0.406};
	double pixel_stdev[3]{0.229, 0.224, 0.225};

	// TRT engine for the model
	ICudaEngine *engine;

    // execution context
    IExecutionContext *execution_context;

    // cuda stream
    cudaStream_t cuda_stream;

	int input_nelem;
	int cmap_nelem;
	int paf_nelem;
	int input_npixels;
};

#endif /* TRT_POSE_H_ */
