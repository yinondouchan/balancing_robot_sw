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

#include <iostream>
#include <opencv2/highgui.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;

// input dimensions
#define NET_INPUT_WIDTH 224
#define NET_INPUT_HEIGHT 224
#define NET_INPUT_CHANNELS 3

// input number of elements
#define INPUT_NELEM NET_INPUT_WIDTH * NET_INPUT_HEIGHT * NET_INPUT_CHANNELS

// number of pixels in input
#define INPUT_NPIXELS NET_INPUT_WIDTH * NET_INPUT_HEIGHT

// width and height of both the confidence maps and part affinity fields
#define OUTPUT_MAP_DIM 56

// number of parts (elbow, eye, knee, etc.)
#define NUM_PARTS 18

// number of links (neck, leg, etc.)
#define NUM_LINKS 21

// number of elements in confidence map
#define CMAP_NELEM NUM_PARTS * OUTPUT_MAP_DIM * OUTPUT_MAP_DIM

// number of elements in part affinity fields
#define PAF_NELEM 2 * NUM_LINKS * OUTPUT_MAP_DIM * OUTPUT_MAP_DIM

#define PEAK_CONFIDENCE_THRESHOLD 0.1

#define LINK_THRESHOLD 0.1

#define PEAKS_WINDOW_SIZE 5

#define MAX_PARTS 100

#define MAX_OBJECTS 100

#define LINE_INTEGRAL_SAMPLES 7

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
	void init(std::string engine_path);
	void write_engine_to_disk(ICudaEngine *engine, std::string path);
	ICudaEngine* loadTRTEngine(const std::string planFilePath, PoseEstimationLogger& logger);
	void run_inference(Mat &input);
	void draw_output_on_frame(const Mat &frame);
private:
	void init_topology();
	void allocate_buffers();
	void free_buffers();
	void preprocess_input(Mat &input, std::vector<Mat> &output_channels);
	void postprocess_output(float *cmap_raw, float *paf_raw);

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

    // output part count vector
	std::vector<int> counts;

	// output part peak points vector C x M x 2
	std::vector<std::vector<Point2i>> peaks;

	// output refined peak points vector C x M x 2
	std::vector<std::vector<Point2f>> refined_peaks;

	// score graph K x M x M
	std::vector<Mat> score_graph;

	// connection graph K x 2 x M
	const int connection_dims[3]{NUM_LINKS, 2, MAX_PARTS};
	Mat connections;

	// object count
	int object_count;

	// object vector OxC
	Mat objects;

	// topology
	std::vector<std::array<int,4>> topology;

	// part inds to meaning
	const std::vector<std::string> keypoints{"nose", "left_eye", "right_eye", "left_ear", "right_ear",
		"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
		"left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"};

	// linkage inds to part inds
	const std::vector<std::pair<int, int>> skeleton{
				std::make_pair(16, 14),
				std::make_pair(14, 12),
				std::make_pair(17, 15),
				std::make_pair(15, 13),
				std::make_pair(12, 13),
				std::make_pair(6, 8),
				std::make_pair(7, 9),
				std::make_pair(8, 10),
				std::make_pair(9, 11),
				std::make_pair(2, 3),
				std::make_pair(1, 2),
				std::make_pair(1, 3),
				std::make_pair(2, 4),
				std::make_pair(3, 5),
				std::make_pair(4, 6),
				std::make_pair(5, 7),
				std::make_pair(18, 1),
				std::make_pair(18, 6),
				std::make_pair(18, 7),
				std::make_pair(18, 12),
				std::make_pair(18, 13)
	};
};

#endif /* TRT_POSE_H_ */
