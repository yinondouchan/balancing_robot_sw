#include "exposure_factor_estimation.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <assert.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>

bool ExposureFactorEstimator::fileExists(const std::string fileName, bool verbose)
{
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName)))
    {
        if (verbose) std::cout << "File does not exist : " << fileName << std::endl;
        return false;
    }
    return true;
}

ExposureFactorEstimator::ExposureFactorEstimator()
{
	// create TensorRT engine
	engine = create_trt_engine();

	// create execution context
	context = engine->createExecutionContext();

	// allocate buffers
	allocate_buffers(context);
}

ExposureFactorEstimator::~ExposureFactorEstimator()
{
	std::cout << "exposure factor estimator: freeing host and device buffers" << std::endl;
	// free device buffers
	for (int i = 0; i < buffers.size(); i++)
	{
		cudaFree(buffers[i]);
	}

	// free host buffer
	cudaFreeHost(output_host_buffer);

	std::cout << "exposure factor estimator: destroying tensorrt related stuff" << std::endl;

    if (context)
    {
    	// destroy execution context
    	context->destroy();
    	context = nullptr;
    }

    if (engine)
    {
    	// destroy engine
    	engine->destroy();
    	engine = nullptr;
    }
}

nvinfer1::ICudaEngine* ExposureFactorEstimator::create_trt_engine()
{
    // read the model from memory
	std::string engine_file_path = "../processing/exposure_factor_best.engine";
    std::cout << "Loading TRT Engine..." << std::endl;
    assert(fileExists(engine_file_path, true));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(engine_file_path);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculate model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    // instantiate the engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* deserialized_engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
    free(modelMem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return deserialized_engine;
}

void ExposureFactorEstimator::estimate_exposure_factors(Mat &frame, Rect2d roi, double &out_height_factor, double &out_width_factor)
{
	// preprocess frame to input
	Mat input;
	preprocess_image(frame, roi, input);

	// decompose 3-channel image to 3 grayscale images
	Mat input_decomposed[3];
	split(input, input_decomposed);

	// shove input to network
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int input_index = engine->getBindingIndex("input_0");
    int output_index = engine->getBindingIndex("output_0");

    // create CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU, execute the batch asynchronously, and DMA it back:
    int channel_size_bytes = EXPOSURE_FACTOR_INPUT_HEIGHT * EXPOSURE_FACTOR_INPUT_WIDTH * sizeof(float);
    CHECK(cudaMemcpyAsync(buffers[input_index], input_decomposed[0].data, channel_size_bytes, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync((int8_t*)buffers[input_index] + channel_size_bytes, input_decomposed[1].data, channel_size_bytes, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync((int8_t*)buffers[input_index] + 2 * channel_size_bytes, input_decomposed[2].data, channel_size_bytes, cudaMemcpyHostToDevice, stream));

    context->execute(1, &buffers[0]);

    // copy results to host buffers
    CHECK(cudaMemcpyAsync(output_host_buffer, buffers[output_index], EXPOSURE_FACTOR_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // wait until all operations end
    cudaStreamSynchronize(stream);

    float *result = (float*)output_host_buffer;
    out_height_factor = result[0];
    out_width_factor = result[1];
}

Rect2d ExposureFactorEstimator::clip_roi(Mat &frame, Rect2d roi)
{
	if (roi.x < 0) roi.x = 0;
	if (roi.x >= frame.cols) roi.x = frame.cols - 1;
	if (roi.y < 0) roi.y = 0;
	if (roi.y >= frame.rows) roi.y = frame.rows - 1;

	if ((roi.x + roi.width) >= frame.cols) roi.width = frame.cols - roi.x - 1;
	if ((roi.y + roi.height) >= frame.rows) roi.height = frame.rows - roi.y - 1;

	return roi;
}

void ExposureFactorEstimator::preprocess_image(Mat &frame, Rect2d &roi, Mat &out_frame)
{
	// clip ROI to fit frame boundaries
	Rect2d roi_clipped = clip_roi(frame, roi);

	// crop image
	out_frame = frame(roi_clipped);
	double crop_aspect_ratio = (double)out_frame.cols / (double)out_frame.rows;
	double input_aspect_ratio = (double)EXPOSURE_FACTOR_INPUT_WIDTH / (double)EXPOSURE_FACTOR_INPUT_HEIGHT;

	int resize_height, resize_width;

	if (crop_aspect_ratio < input_aspect_ratio)
	{
		// resize height to input height and keep aspect ratio
		resize_height = EXPOSURE_FACTOR_INPUT_HEIGHT;
		resize_width = EXPOSURE_FACTOR_INPUT_HEIGHT * crop_aspect_ratio;
	}
	else
	{
		// resize width to input width and keep aspect ratio
		resize_width = EXPOSURE_FACTOR_INPUT_WIDTH;
		resize_height = EXPOSURE_FACTOR_INPUT_WIDTH / crop_aspect_ratio;
	}

	// resize image
	resize(out_frame, out_frame, cv::Size(resize_width, resize_height));

	// letterbox image
	float pad_height = EXPOSURE_FACTOR_INPUT_HEIGHT - resize_height;
	float pad_width = EXPOSURE_FACTOR_INPUT_WIDTH - resize_width;
	int pad_top = floor(pad_height / 2);
	int pad_bottom = ceil(pad_height / 2);
	int pad_left = floor(pad_width / 2);
	int pad_right = ceil(pad_width / 2);
    copyMakeBorder(out_frame, out_frame, pad_top, pad_bottom, pad_left,
                       pad_right, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));

    // make sure returned image is in the right dimensions
    assert(out_frame.rows == EXPOSURE_FACTOR_INPUT_HEIGHT);
    assert(out_frame.cols == EXPOSURE_FACTOR_INPUT_WIDTH);

    // normalize inputs to between 0 and 1
    out_frame.convertTo(out_frame, CV_32FC3, 1/255.0);
}

// allocate host and device buffers
void ExposureFactorEstimator::allocate_buffers(IExecutionContext *context)
{
	// for now set batch size to 1
	int batchSize = 1;

	// get engine
    const ICudaEngine& engine = context->getEngine();

    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int num_of_bindings = engine.getNbBindings();
    buffers.resize(num_of_bindings, nullptr);

    int input_binding_index = engine.getBindingIndex("input_0");
    int output_binding_index = engine.getBindingIndex("output_0");

    assert(input_binding_index != -1 && "Invalid input binding index");
    assert(output_binding_index != -1 && "Invalid output binding index");

    Dims input_dims = engine.getBindingDimensions(input_binding_index);
    nvinfer1::DataType input_dtype = engine.getBindingDataType(input_binding_index);

    Dims output_dims = engine.getBindingDimensions(output_binding_index);
    nvinfer1::DataType output_dtype = engine.getBindingDataType(output_binding_index);

    // allocate input device buffer
    CHECK(cudaMalloc(&buffers.at(input_binding_index), EXPOSURE_FACTOR_INPUT_HEIGHT * EXPOSURE_FACTOR_INPUT_WIDTH * EXPOSURE_FACTOR_INPUT_CHANNELS * sizeof(float)));

    // allocate output device buffer
    CHECK(cudaMalloc(&buffers.at(output_binding_index), EXPOSURE_FACTOR_OUTPUT_SIZE * sizeof(float)));

    // allocate output host buffer
    CHECK(cudaMallocHost(&output_host_buffer, EXPOSURE_FACTOR_OUTPUT_SIZE * sizeof(float)));
}

inline void* ExposureFactorEstimator::safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}
