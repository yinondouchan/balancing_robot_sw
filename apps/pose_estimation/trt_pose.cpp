
#include "trt_pose.h"
#include "find_peaks.hpp"
#include "refine_peaks.hpp"
#include "paf_score_graph.hpp"
#include "munkres.hpp"
#include "connect_parts.hpp"

#include <iostream>
#include <assert.h>
#include <sstream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;


PoseEstimation::PoseEstimation()
{

}

PoseEstimation::~PoseEstimation()
{
	free_buffers();

    execution_context->destroy();
    execution_context = nullptr;
}

void PoseEstimation::init(std::string engine_path)
{
	std::cout << "Loading OpenPose TRT engine" << std::endl;
	init_topology();

	// load TRT engine
	engine = loadTRTEngine(engine_path, gLogger);

    // create execution context
    execution_context = engine->createExecutionContext();

    // allocate host and device buffers
	allocate_buffers();

    // create a cuda stream
    std::cout << "Creating CUDA stream" << std::endl;
    cudaStreamCreate(&cuda_stream);

	// init peaks and counts vectors
	counts.resize(NUM_PARTS);
	peaks.resize(NUM_PARTS);
	for (int i = 0; i < NUM_PARTS; i++)
	{
		peaks[i].resize(MAX_PARTS);
		for (int j = 0; j < MAX_PARTS; j++)
		{
			peaks[i][j].x = 0;
			peaks[i][j].y = 0;
		}
	}

	// init refined peaks vectors
	refined_peaks.resize(NUM_PARTS);
	for (int i = 0; i < NUM_PARTS; i++)
	{
		refined_peaks[i].resize(MAX_PARTS);
		for (int j = 0; j < MAX_PARTS; j++)
		{
			refined_peaks[i][j].x = 0.0;
			refined_peaks[i][j].y = 0.0;
		}
	}

	// init score graph
	score_graph.resize(NUM_LINKS);
	for (int k = 0; k < score_graph.size(); k++)
	{
		score_graph[k].create(Size(MAX_PARTS, MAX_PARTS), CV_32FC1);
	}

	// init connection graph. Set all of its values to -1
	connections.create(3, connection_dims, CV_32SC1);
	connections.setTo(Scalar(-1));

	objects.create(Size(MAX_OBJECTS, NUM_PARTS), CV_32SC1);
	objects.setTo(Scalar(-1));
}

void PoseEstimation::init_topology()
{
	topology.resize(NUM_LINKS);
	for (int k = 0; k < skeleton.size(); k++)
	{
        topology[k][0] = 2 * k;
        topology[k][1] = 2 * k + 1;
        topology[k][2] = skeleton[k].first - 1;
        topology[k][3] = skeleton[k].second - 1;
	}
}

void PoseEstimation::write_engine_to_disk(ICudaEngine *engine, std::string path)
{
    std::cout << "Serializing the TensorRT Engine..." << std::endl;
    std::cout << engine << std::endl;
    assert(engine && "Invalid TensorRT Engine");
    nvinfer1::IHostMemory* model_stream = engine->serialize();
    assert(model_stream && "Unable to serialize engine");
    assert(!path.empty() && "Engine path is empty");
    std::cout << "Serialized the TensorRT Engine..." << std::endl;

    // write data to output file
    std::cout << "Writing serialized engine to file" << std::endl;
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write(static_cast<const char*>(model_stream->data()), model_stream->size());
    std::ofstream outFile;
    outFile.open(path);
    outFile << gieModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << path << std::endl;
}

ICudaEngine* PoseEstimation::loadTRTEngine(const std::string planFilePath, PoseEstimationLogger& logger)
{
    // reading the model in memory
    //assert(fileExists(planFilePath));
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache(planFilePath);
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

// allocate device buffers
void PoseEstimation::allocate_buffers()
{
	std::cout << "trt_pose: Allocating buffers" << std::endl;

    // allocate device memory for input
    NV_CUDA_CHECK(cudaMalloc(&device_buffers[0], INPUT_NELEM * sizeof(float)));

    NV_CUDA_CHECK(cudaMalloc(&device_buffers[1], CMAP_NELEM * sizeof(float)));

    NV_CUDA_CHECK(cudaMalloc(&device_buffers[2], PAF_NELEM * sizeof(float)));

    // allocate host memory for output - same dimensions as device memory for output
    NV_CUDA_CHECK(cudaMallocHost(&output0_host_buffer, CMAP_NELEM * sizeof(float)));

    // allocate host memory for output - same dimensions as device memory for output
    NV_CUDA_CHECK(cudaMallocHost(&output1_host_buffer, PAF_NELEM * sizeof(float)));
}

void PoseEstimation::free_buffers()
{
	std::cout << "trt_pose: Freeing buffers" << std::endl;

	NV_CUDA_CHECK(cudaFreeHost(output0_host_buffer));
	NV_CUDA_CHECK(cudaFreeHost(output1_host_buffer));
    NV_CUDA_CHECK(cudaFree(device_buffers[0]));
    NV_CUDA_CHECK(cudaFree(device_buffers[1]));
    NV_CUDA_CHECK(cudaFree(device_buffers[2]));
}

void PoseEstimation::preprocess_input(Mat &input, std::vector<Mat> &output_channels)
{
	// convert to float16 and scale input from range 0-255 to range 0-1
	input.convertTo(output, CV_32FC3, 1.0 / 255.0);

	// subtract by mean
	output = output - Scalar(pixel_mean[0], pixel_mean[1], pixel_mean[2]);

	// divide by std
	multiply(output, Scalar(1.0 / pixel_stdev[0], 1.0 / pixel_stdev[1], 1.0 / pixel_stdev[2]), output);

	// separate image to channels
	split(output, output_channels);
}

void PoseEstimation::run_inference(Mat &input)
{
	std::vector<Mat> output_channels;

	// preprocess the input. Retrieve output as separate channels
	preprocess_input(input, output_channels);

	// input should be formatted as CHW and RGB while Mat objects are formatted as HWC and BGR.
	// therefore copy to buffer one channel after another in RGB order
	NV_CUDA_CHECK(cudaMemcpyAsync((float*)device_buffers[0], output_channels[2].data,
									INPUT_NPIXELS * sizeof(float), cudaMemcpyHostToDevice,
									cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync((float*)device_buffers[0] + INPUT_NPIXELS, output_channels[1].data,
									INPUT_NPIXELS * sizeof(float), cudaMemcpyHostToDevice,
									cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync((float*)device_buffers[0] + 2 * INPUT_NPIXELS, output_channels[0].data,
									INPUT_NPIXELS * sizeof(float), cudaMemcpyHostToDevice,
									cuda_stream));

	// do the inference
	execution_context->enqueue(1, device_buffers, cuda_stream, nullptr);

	// copy output from device buffer to host buffer
	NV_CUDA_CHECK(cudaMemcpyAsync(output0_host_buffer, device_buffers[1],
								  CMAP_NELEM * sizeof(float),
								  cudaMemcpyDeviceToHost, cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync(output1_host_buffer, device_buffers[2],
								  PAF_NELEM * sizeof(float),
								  cudaMemcpyDeviceToHost, cuda_stream));

	// block until all GPU-related operations have ended for this inference
	cudaStreamSynchronize(cuda_stream);

	// raw confidence maps and part affinity fields
	float *cmaps_raw = (float*)output0_host_buffer;
	float *pafs_raw = (float*)output1_host_buffer;

	postprocess_output(cmaps_raw, pafs_raw);
}

void PoseEstimation::postprocess_output(float *cmaps_raw, float *pafs_raw)
{
	// convert confidence maps to a vector of Mat objects
	std::vector<Mat> cmaps;

	// convert part affinity fields to a vector of Mat objects
	std::vector<Mat> pafs;

	// populate confidence map
	for (int i = 0; i < NUM_PARTS; i++)
	{
		// reference the raw data from a vector of Mat objects
		cmaps.push_back(Mat(Size(OUTPUT_MAP_DIM, OUTPUT_MAP_DIM), CV_32FC1, cmaps_raw + i * OUTPUT_MAP_DIM * OUTPUT_MAP_DIM));
	}

	// populate part affinity field
	for (int i = 0; i < (2 * NUM_LINKS); i++)
	{
		// reference the raw data from a vector of Mat objects
		pafs.push_back(Mat(Size(OUTPUT_MAP_DIM, OUTPUT_MAP_DIM), CV_32FC1, pafs_raw + i * OUTPUT_MAP_DIM * OUTPUT_MAP_DIM));
	}

	// find peaks in confidence maps
	find_peaks(counts, peaks, cmaps, PEAK_CONFIDENCE_THRESHOLD, PEAKS_WINDOW_SIZE, MAX_PARTS);

	// refine found peaks
	refine_peaks(refined_peaks, counts, peaks, cmaps, PEAKS_WINDOW_SIZE);

	// find part affinity scores for each link
	paf_score_graph(score_graph, pafs, topology, counts, refined_peaks, LINE_INTEGRAL_SAMPLES);

	// perform munkres assignment to find the connection graph
	connections.setTo(Scalar(-1));
	assignment(connections, score_graph, topology, counts, LINK_THRESHOLD);

	// connect parts to obtain final output
	objects.setTo(Scalar(-1));
	connect_parts(object_count, objects, connections, topology, counts, MAX_OBJECTS);
}

void PoseEstimation::draw_output_on_frame(const Mat &frame)
{
	// iterate over objects
	for (int i = 0; i < object_count; i++)
	{
		// iterate over part types and draw points of existing parts
		for (int j = 0; j < objects.cols; j++)
		{
			// find part index in detected parts of this type
			int part_ind = objects.at<int>(i, j);
			if (part_ind < 0) continue;

			// extract the peak of this part
			Point2i point_unnormalized;
			point_unnormalized.x = refined_peaks[j][part_ind].x * frame.cols;
			point_unnormalized.y = refined_peaks[j][part_ind].y * frame.rows;

			// draw peak
			circle(frame, point_unnormalized, 2, Scalar(0, 0, 255), 3);
		}

		// iterate over links and draw lines of existing links
		for (int k = 0; k < NUM_LINKS; k++)
		{
			// find part type indices for this link's source and sink
			int src_part_type_ind = topology[k][2];
			int sink_part_type_ind = topology[k][3];

			// find source and sink peak indices in the extracted normalized peaks
			int src_peak_ind = objects.at<int>(i, src_part_type_ind);
			int sink_peak_ind = objects.at<int>(i, sink_part_type_ind);

			// draw link if both source and sink parts exist
			if ((src_peak_ind >= 0) && (sink_peak_ind >= 0))
			{
				Point2i src_peak;
				Point2i sink_peak;

				// find unnormalized source peak
				src_peak.x = refined_peaks[src_part_type_ind][src_peak_ind].x * frame.cols;
				src_peak.y = refined_peaks[src_part_type_ind][src_peak_ind].y * frame.rows;

				// find unnormalized sink peak
				sink_peak.x = refined_peaks[sink_part_type_ind][sink_peak_ind].x * frame.cols;
				sink_peak.y = refined_peaks[sink_part_type_ind][sink_peak_ind].y * frame.rows;

				// draw the line
				line(frame, src_peak, sink_peak, Scalar(0, 255, 0), 2);
			}
		}
	}
}
