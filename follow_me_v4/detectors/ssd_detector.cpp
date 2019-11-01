#include "ssd_detector.h"
#include "ssd/sampleUffSSD.h"

std::vector<std::pair<int64_t, nvinfer1::DataType>> SSDDetector::calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

SSDDetector::~SSDDetector()
{
	destroy();
}

void SSDDetector::allocate_buffers(IExecutionContext *context)
{
	// for now set batch size to 1
	int batchSize = 1;

	// get engine
    const ICudaEngine& engine = context->getEngine();

    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int num_of_bindings = engine.getNbBindings();
    buffers.reserve(num_of_bindings);
    std::vector<std::pair<int64_t, nvinfer1::DataType>> buffersSizes = calculateBindingBufferSizes(engine, num_of_bindings, batchSize);

    for (int i = 0; i < num_of_bindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }
}

ICudaEngine* SSDDetector::load_engine()
{
    // read the model in memory
    std::stringstream trtModelStream;
    trtModelStream.seekg(0, trtModelStream.beg);
    std::ifstream cache("../detectors/ssd/ssd_float32.engine");
    assert(cache.good());
    trtModelStream << cache.rdbuf();
    cache.close();

    // calculate model size
    trtModelStream.seekg(0, std::ios::end);
    const int modelSize = trtModelStream.tellg();
    trtModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    trtModelStream.read((char*) modelMem, modelSize);

    // deserialize the TensorRT engine
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);
    runtime->destroy();
    return engine;
}

// initialization
void SSDDetector::init()
{
    // init nvidia inference plugins library
	std::cout << "SSD: Initializing NvInferPlugins" << std::endl;
    initLibNvInferPlugins(&gLogger, "");

    // load the model and create the TensorRT engine
    std::cout << "SSD: loading TensorRT engine" << std::endl;
	engine = load_engine();

	std::cout << "SSD: creating execution context" << std::endl;
	context = engine->createExecutionContext();

	// create the CUDA stream
	std::cout << "SSD: creating CUDA stream" << std::endl;
    CHECK(cudaStreamCreate(&stream));

    // allocate the buffers
    std::cout << "SSD: allocating buffers" << std::endl;
    allocate_buffers(context);
}

// detect objects and return their bounding boxes and class names
void SSDDetector::detect(Mat &image, std::vector<Rect2d> &out_bboxes, std::vector<std::string> &class_names)
{
    vector<float> detectionOut(100000);
    vector<int> keepCount(1);

    // subtract image with mean
    Mat image_processed;
    //subtract(image)

	do_inference(*context, (float*)image.data, &detectionOut[0], &keepCount[0], 1);
}

// do the inference
void SSDDetector::do_inference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
    int outputIndex1 = outputIndex0 + 1;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU, execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    context.execute(batchSize, &buffers[0]);

    // copy results to host buffers
    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));

    // wait until all operations end
    cudaStreamSynchronize(stream);
}

void SSDDetector::destroy()
{
    // release the stream
    cudaStreamDestroy(stream);

    // release the host and device buffers
    for (int i = 0; i < buffers.size(); i++)
    {
    	CHECK(cudaFree(buffers[i]));
    }
}
