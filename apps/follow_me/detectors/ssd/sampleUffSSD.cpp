#include "sampleUffSSD.h"

#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

using namespace nvinfer1;

const char* INPUT_BLOB_NAME = "Input";

static samplesCommon::Args args;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "sample_uff_ssd: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)

static constexpr int OUTPUT_CLS_SIZE = 37;
static constexpr int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const char* OUTPUT_BLOB_NAME0 = "NMS";
const char* CLASSIFIER_CLS_SOFTMAX_NAME = "random_name";
//INT8 Calibration, currently set to calibrate over 500 images
static constexpr int CAL_BATCH_SIZE = 50;
static constexpr int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

// Concat layers
// mbox_priorbox, mbox_loc, mbox_conf
const int concatAxis[2] = {1, 1};
const bool ignoreBatch[2] = {false, false};

DetectionOutputParameters detectionOutputParam{true, false, 0, OUTPUT_CLS_SIZE, 100, 100, 0.5, 0.6, CodeTypeSSD::TF_CENTER, {1, 2, 0}, true, true};

// Visualization
const float visualizeThreshold = 0.4;

void printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    assert(samplesCommon::getElementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * samplesCommon::getElementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpyAsync(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = std::distance(outputs, std::max_element(outputs, outputs + eltCount));

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
            std::cout << "***";
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/ssd/",
                                  "data/ssd/VOC2007/",
                                  "data/ssd/VOC2007/PPMImages/",
                                  "data/samples/ssd/",
                                  "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

void populateTFInputData(float* data)
{

    auto fileName = locateFile("inp_bus.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        istringstream iss(line);
        float num;
        iss >> num;
        data[id++] = num;
    }

    return;
}

void populateClassLabels(std::string (&CLASSES)[OUTPUT_CLS_SIZE])
{

    // auto fileName = locateFile("ssd_coco_labels.txt");
    auto fileName = locateFile("sample_labels.txt");
    std::ifstream labelFile(fileName);
    string line;
    int id = 0;
    while (getline(labelFile, line))
    {
        CLASSES[id++] = line;
    }

    return;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

ICudaEngine* ssd_loadModelAndCreateEngine()
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

void doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();
    std::cout << nbBindings << " Binding" << std::endl;
    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
        std::cout << "Allocating buffer sizes for binding index: " << i << " of size : " ;
        std::cout << bufferSizesOutput.first << " * " << samplesCommon::getElementSize(bufferSizesOutput.second);
        std::cout << " B" << std::endl;
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = outputIndex0 + 1; //engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

    float t_average = 0;
    float t_iter = 0;
    int iterations = 10;
    int avgRuns = 100;

    for (int i=0; i < iterations; i++)
    {
	t_average = 0;
	for (int j=0; j < avgRuns; j++)
	{
            auto t_start = std::chrono::high_resolution_clock::now();
            context.execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            t_average += total;

	}
    	t_average /= avgRuns;
        std::cout << "Time taken for inference per run is " << t_average << " ms." << std::endl;
	t_iter += t_average;
    }
    t_iter /= iterations;
    std::cout << "Average time spent per iteration is " << t_iter << " ms." << std::endl;

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
#ifdef SSD_INT8_DEBUG
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                    buffers[bindingIdx]);
#endif
    }

    CHECK(cudaMemcpyAsync(detectionOut, buffers[outputIndex0], batchSize * detectionOutputParam.keepTopK * 7 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(keepCount, buffers[outputIndex1], batchSize * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    std::cout << "Time taken for inference is " << t_average << " ms." << std::endl;
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}

/*void ssd_doInference(IExecutionContext& context, float* inputData, float* detectionOut, int* keepCount, int batchSize)
{
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
        outputIndex1 = outputIndex0 + 1;

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

/*void ssd_allocate_buffers()
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    int nbBindings = engine.getNbBindings();
    std::vector<void*> buffers(nbBindings);
    std::vector<std::pair<int64_t, DataType>> buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    for (int i = 0; i < nbBindings; ++i)
    {
        auto bufferSizesOutput = buffersSizes[i];
        buffers[i] = samplesCommon::safeCudaMalloc(bufferSizesOutput.first * samplesCommon::getElementSize(bufferSizesOutput.second));
    }
}

void ssd_destroy()
{
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
}*/

//////////////////////////////////////////////////////////////////////////////////////////////////////
namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
}

// Flattens all input tensors and concats their flattened version together
// along the major non-batch dimension, i.e axis = 1
class FlattenConcat : public IPluginV2
{
public:
    // Ordinary ctor, plugin not yet configured for particular inputs/output
    FlattenConcat() {}

    // Ctor for clone()
    FlattenConcat(const int* flattenedInputSize, int numInputs, int flattenedOutputSize)
        : mFlattenedOutputSize(flattenedOutputSize)
    {
        for (int i = 0; i < numInputs; ++i)
            mFlattenedInputSize.push_back(flattenedInputSize[i]);
    }

    // Ctor for loading from serialized byte array
    FlattenConcat(const void* data, size_t length)
    {
        const char* d = reinterpret_cast<const char*>(data);
        const char* a = d;

        size_t numInputs = read<size_t>(d);
        for (size_t i = 0; i < numInputs; ++i)
        {
            mFlattenedInputSize.push_back(read<int>(d));
        }
        mFlattenedOutputSize = read<int>(d);

        assert(d == a + length);
    }

    int getNbOutputs() const override
    {
        // We always return one output
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        // At least one input
        assert(nbInputDims >= 1);
        // We only have one output, so it doesn't
        // make sense to check index != 0
        assert(index == 0);

        size_t flattenedOutputSize = 0;
        int inputVolume = 0;

        for (int i = 0; i < nbInputDims; ++i)
        {
            // We only support NCHW. And inputs Dims are without batch num.
            assert(inputs[i].nbDims == 3);

            inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            flattenedOutputSize += inputVolume;
        }

        return DimsCHW(flattenedOutputSize, 1, 1);
    }

    int initialize() override
    {
        // Called on engine initialization, we initialize cuBLAS library here,
        // since we'll be using it for inference
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        // Called on engine destruction, we destroy cuBLAS data structures,
        // which were created in initialize()
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int maxBatchSize) const override
    {
        // The operation is done in place, it doesn't use GPU memory
        return 0;
    }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        // Does the actual concat of inputs, which is just
        // copying all inputs bytes to output byte array
        size_t inputOffset = 0;
        float* output = reinterpret_cast<float*>(outputs[0]);

        for (size_t i = 0; i < mFlattenedInputSize.size(); ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            {
                CHECK(cublasScopy(mCublas, mFlattenedInputSize[i],
                                  input + batchIdx * mFlattenedInputSize[i], 1,
                                  output + (batchIdx * mFlattenedOutputSize + inputOffset), 1));
            }
            inputOffset += mFlattenedInputSize[i];
        }

        return 0;
    }

    size_t getSerializationSize() const override
    {
        // Returns FlattenConcat plugin serialization size
        size_t size = sizeof(mFlattenedInputSize[0]) * mFlattenedInputSize.size()
            + sizeof(mFlattenedOutputSize)
            + sizeof(size_t); // For serializing mFlattenedInputSize vector size
        return size;
    }

    void serialize(void* buffer) const override
    {
        // Serializes FlattenConcat plugin into byte array

        // Cast buffer to char* and save its beginning to a,
        // (since value of d will be changed during write)
        char* d = reinterpret_cast<char*>(buffer);
        char* a = d;

        size_t numInputs = mFlattenedInputSize.size();

        // Write FlattenConcat fields into buffer
        write(d, numInputs);
        for (size_t i = 0; i < numInputs; ++i)
        {
            write(d, mFlattenedInputSize[i]);
        }
        write(d, mFlattenedOutputSize);

        // Sanity check - checks if d is offset
        // from a by exactly the size of serialized plugin
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        // We only support one output
        assert(nbOutputs == 1);

        // Reset plugin private data structures
        mFlattenedInputSize.clear();
        mFlattenedOutputSize = 0;

        // For each input we save its size, we also validate it
        for (int i = 0; i < nbInputs; ++i)
        {
            int inputVolume = 0;

            // We only support NCHW. And inputs Dims are without batch num.
            assert(inputs[i].nbDims == 3);

            // All inputs dimensions along non concat axis should be same
            for (size_t dim = 1; dim < 3; dim++)
            {
                assert(inputs[i].d[dim] == inputs[0].d[dim]);
            }

            // Size of flattened input
            inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mFlattenedInputSize.push_back(inputVolume);
            mFlattenedOutputSize += mFlattenedInputSize[i];
        }
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    }

    const char* getPluginType() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    void destroy() override {}

    IPluginV2* clone() const override
    {
        return new FlattenConcat(mFlattenedInputSize.data(), mFlattenedInputSize.size(), mFlattenedOutputSize);
    }

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mPluginNamespace.c_str();
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    // Number of elements in each plugin input, flattened
    std::vector<int> mFlattenedInputSize;
    // Number of elements in output, flattened
    int mFlattenedOutputSize{0};
    // cuBLAS library handle
    cublasHandle_t mCublas;
    // We're not using TensorRT namespaces in
    // this sample, so it's just an empty string
    std::string mPluginNamespace = "";
};

// PluginCreator boilerplate code for FlattenConcat plugin
class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = 0;
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        return new FlattenConcat();
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {

        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mPluginNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);


//int main(int argc, char* argv[])
//{
//    // Parse command-line arguments.
//    samplesCommon::parseArgs(args, argc, argv);
//
//    initLibNvInferPlugins(&gLogger, "");
//    auto fileName = locateFile("sample_unpruned_mobilenet_v2.uff");
//    std::cout << fileName << std::endl;
//
//    // changed BATCH SIZE back to 2
//    const int N = 1;
//    //const int N = 2;
//
//    auto parser = createUffParser();
//
//    // BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
//
//    std::cout << "Registering UFF model" << std::endl;
//    parser->registerInput("Input", DimsCHW(INPUT_C, INPUT_H, INPUT_W), UffInputOrder::kNCHW);
//    std::cout << "Registered Input" << std::endl;
//    // parser->registerOutput("ssd_resnet18keras_feature_extractor/model/activation_16/Relu");
//    parser->registerOutput("NMS");
//    std::cout << "Registered output NMS" << std::endl;
//
//    IHostMemory* trtModelStream{nullptr};
//
//    // Int8EntropyCalibrator calibrator(calibrationStream, FIRST_CAL_BATCH, "CalibrationTableSSD");
//
//    std::cout << "Creating engine" << std::endl;
//    ICudaEngine* tmpEngine = loadModelAndCreateEngine(fileName.c_str(), N, parser, trtModelStream);
//    assert(tmpEngine != nullptr);
//    assert(trtModelStream != nullptr);
//    tmpEngine->destroy();
//    std::cout << "Created engine" << std::endl;
//    // Read a random sample image.
//    srand(unsigned(time(nullptr)));
//    // Available images.
//    std::vector<std::string> imageList = {"dog.ppm", "image1.ppm"};
//    std::vector<samplesCommon::PPM<INPUT_C, INPUT_H, INPUT_W>> ppms(N);
//
//    assert(ppms.size() <= imageList.size());
//    std::cout << " Num batches  " << N << std::endl;
//    for (int i = 0; i < N; ++i)
//    {
//        readPPMFile(locateFile(imageList[i%2]), ppms[i]);
//    }
//
//    vector<float> data(N * INPUT_C * INPUT_H * INPUT_W);
//
//    // for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
//    // {
//    //     for (int c = 0; c < INPUT_C; ++c)
//    //     {
//    //         for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
//    //         {
//    //             data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * INPUT_C + c]) - 1.0;
//    //         }
//    //     }
//    // }
//
//    // RGB preprocessing based on keras pipeline
//    for (int i = 0, volImg = INPUT_C * INPUT_H * INPUT_W; i < N; ++i)
//    {
//      for (unsigned j = 0, volChl = INPUT_H * INPUT_W; j < volChl; ++j)
//      {
//        data[i * volImg + 0 * volChl + j] = float(ppms[i].buffer[j * INPUT_C + 0]) - 123.68;
//        data[i * volImg + 1 * volChl + j] = float(ppms[i].buffer[j * INPUT_C + 1]) - 116.779;
//        data[i * volImg + 2 * volChl + j] = float(ppms[i].buffer[j * INPUT_C + 2]) - 103.939;
//      }
//    }
//    std::cout << " Data Size  " << data.size() << std::endl;
//
//    // Deserialize the engine.
//    std::cout << "*** deserializing" << std::endl;
//    IRuntime* runtime = createInferRuntime(gLogger);
//    assert(runtime != nullptr);
//    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
//    assert(engine != nullptr);
//    trtModelStream->destroy();
//    IExecutionContext* context = engine->createExecutionContext();
//    assert(context != nullptr);
//    //SimpleProfiler profiler("layerTime");
//    //context->setProfiler(&profiler);
//    // Host memory for outputs.
//    vector<float> detectionOut(100000); //(N * detectionOutputParam.keepTopK * 7);
//    vector<int> keepCount(N);
//
//    // Run inference.
//    doInference(*context, &data[0], &detectionOut[0], &keepCount[0], N);
//    cout << " KeepCount " << keepCount[0] << "\n";
//    //cout << profiler;
//    /***********************************************************************************
//    std::string CLASSES[OUTPUT_CLS_SIZE];
//
//    populateClassLabels(CLASSES);
//
//    for (int p = 0; p < N; ++p)
//    {
//        for (int i = 0; i < keepCount[p]; ++i)
//        {
//            float* det = &detectionOut[0] + (p * detectionOutputParam.keepTopK + i) * 7;
//            if (det[2] < visualizeThreshold)
//                continue;
//
//            // Output format for each detection is stored in the below order
//            // [image_id, label, confidence, xmin, ymin, xmax, ymax]
//            assert((int) det[1] < OUTPUT_CLS_SIZE);
//            std::string storeName = CLASSES[(int) det[1]] + "-" + std::to_string(det[2]) + ".ppm";
//
//            printf("Detected %s in the image %d (%s) with confidence %f%% and coordinates (%f,%f),(%f,%f).\nResult stored in %s.\n", CLASSES[(int) det[1]].c_str(), int(det[0]), ppms[p].fileName.c_str(), det[2] * 100.f, det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H, storeName.c_str());
//
//            samplesCommon::writePPMFileWithBBox(storeName, ppms[p], {det[3] * INPUT_W, det[4] * INPUT_H, det[5] * INPUT_W, det[6] * INPUT_H});
//        }
//    }
//    ************************************************************************************/
//    // Destroy the engine.
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//
//    return EXIT_SUCCESS;
//}
