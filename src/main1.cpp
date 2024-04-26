#include <iostream>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"


#include "NvInfer.h"
#include "NvOnnxParser.h"


#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

struct SampleYolov5Preprocessing
{
    // Preprocessing values are available here:
    // https://github.com/onnx/models/tree/master/models/image_classification/resnet
    std::vector<int> inputDims{1, 3, 224, 224};
};

struct SampleYolov5Params{

    bool verbose{false};
    bool writeNetworkTensors{false};
    int dlaCore{-1};

    SampleYolov5Preprocessing mPreproc;
    // 模型文件名
    std::string modelFileName;
    // 数据路径
    std::vector<std::string> dataDirs;
    
    std::string dynamicRangeFileName;
    
    std::string imageFileName;
    
    std::string referenceFileName;
    
    std::string networkTensorsFileName;
};


// int main(){
//     std::cout << "Hello, from yolov5inferm!\n";
//     return 0;
// }

class SampleYolov5{
    private:
    // 函数模板
    template <typename T>
    // 智能指针，防止内存泄露，第二个参数代表自定义的删除器
    using UniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    UniquePtr<nvinfer1::IRuntime> mRuntime{}; //!< The TensorRT Runtime used to deserialize the engine.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> mInOut; //!< Input and output mapping of the network

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network


    public:
    SampleYolov5(const SampleYolov5Params& params)
    : mParams(params)
    {        
    }

    // 传入多种参数的容器
    SampleYolov5Params mParams;

    //!
    //! 建立网络engine  包含
    //!
    bool build();
    

    bool constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
    UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvinfer1::IBuilderConfig>& config,
    UniquePtr<nvonnxparser::IParser>& parser);

    bool infer();

    bool prepareInput(const samplesCommon::BufferManager& buffers);

    bool verifyOutput(const samplesCommon::BufferManager& buffers) const;
};

bool SampleYolov5::build(){

    // 1 创建builder
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder){
        // 官方log接口，报错日志
        sample::gLogError << "Unable to create builder object." << std::endl;
        return false;
    }

    // 把kEXPLICIT_BATCH转换为uint32_t类型后的值， uint32_t是用typedef 定义的32位无符号数， 把这个转换后的值作为左移的位数传给1U作为标志位
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // 创建网络的版本2的函数
    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){
        // 官方log接口
        sample::gLogError << "Unable to create network object." << mParams.referenceFileName << std::endl;
        return false;
    }

    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        // 官方log接口
        sample::gLogError << "Unable to create config object." << mParams.referenceFileName << std::endl;
        return false;
    }

    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if(!parser){
        // 官方log接口，报错日志
        sample::gLogError << "Unable to create parser object." << std::endl;
        return false;
    }

    auto parsed = parser->parseFromFile(mParams.modelFileName.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if(!parsed){
        // 官方log接口
        sample::gLogError << "Unable to parse ONNX model file:" << mParams.modelFileName << std::endl;
        return false;
    }
    // auto constructed = constructNetwork(builder, network, config, parser);
    // if(!constructed){
    //     // 官方log接口
    //     sample::gLogError << "Unable to parse ONNX model file:" << mParams.modelFileName << std::endl;
    //     return false;
    // }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};

    if(!plan){
        sample::gLogError << "Unable to build serialized plan." << std::endl;
        return false;
    }

    mRuntime = UniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if(!mRuntime){
        sample::gLogError << "Unable to create runtime." << std::endl;
        return false;
    }


    // 建立tensorrt engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if(!mEngine){
        sample::gLogError << "Unable to build cuda engine." << std::endl;
        return false;
    }

    // derive input/output dims from engine bindings
    const int inputIndex = mEngine.get()->getBindingIndex(mInOut["input"].c_str());
    mInputDims = mEngine.get()->getBindingDimensions(inputIndex);

    const int outputIndex = mEngine.get()->getBindingIndex(mInOut["output"].c_str());
    mOutputDims = mEngine.get()->getBindingDimensions(outputIndex);
    return true;
}

bool SampleYolov5::infer(){
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if(!context){
        return false;
    }

    // Read the input data into the managed buffers
    // There should be just 1 input tensor

    if (!prepareInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

     // Asynchronously enqueue the inference work
    if (!context->enqueueV2(buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    CHECK(cudaStreamSynchronize(stream));

    // Release stream
    CHECK(cudaStreamDestroy(stream));

    return verifyOutput(buffers);

}

//!
//! \brief Preprocess inputs and allocate host/device input buffers
//!
bool SampleYolov5::prepareInput(const samplesCommon::BufferManager& buffers)
{
    if (samplesCommon::toLower(samplesCommon::getFileType(mParams.imageFileName)).compare("ppm") != 0)
    {
        sample::gLogError << "Wrong format: " << mParams.imageFileName << " is not a ppm file." << std::endl;
        return false;
    }

    int channels = mParams.mPreproc.inputDims.at(1);
    int height = mParams.mPreproc.inputDims.at(2);
    int width = mParams.mPreproc.inputDims.at(3);
    int max{0};
    std::string magic;

    std::vector<uint8_t> fileData(channels * height * width);

    std::ifstream infile(mParams.imageFileName, std::ifstream::binary);
    ASSERT(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> magic >> width >> height >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(fileData.data()), width * height * channels);

    uint8_t* hostInputBuffer = static_cast<uint8_t*>(buffers.getHostBuffer(mInOut["input"]));

    // Convert HWC to CHW and Normalize
    for (int c = 0; c < channels; ++c)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                int dstIdx = c * height * width + h * width + w;
                int srcIdx = h * width * channels + w * channels + c;
                hostInputBuffer[dstIdx] = fileData[srcIdx];
            }
        }
    }
    return true;
}

bool SampleYolov5::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
    UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvinfer1::IBuilderConfig>& config,
    UniquePtr<nvonnxparser::IParser>& parser){

    auto parsed = parser->parseFromFile(mParams.modelFileName.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if(!parsed){
        sample::gLogError << "Unable to parse ONNX model file: " << mParams.modelFileName << std::endl;
        return false;
    }

    return true;
}

bool SampleYolov5::verifyOutput(const samplesCommon::BufferManager& buffers) const
{
    // copy output host buffer data for further processing
    const float* probPtr = static_cast<const float*>(buffers.getHostBuffer(mInOut.at("output")));
    std::vector<float> output(probPtr, probPtr + mOutputDims.d[1]);

    std::cout << "output: ";
    std::copy(output.begin(), output.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    return true;
}