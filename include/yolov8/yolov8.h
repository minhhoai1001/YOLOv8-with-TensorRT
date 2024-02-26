//
// Created by ubuntu on 3/16/23.
//
#ifndef JETSON_DETECT_YOLOV8_HPP
#define JETSON_DETECT_YOLOV8_HPP
#include "fstream"
#include "common.hpp"
#include "NvInferPlugin.h"

// using namespace det;

class YOLOv8
{
public:
	explicit YOLOv8(const std::string& engine_file_path);
	~YOLOv8();

	void make_pipe(bool warmup = true);
	void copy_from_Mat(const cv::Mat& image);
	void copy_from_Mat(const cv::Mat& image, cv::Size& size);
	void letterbox(
		const cv::Mat& image,
		cv::Mat& out,
		cv::Size& size
	);
	void infer();
	void postprocess(std::vector<det::Object>& objs);
	void postprocess_on_fish(std::vector<det::Object>& objs, cv::Rect_<int>split_rect);
	static void draw_objects(
		const cv::Mat& image,
		cv::Mat& res,
		const std::vector<det::Object>& objs,
		const std::vector<std::string>& CLASS_NAMES,
		const std::vector<std::vector<unsigned int>>& COLORS
	);
	static void draw_objects_original_image(
		cv::Rect_<int> split_position,
		cv::Mat& res,
		const std::vector<det::Object>& objs,
		const std::vector<std::string>& CLASS_NAMES,
		const std::vector<std::vector<unsigned int>>& COLORS
	);
	int num_bindings;
	int num_inputs = 0;
	int num_outputs = 0;
	std::vector<det::Binding> input_bindings;
	std::vector<det::Binding> output_bindings;
	std::vector<void*> host_ptrs;
	std::vector<void*> device_ptrs;

	det::PreParam pparam;
private:
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
	cudaStream_t stream = nullptr;
	Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

};

#endif //JETSON_DETECT_YOLOV8_HPP
