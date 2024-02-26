//
// Created by ubuntu on 3/16/23.
//
#ifndef JETSON_SEGMENT_YOLOV8_SEG_HPP
#define JETSON_SEGMENT_YOLOV8_SEG_HPP
#include <fstream>
#include "common.hpp"
#include "NvInferPlugin.h"

// using namespace seg;

class YOLOv8_seg
{
public:
	explicit YOLOv8_seg(const std::string& engine_file_path);
	~YOLOv8_seg();

	void make_pipe(bool warmup = true);
	void copy_from_Mat(const cv::Mat& image);
	void copy_from_Mat(const cv::Mat& image, cv::Size& size);
	void letterbox(
		const cv::Mat& image,
		cv::Mat& out,
		cv::Size& size
	);
	void infer();
	void postprocess(
		std::vector<seg::Object>& objs,
		float score_thres = 0.25f,
		float iou_thres = 0.65f,
		int topk = 100,
		int seg_channels = 32,
		int seg_h = 160,
		int seg_w = 160
	);
	static void draw_objects(
		const cv::Mat& image,
		cv::Mat& res,
		const std::vector<seg::Object>& objs,
		const std::vector<std::string>& CLASS_NAMES,
		const std::vector<std::vector<unsigned int>>& COLORS,
		const std::vector<std::vector<unsigned int>>& MASK_COLORS
	);
	static void draw_contour(
		const cv::Mat& image,
		cv::Mat& res,
		const std::vector<seg::Object>& objs
	);
	int num_bindings;
	int num_inputs = 0;
	int num_outputs = 0;
	std::vector<seg::Binding> input_bindings;
	std::vector<seg::Binding> output_bindings;
	std::vector<void*> host_ptrs;
	std::vector<void*> device_ptrs;

	seg::PreParam pparam;
private:
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
	cudaStream_t stream = nullptr;
	Logger gLogger{ nvinfer1::ILogger::Severity::kERROR };

};

#endif //JETSON_SEGMENT_YOLOV8_SEG_HPP
