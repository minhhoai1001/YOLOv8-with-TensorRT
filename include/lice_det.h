#ifndef __LICE_DETECTION_H__
#define __LICE_DETECTION_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "logger.h"
#include "yolov8/yolov8.h"
#include "yolov8/common.hpp"

class LICE_DET
{
public:
	typedef std::tuple<std::vector<cv::Mat>, std::vector<cv::Rect_<int>>> ImageTuple;

	LICE_DET(std::string& model_path, LiceDetConf& config);
	~LICE_DET();
	void predict(const cv::Mat& image, std::vector<seg::Object> &objs, std::vector<FishObject>& fish_with_lice);
	cv::Mat extractFish(const cv::Mat& image, seg::Object &obj);
	ImageTuple splitImage(cv::Mat image, int overlap);
	float calculateIoU(const cv::Rect_<float>& box1, const cv::Rect_<float>& box2);
	std::vector<det::Object> nonMaxSuppression(const std::vector<det::Object>& objs);
	void updateFishLice(seg::Object& fish, std::vector<det::Object>& lices, FishObject& fish_with_lice);
private:
    YOLOv8* yolov8_det;
    LiceDetConf* config;
	std::vector<cv::Mat> fish_imgs;
};

#endif /* End of #ifndef __LICE_COUNTING_H__ */