#ifndef __FISH_SEGMENTATION_H__
#define __FISH_SEGMENTATION_H__

#include "yolov8/yolov8-seg.h"
#include "yolov8/common.hpp"
#include "common.h"
#include "logger.h"

class FISH_SEG
{
public:
    FISH_SEG(std::string& model_path, FishSegConf& config);
    ~FISH_SEG();
    void predict(const cv::Mat& image, std::vector<seg::Object> &objs);

private:
    YOLOv8_seg* yolov8_seg;
    FishSegConf* config;
};

#endif /*__FISH_SEGMENTATION_H__ */