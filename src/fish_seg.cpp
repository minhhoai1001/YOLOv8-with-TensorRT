#ifndef __FISH_SEGMENTATION_CPP__
#define __FISH_SEGMENTATION_CPP__

#include "fish_seg.h"

FISH_SEG::FISH_SEG(std::string& model_path, FishSegConf& config): config(&config)
{
    LOG_INFO("Load model: %s", model_path.c_str());
    yolov8_seg = new YOLOv8_seg(model_path);
    yolov8_seg->make_pipe(false);
}

FISH_SEG::~FISH_SEG()
{
    delete yolov8_seg;
}

void FISH_SEG::predict(const cv::Mat& image, std::vector<seg::Object> &objs)
{
    objs.clear();
    yolov8_seg->copy_from_Mat(image, config->size);
    yolov8_seg->infer();
    yolov8_seg->postprocess(objs, config->score_thres, config->iou_thres, config->topk, config->seg_channels, config->seg_h, config->seg_w);
}

#endif /*__FISH_SEGMENTATION_CPP__ */