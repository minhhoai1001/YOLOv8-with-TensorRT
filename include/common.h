#ifndef __COMMON_H__
#define __COMMON_H__

#include <iostream>
#include <string>

struct FishSegConf 
{
    cv::Size size;
    int topk;
	int seg_h;
	int seg_w;
	int seg_channels;
	float score_thres;
	float iou_thres;
};

struct LiceDetConf 
{
    cv::Size size;
    int overlap;
	float score_thres;
	float iou_thres;
};

struct FilterThreshold
{
	int width_high;
	int width_low;
	int height_high;
	int height_low;
	float wh_ratio;
};
	
struct LiceObject
{
	cv::Rect_<float> rect;
	int label = 0;
	float prob = 0.0;
};

struct FishObject
{
	cv::Rect_<float> rect;
	int label = 0;
	float prob = 0.0;
	cv::Mat boxMask;
	std::vector<LiceObject> lice;
};

const std::vector<std::string> CLASS_NAMES = {
	"female_w_egg", "female_wo_egg", "mobile"
};

const std::vector<std::vector<unsigned int>> COLORS = {
	{ 0, 114, 189 }, { 255, 157, 151 }, { 100, 115, 255 }
};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
	{ 255, 56, 56 }, { 255, 157, 151 }, { 255, 112, 31 },
	{ 255, 178, 29 }, { 207, 210, 49 }, { 72, 249, 10 },
	{ 146, 204, 23 }, { 61, 219, 134 }, { 26, 147, 52 },
	{ 0, 212, 187 }, { 44, 153, 168 }, { 0, 194, 255 },
	{ 52, 69, 147 }, { 100, 115, 255 }, { 0, 24, 236 },
	{ 132, 56, 255 }, { 82, 0, 133 }, { 203, 56, 255 },
	{ 255, 149, 200 }, { 255, 55, 199 }
};

// struct LiceObject 
// {
// 	int label = 0;
// 	float prob = 0.0;
// 	cv::Mat image;
// };

#endif