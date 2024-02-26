#ifndef __LICE_DETECTION_CPP__
#define __LICE_DETECTION_CPP__

#include "lice_det.h"

LICE_DET::LICE_DET(std::string& model_path, LiceDetConf& config): config(&config)
{
    LOG_INFO("Load model: %s", model_path.c_str());
    yolov8_det = new YOLOv8(model_path);
    yolov8_det->make_pipe(false);
}

LICE_DET::~LICE_DET()
{
    delete yolov8_det;
}

void LICE_DET::predict(const cv::Mat& image, std::vector<seg::Object> &objs, std::vector<FishObject>& fish_with_lice)
{
    std::vector<det::Object> lice_on_fish;

    for (auto& fish : objs)
    {
        // int x_ori = fish.rect.x;
        // int y_ori = fish.rect.y;
        std::vector<det::Object> lice_objs;

        cv::Mat fish_img = extractFish(std::ref(image), fish);
        ImageTuple split_result = splitImage(fish_img, config->overlap);
        std::vector<cv::Mat> split_images = std::get<0>(split_result);
        std::vector<cv::Rect_<int>> split_positions = std::get<1>(split_result);
        for (size_t i = 0; i < split_images.size(); i++)
        {
            cv::Mat& split_image = split_images[i];
            cv::Rect_<int>& split_position = split_positions[i];

            // split_position.x += x_ori;
            // split_position.y += y_ori;

            yolov8_det->copy_from_Mat(split_image, config->size);
            yolov8_det->infer();
            yolov8_det->postprocess_on_fish(lice_objs, split_position);
        }

        lice_on_fish = nonMaxSuppression(lice_objs);
        FishObject fish_lice;
        updateFishLice(fish, lice_objs, fish_lice);
        fish_with_lice.push_back(fish_lice);
    }
}

cv::Mat LICE_DET::extractFish(const cv::Mat& image, seg::Object &obj)
{
    cv::Mat invMask;
    cv::Mat mask = image.clone();
    cv::bitwise_not(obj.boxMask, invMask);
    mask(obj.rect).setTo(cv::Scalar(0,0,0), invMask);

    return mask(obj.rect);
}

LICE_DET::ImageTuple LICE_DET::splitImage(cv::Mat image, int overlap)
{
    int original_height = image.rows;
    int original_width = image.cols;

    // Calculate the number of splits in the horizontal and vertical directions
    int split_size = config->size.height;
    int num_splits_horizontal = (original_width - overlap) / (split_size - overlap) + 1;
    int num_splits_vertical = (original_height - overlap) / (split_size - overlap) + 1;

    std::vector<cv::Mat> split_images;
    std::vector<cv::Rect_<int>> split_positions;

    // Iterate over the splits and extract the corresponding region from the original image
    for (int split_index = 0; split_index < num_splits_horizontal * num_splits_vertical; split_index++) {
        int x_start = (split_index % num_splits_horizontal) * (split_size - overlap);
        int y_start = (split_index / num_splits_horizontal) * (split_size - overlap);

        // Calculate the split region boundaries
        int x_end = x_start + split_size;
        int y_end = y_start + split_size;

        // Adjust the split region boundaries if they exceed the original image size
        if (x_end > original_width)
            x_end = original_width;
        if (y_end > original_height)
            y_end = original_height;

        // Calculate the split region size after adjustments
        int split_width = x_end - x_start;
        int split_height = y_end - y_start;

        // Extract the split region
        cv::Rect region(x_start, y_start, split_width, split_height);
        cv::Mat split_region = image(region).clone();

        // Pad the split region with black pixels if it is smaller than 640x640
        if (split_width < split_size || split_height < split_size) {
            cv::Mat padded_region = cv::Mat::zeros(split_size, split_size, CV_8UC3);
            cv::Rect roi(0, 0, split_width, split_height);
            split_region.copyTo(padded_region(roi));
            split_images.push_back(padded_region);
        }
        else {
            split_images.push_back(split_region);
        }

        // Create a cv::Rect_<int> struct and add it to the split_positions vector
        cv::Rect_<int> position;
        position.x = x_start;
        position.y = y_start;
        position.width = split_width;
        position.height = split_height;
        split_positions.push_back(position);
    }

    return std::make_tuple(split_images, split_positions);
}

float LICE_DET::calculateIoU(const cv::Rect_<float>& box1, const cv::Rect_<float>& box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = box1.width * box1.height + box2.width * box2.height - intersectionArea;

    return intersectionArea / unionArea;
}


std::vector<det::Object> LICE_DET::nonMaxSuppression(const std::vector<det::Object>& objs) 
{
    std::vector<det::Object> selectedObjs;

    // Sort the objects based on probability in descending order
    std::vector<det::Object> sortedObjs = objs;
    std::sort(sortedObjs.begin(), sortedObjs.end(), [](const det::Object& a, const det::Object& b) {
        return a.prob > b.prob;
    });

    // Iterate over the sorted objects
    for (size_t i = 0; i < sortedObjs.size(); ++i) {
        const det::Object& currentObj = sortedObjs[i];
        if (currentObj.prob < config->score_thres){
            continue;
        } 
            
        // Check if the current object has sufficient overlap with any of the selected objects
        bool foundOverlap = false;
        for (const det::Object& selectedObj : selectedObjs) {
            float iou = calculateIoU(currentObj.rect, selectedObj.rect);

            if (iou > config->iou_thres) {
                foundOverlap = true;
                break;
            }
        }

        // If no overlap is found, add the current object to the selected objects
        if (!foundOverlap) {
            selectedObjs.push_back(currentObj);
        }
    }

    return selectedObjs;
}

void LICE_DET::updateFishLice(seg::Object& fish, std::vector<det::Object>& lices, FishObject& fish_with_lice)
{
    std::vector<LiceObject> lice_objs;
    for(auto& lice: lices)
    {   
        LiceObject lice_obj;
        lice_obj.rect   = lice.rect;
        lice_obj.label  = lice.label;
        lice_obj.prob   = lice.prob;
        lice_objs.push_back(lice_obj);
    }
    fish_with_lice.rect     = fish.rect;
    fish_with_lice.label    = fish.label;
    fish_with_lice.prob     = fish.prob;
    fish_with_lice.boxMask  = fish.boxMask;
    fish_with_lice.lice     = lice_objs;
}

// std::vector<det::Object> liceDetection(cv::Mat& image, std::vector<seg::Object>& objs, YOLOv8* yolov8)
// {
//     int x_ori, y_ori;
//     cv::Size size = cv::Size{ 640, 640 };

//     std::vector<det::Object> lice_objs;
//     std::vector<det::Object> final_result, lice_per_fish;

//     // For each segmented fish in image
//     for (auto fish : objs)
//     {
//         x_ori = fish.rect.x;
//         y_ori = fish.rect.y;

//         cv::Mat invMask;
//         cv::Mat mask = image.clone();
//         cv::bitwise_not(fish.boxMask, invMask);
//         mask(fish.rect).setTo(cv::Scalar(0,0,0), invMask);

//         cv::Mat fish_img = mask(fish.rect);

//         ImageTuple split_result = splitImage(fish_img, 60);
//         // Access the split images and split positions from the tuple
//         std::vector<cv::Mat> split_images = std::get<0>(split_result);
//         std::vector<cv::Rect_<int>> split_positions = std::get<1>(split_result);

//         // For each split of segmented image
//         for (size_t i = 0; i < split_images.size(); i++) 
//         {
//             cv::Mat& split_image = split_images[i];
//             cv::Rect_<int>& split_position = split_positions[i];

//             split_position.x += x_ori;
//             split_position.y += y_ori;

//             yolov8->copy_from_Mat(split_image, size);
//             yolov8->infer();
//             // yolov8->postprocess(lice_objs);
//             yolov8->postprocess_2(lice_objs, split_position);
//         }
//     }

//     float iou_thresh = 0.2;
//     float score_thresh = 0.45;
//     lice_per_fish = nonMaxSuppression(lice_objs, iou_thresh, score_thresh);

//     return lice_per_fish;
// }

#endif /*__LICE_DETECTION_CPP__ */