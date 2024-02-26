#include "lice_det.h"
#include "fish_seg.h"
#include "filter.h"

#include <yaml-cpp/yaml.h>
#include <atomic>
#include <csignal>
#include <opencv2/core/utils/logger.hpp>

std::atomic<bool> bstop_signal(false);

// Signal handler function
void signalHandler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM) 
    {
        LOG_INFO("Ctrl+C signal received. Exiting...");
        bstop_signal = true;
    }
}

int main(int argc, char *argv[]) 
{
    if (argc < 2)
    {
        std::cout << "Please run with file name \n";
        return -1;
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // 0. Signal handler
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    FishSegConf fishSegConf;
    fishSegConf.size            = cv::Size{640, 640};
    fishSegConf.topk            = 100;
    fishSegConf.seg_h           = 160;
    fishSegConf.seg_w           = 160;
    fishSegConf.seg_channels    = 32;
    fishSegConf.score_thres     = 0.8;
    fishSegConf.iou_thres       = 0.65;

    LiceDetConf liceDetConf;
    liceDetConf.size            = cv::Size{640, 640};
    liceDetConf.overlap         = 60;
    liceDetConf.score_thres     = 0.6;
    liceDetConf.iou_thres       = 0.2;

    FilterThreshold filterThres;
    filterThres.width_low       = 3840/2;
    filterThres.height_high     = 2160/2;
    filterThres.height_low      = 2160/4;
    filterThres.wh_ratio        = 2.8;

    std::string fish_model = "./models/fish-seg-yolov8l.engine";
    std::unique_ptr<FISH_SEG> fish_seg = std::make_unique<FISH_SEG>(fish_model, fishSegConf);

    std::string lice_model = "./models/lice-det-yolov8l.engine";
    std::unique_ptr<LICE_DET> lice_det = std::make_unique<LICE_DET>(lice_model, liceDetConf);

    std::unique_ptr<FILTER> filter = std::make_unique<FILTER>(filterThres);

    LOG_INFO("Lice detect on video: %s", argv[1]);
    std::string video_file_path = argv[1];
    
    cv::VideoWriter writer("detect-lice.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 20.0, cv::Size(3840, 2160));
    std::string pipeline = "filesrc location=" + video_file_path +
                           " ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw(memory:NVMM), format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) 
    {
        LOG_ERROR("Error opening H265");
        cap.release();
        writer.release();
        return 1;
    }

    while (!bstop_signal) 
    {
        std::vector<seg::Object> objs;
        std::vector<seg::Object> objs_filter;
        std::vector<FishObject> fish_with_lice;

        cv::Mat frame, res;
        if (!cap.read(frame)) {
            // End of video, break out of the loop
            break;
        }
        fish_seg->predict(std::ref(frame), objs);
        filter->size(objs, objs_filter);
        lice_det->predict(std::ref(frame), objs_filter, fish_with_lice);

        if (fish_with_lice.size()>0)
        {
            for(auto& fish: fish_with_lice)
            {
                cv::rectangle(frame, fish.rect, cv::Scalar(0, 255, 0), 4);
                if(fish.lice.size()==0) continue;
                LOG_INFO("have lice");
                for(auto& lice : fish.lice)
                {
                    float x_start = lice.rect.x + fish.rect.x;
                    float y_start = lice.rect.y + fish.rect.y;
                    cv::Rect point(x_start, y_start, lice.rect.width, lice.rect.height);
                    cv::rectangle(frame, point, cv::Scalar(0, 255, 255), 4);
                }
            }
            static int cnt = 0;
            std::string name = std::to_string(cnt)+".jpg";
            cv::imwrite(name, frame);
            cnt++;
        }
    }

    cap.release();
    writer.release();

    return 0;
}