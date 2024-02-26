#ifndef __FILTER_H__
#define __FILTER_H__

#include "yolov8/common.hpp"
#include "common.h"
#include <vector>

class FILTER
{
public:
    FILTER(FilterThreshold& filterThres);
    ~FILTER();
    void size(const std::vector<seg::Object>& objs, std::vector<seg::Object>& objs_filter);
private:
    FilterThreshold* thres;
};

#endif /* End of #ifndef __FILTER_H__ */