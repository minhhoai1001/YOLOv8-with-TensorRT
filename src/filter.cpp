#ifndef __FILTER_CPP__
#define __FILTER_CPP__

#include "filter.h"

FILTER::FILTER(FilterThreshold& filterThres): thres(&filterThres)
{

}

FILTER::~FILTER()
{
    
}

void FILTER::size(const std::vector<seg::Object>& objs, std::vector<seg::Object>& objs_filter)
{
    objs_filter.clear();
    
    for (auto& obj : objs)
	{
		int idx = obj.label;

		if(idx == 0) //Side-Side view
		{
			if(obj.rect.width > thres->width_low && 
                (obj.rect.height < thres->height_high || obj.rect.height > thres->height_low))
			{
				float wh_ratio = obj.rect.width/obj.rect.height;
				if(wh_ratio >= thres->wh_ratio)
				{
					seg::Object passed = obj;
					objs_filter.push_back(passed);
				}
			}
		}
		else if(idx == 1) //Top-Down view
		{
			if(obj.rect.width > thres->width_low || obj.rect.height > 1620)
			{
				float wh_ratio = obj.rect.width/obj.rect.height;
				if(wh_ratio >= 1.5 || wh_ratio < 0)
				{
					seg::Object passed = obj;
					objs_filter.push_back(passed);
				}
			}
		}
	}
}

#endif /* End of #ifndef __FILTER_CPP__ */