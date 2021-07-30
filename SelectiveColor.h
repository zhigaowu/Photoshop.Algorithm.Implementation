
/*
    reference URI: http://blog.pkh.me/p/22-understanding-selective-coloring-in-adobe-photoshop.html
    note: white/median/black is not implemented
*/

#ifndef _PHOTOSHOP_SELECTIVE_COLOR_
#define _PHOTOSHOP_SELECTIVE_COLOR_

#include <opencv2/opencv.hpp>

class SelectiveColor {
public:
#if OPENCV_COLOR_RGB

    static const int Component_Red = 0;
    static const int Component_Green = 1;
    static const int Component_Blue = 2;

    static const int Component_Cyan = 3;
    static const int Component_Magtenta = 4;
    static const int Component_Yellow = 5;

#else

    static const int Component_Blue = 0;
    static const int Component_Green = 1;
    static const int Component_Red = 2;

    static const int Component_Yellow = 3;
    static const int Component_Magtenta = 4;
    static const int Component_Cyan = 5;

#endif

    static const int Mode_Absolute = 0;
    static const int Mode_Relative = 1;

public:
    SelectiveColor();
    ~SelectiveColor();

    int Adjust(cv::Mat& dst, const cv::Mat& src, int mode, int component, float cyan, float magtenta, float yellow);
    int Adjust(cv::cuda::GpuMat& dst, const cv::cuda::GpuMat& src, int mode, int component, float cyan, float magtenta, float yellow, cv::cuda::Stream& stream = cv::cuda::Stream::Null());

private:
    SelectiveColor(const SelectiveColor&) = delete;
    SelectiveColor(SelectiveColor&&) = delete;
    SelectiveColor& operator=(const SelectiveColor&) = delete;
    SelectiveColor& operator=(SelectiveColor&&) = delete;
};

#endif
