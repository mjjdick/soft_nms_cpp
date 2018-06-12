#ifndef PROJECT_SOFT_NMS_H
#define PROJECT_SOFT_NMS_H

#include <opencv2/opencv.hpp>

struct BboxWithScore
{
    float tx,ty,bx,by,rz,score;
    BboxWithScore()
    {
        tx = 0.;
        ty = 0.;
        bx = 0.;
        by = 0.;
        rz = 0.;
        score = 0.;
    }
};

namespace Robosense
{
    //method 0 : origin nms, 1: liner, 2: gaussian
    void softNmsWithNoRotation(std::vector<BboxWithScore>& bboxes,const int& method = 0,
                               const float& sigma = 0.5,const float& iou_thre = 0.3,const float& threshold = 0.01);

    void softNmsWithRotation(std::vector<BboxWithScore>& bboxes,const int& method = 0,
                             const float& sigma = 0.5,const float& iou_thre = 0.3,const float& threshold = 0.01);

    float calIOUWithNoRotation(const BboxWithScore& bbox1,const BboxWithScore& bbox2);
    float calIOUWithRotation(const BboxWithScore& bbox1,const BboxWithScore& bbox2);
}

#endif //PROJECT_SOFT_NMS_H
