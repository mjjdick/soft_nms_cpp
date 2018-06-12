#include "soft_nms/soft_nms.h"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>
BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)
typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float> > Polygon;

namespace Robosense
{
    Polygon toPolygon(const BboxWithScore& bbox)
    {
        cv::Point2f min_point,max_point;
        min_point.x = -bbox.bx / 2.;min_point.y = -bbox.by / 2.;
        max_point.x = bbox.bx / 2.;max_point.y = bbox.by / 2.;
        std::vector<cv::Point2f> pts(4);
        pts[0].x = min_point.x;pts[0].y = min_point.y;
        pts[1].x = min_point.x;pts[1].y = max_point.y;
        pts[2].x = max_point.x;pts[2].y = max_point.y;
        pts[3].x = max_point.x;pts[3].y = min_point.y;

        std::vector<cv::Point2f> ptd(4,cv::Point2f(0.,0.));
        float polygon_pts[5][2];
        for (int j = 0; j < 4; ++j)
        {
            ptd[j].x = pts[j].x * cos(-bbox.rz) + pts[j].y * sin(-bbox.rz);
            ptd[j].y = -pts[j].x * sin(-bbox.rz) + pts[j].y * cos(-bbox.rz);

            polygon_pts[j][0] = (ptd[j].x) + bbox.tx;
            polygon_pts[j][1] = (ptd[j].y) + bbox.ty;
        }
        polygon_pts[4][0] = polygon_pts[0][0];
        polygon_pts[4][1] = polygon_pts[0][1];
        Polygon result;
        boost::geometry::append(result,polygon_pts);
        boost::geometry::correct(result);
        return result;
    }

    void softNmsWithNoRotation(std::vector<BboxWithScore>& bboxes,const int& method,
                               const float& sigma,const float& iou_thre,const float& threshold)
    {
        if (bboxes.empty())
        {
            return;
        }

        int N = bboxes.size();
        float max_score,max_pos,cur_pos,weight;
        BboxWithScore tmp_bbox,index_bbox;
        for (int i = 0; i < N; ++i)
        {
            max_score = bboxes[i].score;
            max_pos = i;
            tmp_bbox = bboxes[i];
            cur_pos = i + 1;

            //get max bbox
            while (cur_pos < N)
            {
                if (max_score < bboxes[cur_pos].score)
                {
                    max_score = bboxes[cur_pos].score;
                    max_pos = cur_pos;
                }
                cur_pos ++;
            }

            //add max bbox as a detection
            bboxes[i] = bboxes[max_pos];

            //swap i th box with position of max box
            bboxes[max_pos] = tmp_bbox;
            tmp_bbox = bboxes[i];

            cur_pos = i + 1;

            while (cur_pos < N)
            {
                index_bbox = bboxes[cur_pos];

                float area = index_bbox.bx * index_bbox.by;
                float iou = calIOUWithNoRotation(tmp_bbox,index_bbox);
                if (iou <= 0)
                {
                    cur_pos++;
                    continue;
                }
                iou /= area;
                if (method == 1) // linear
                {
                    if (iou > iou_thre)
                    {
                        weight = 1 - iou;
                    } else
                    {
                        weight = 1;
                    }
                }else if (method == 2) // gaussian
                {
                    weight = exp(-(iou * iou) / sigma);
                }else // original NMS
                {
                    if (iou > iou_thre)
                    {
                        weight = 0;
                    }else
                    {
                        weight = 1;
                    }
                }
                bboxes[cur_pos].score *= weight;
                if (bboxes[cur_pos].score <= threshold)
                {
                    bboxes[cur_pos] = bboxes[N - 1];
                    N --;
                    cur_pos = cur_pos - 1;
                }
                cur_pos++;
            }
        }

        bboxes.resize(N);
    }

    void softNmsWithRotation(std::vector<BboxWithScore>& bboxes,const int& method,
                             const float& sigma,const float& iou_thre,const float& threshold)
    {
        if (bboxes.empty())
        {
            return;
        }

        int N = bboxes.size();
        float max_score,max_pos,cur_pos,weight;
        BboxWithScore tmp_bbox,index_bbox;
        for (int i = 0; i < N; ++i)
        {
            max_score = bboxes[i].score;
            max_pos = i;
            tmp_bbox = bboxes[i];
            cur_pos = i + 1;

            //get max bbox
            while (cur_pos < N)
            {
                if (max_score < bboxes[cur_pos].score)
                {
                    max_score = bboxes[cur_pos].score;
                    max_pos = cur_pos;
                }
                cur_pos ++;
            }

            //add max bbox as a detection
            bboxes[i] = bboxes[max_pos];

            //swap i th box with position of max box
            bboxes[max_pos] = tmp_bbox;
            tmp_bbox = bboxes[i];

            cur_pos = i + 1;

            while (cur_pos < N)
            {
                index_bbox = bboxes[cur_pos];

                float iou = calIOUWithRotation(tmp_bbox,index_bbox);
                if (iou <= 0)
                {
                    cur_pos++;
                    continue;
                }
                if (method == 1) // linear
                {
                    if (iou > iou_thre)
                    {
                        weight = 1 - iou;
                    } else
                    {
                        weight = 1;
                    }
                }else if (method == 2) // gaussian
                {
                    weight = exp(-(iou * iou) / sigma);
                }else // original NMS
                {
                    if (iou > iou_thre)
                    {
                        weight = 0;
                    }else
                    {
                        weight = 1;
                    }
                }
                bboxes[cur_pos].score *= weight;
                if (bboxes[cur_pos].score <= threshold)
                {
                    bboxes[cur_pos] = bboxes[N - 1];
                    N --;
                    cur_pos = cur_pos - 1;
                }
                cur_pos++;
            }
        }

        bboxes.resize(N);
    }

    float calIOUWithRotation(const BboxWithScore& bbox1,const BboxWithScore& bbox2)
    {
        Polygon polygon_bbox1,polygon_bbox2;
        polygon_bbox1 = toPolygon(bbox1);
        polygon_bbox2 = toPolygon(bbox2);

        std::vector<Polygon> in, un;
        boost::geometry::intersection(polygon_bbox1, polygon_bbox2, in);
        boost::geometry::union_(polygon_bbox1, polygon_bbox2, un);
        float inter_area = in.empty() ? 0. : boost::geometry::area(in.front());
        float union_area = boost::geometry::area(un.front());
        return inter_area / union_area;
    }

    float calIOUWithNoRotation(const BboxWithScore& bbox1,const BboxWithScore& bbox2)
    {
        float iw = (std::min(bbox1.tx + bbox1.bx / 2.,bbox2.tx + bbox2.bx / 2.) -
                    std::max(bbox1.tx - bbox1.bx / 2.,bbox2.tx - bbox2.bx / 2.));
        if (iw < 0)
        {
            return 0.;
        }

        float ih = (std::min(bbox1.ty + bbox1.by / 2.,bbox2.ty + bbox2.by / 2.) -
                    std::max(bbox1.ty - bbox1.by / 2.,bbox2.ty - bbox2.by / 2.));

        if (ih < 0)
        {
            return 0.;
        }

        return iw * ih;
    }
}

