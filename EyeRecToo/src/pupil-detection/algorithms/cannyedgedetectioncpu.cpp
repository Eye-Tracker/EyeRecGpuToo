#include "cannyedgedetectioncpu.h"


void CannyEdgeDetectionCPU::edgetrace(cv::Mat *strong, cv::Mat *weak, cv::Mat *check) {
    int pic_x=strong->cols;
    int pic_y=strong->rows;

    int lines[MAX_LINE];
    int lines_idx = 0;

    int idx = 0;

    for(int i = 1; i < pic_y - 1; i++){
        for(int j = 1; j < pic_x - 1; j++){

            //If the Non Maximum Suppression array with high thresholding isn't 0 and
            //our output check array isn't filled before
            if(strong->data[idx + j] != 0 && check->data[idx + j] == 0) {
                check->data[idx + j] = 255; //Mark the check array

                lines_idx = 1;
                lines[0] = idx + j;

                int cur_idx = 0;
                while(cur_idx < lines_idx && lines_idx < MAX_LINE-1) {
                    int cur_pos = lines[cur_idx];

                    //Check the neighbors of the current pixel
                    if(cur_pos - pic_x - 1 >= 0 && cur_pos + pic_x + 1 < pic_x * pic_y) { //Check if within borders
                        for(int k1 = -1; k1 < 2; k1++) {
                            for(int k2 = -1; k2 < 2; k2++) {
                                //Check if check array isn't filled and if the non maximum suppression array is set
                                if(check->data[(cur_pos + (k1 * pic_x)) + k2] == 0 && weak->data[(cur_pos + (k1 * pic_x)) + k2] != 0) {
                                    check->data[(cur_pos + (k1 * pic_x)) + k2] = 255; //Mark the check array at the position

                                    lines_idx++;
                                    lines[lines_idx - 1] = (cur_pos + (k1 * pic_x)) + k2;
                                }
                            }
                        }
                    }
                    cur_idx++;
                }
            }

        }

        idx += pic_x;
    }
}

cv::Mat CannyEdgeDetectionCPU::execute() {
    int k_sz=16;
    //Used for noise reduction by applying a gaussian filter
    float gau[16] = {0.000000220358050f,0.000007297256405f,0.000146569312970f,0.001785579770079f,
                     0.013193749090229f,0.059130281094460f,0.160732768610747f,0.265003534507060f,0.265003534507060f,
                     0.160732768610747f,0.059130281094460f,0.013193749090229f,0.001785579770079f,0.000146569312970f,
                     0.000007297256405f,0.000000220358050f};
    float deriv_gau[16] = {-0.000026704586264f,-0.000276122963398f,-0.003355163265098f,-0.024616683775044f,-0.108194751875585f,
                           -0.278368310241814f,-0.388430056419619f,-0.196732206873178f,0.196732206873178f,0.388430056419619f,
                           0.278368310241814f,0.108194751875585f,0.024616683775044f,0.003355163265098f,0.000276122963398f,0.000026704586264f};

    cv::Point anchor = cv::Point( -1, -1 );
    float delta = 0;
    int ddepth = -1;

    pic_->convertTo(*pic_, CV_32FC1);

    cv::Mat gau_x = cv::Mat(1, k_sz, CV_32FC1,&gau);
    cv::Mat deriv_gau_x = cv::Mat(1, k_sz, CV_32FC1,&deriv_gau);

    cv::Mat g_x;
    cv::Mat g_y;

    //Apply the gaussian
    cv::transpose(*pic_,*pic_);
    //Apply gaussian with derivation twice for x and y direction respectively
    filter2D(*pic_, g_x, ddepth , gau_x, anchor, delta, cv::BORDER_REPLICATE );
    cv::transpose(*pic_,*pic_);
    cv::transpose(g_x,g_x);
    filter2D(g_x, g_x, ddepth , deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE );

    filter2D(*pic_, g_y, ddepth , gau_x, anchor, delta, cv::BORDER_REPLICATE );
    cv::transpose(g_y,g_y);
    filter2D(g_y, g_y, ddepth , deriv_gau_x, anchor, delta, cv::BORDER_REPLICATE );
    cv::transpose(g_y,g_y);

    //Compute the intensity gradient. Hypot is defined as sqrt(x^2+y^2)
    //Fast calucation possible by convolotion with sobel operator (see https://rosettacode.org/wiki/Canny_edge_detector)
    cv::Mat res=cv::Mat::zeros(pic_->rows, pic_->cols, CV_32FC1);
    float * g, *p_x, *p_y;
    for(int i=0; i<res.rows; i++){
        g=res.ptr<float>(i);
        p_x=g_x.ptr<float>(i);
        p_y=g_y.ptr<float>(i);

        for(int j=0; j<res.cols; j++){
            g[j]=hypot(p_x[j], p_y[j]);
        }
    }

    //Calculate the threshold value
    int PercentOfPixelsNotEdges=0.7 * res.cols * res.rows;
    float ThresholdRatio=0.4f;

    float high_th=0;
    float low_th=0;

    int h_sz=64;
    int hist[64];
    for(int i=0; i<h_sz; i++) hist[i]=0;

    cv::normalize(res, res, 0, 1, cv::NORM_MINMAX, CV_32FC1);

    cv::Mat res_idx=cv::Mat::zeros(pic_->rows, pic_->cols, CV_8U);
    cv::normalize(res, res_idx, 0, 63, cv::NORM_MINMAX, CV_32S);

    int *p_res_idx=0;
    for(int i=0; i<res.rows; i++){
        p_res_idx=res_idx.ptr<int>(i);
        for(int j=0; j<res.cols; j++){
            hist[p_res_idx[j]]++;
        }}

    int sum=0;
    for(int i=0; i<h_sz; i++){
        sum+=hist[i];
        if(sum>PercentOfPixelsNotEdges){
            high_th=float(i+1)/float(h_sz);
            break;
        }
    }
    low_th=ThresholdRatio*high_th;

    //non maximum supression + interpolation
    cv::Mat non_ms=cv::Mat::zeros(pic_->rows, pic_->cols, CV_8U);
    cv::Mat non_ms_hth=cv::Mat::zeros(pic_->rows, pic_->cols, CV_8U);

    float ix,iy;

    char *p_non_ms,*p_non_ms_hth;
    float * p_res_t, *p_res_b;
    for(int i=1; i<res.rows-1; i++){
        p_non_ms=non_ms.ptr<char>(i);
        p_non_ms_hth=non_ms_hth.ptr<char>(i);

        g=res.ptr<float>(i);
        p_res_t=res.ptr<float>(i-1);
        p_res_b=res.ptr<float>(i+1);

        p_x=g_x.ptr<float>(i);
        p_y=g_y.ptr<float>(i);

        for(int j=1; j<res.cols-1; j++){
            iy=p_y[j];
            ix=p_x[j];

            float grad1, grad2, d;

            bool inDeg = true;
            if( (iy<=0 && ix>-iy) || (iy>=0 && ix<-iy) ){
                d=abs(iy/ix);
                grad1=( g[j+1]*(1-d) ) + ( p_res_t[j+1]*d );
                grad2=( g[j-1]*(1-d) ) + ( p_res_b[j-1]*d );
            } else if( (ix>0 && -iy>=ix)  || (ix<0 && -iy<=ix) ){
                d=abs(ix/iy);
                grad1=( p_res_t[j]*(1-d) ) + ( p_res_t[j+1]*d );
                grad2=( p_res_b[j]*(1-d) ) + ( p_res_b[j-1]*d );
            } else if( (ix<=0 && ix>iy) || (ix>=0 && ix<iy) ){
                d=abs(ix/iy);
                grad1=( p_res_t[j]*(1-d) ) + ( p_res_t[j-1]*d );
                grad2=( p_res_b[j]*(1-d) ) + ( p_res_b[j+1]*d );
            } else if( (iy<0 && ix<=iy) || (iy>0 && ix>=iy)){
                d=abs(iy/ix);
                grad1=( g[j-1]*(1-d) ) + ( p_res_t[j-1]*d );
                grad2=( g[j+1]*(1-d) ) + ( p_res_b[j+1]*d );
            } else {
                inDeg = false;
            }

            if(inDeg && (g[j]>=grad1 && g[j]>=grad2)){
                p_non_ms[j]= (char) 255;
                if(g[j]>high_th)
                    p_non_ms_hth[j]= (char) 255;
            }
        }
    }

    //This should in theory trace the edges with hysteresis
    ////bw select
    cv::Mat res_lin=cv::Mat::zeros(pic_->rows, pic_->cols, CV_8U);
    matlab_bwselect(&non_ms_hth, &non_ms, &res_lin);

    pic_->convertTo(*pic_, CV_8U);

    return res_lin;
}
