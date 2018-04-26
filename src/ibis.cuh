/* -- Serge Bobbia : serge.bobbia@u-bourgogne.fr -- Le2i 2018
 * This work is distributed for non commercial use only,
 * it implements the IBIS method as described in the ICPR 2018 paper.
 * The multi-threading is done with openMP, which is not an optimal solution
 * But was chosen as it was the simpliest way to do it without further justifications.
 *
 *  --------------------------- Benchmark setup : -----------------------------------------------------
 * activate or not the THREAD count from 2 to number of physical CPU core (for best performance)
 * size_roi = 9
 * MATLAB_lab = 0
 * MASK_chrono = 0
 * VISU = 0
 * VISU_all = 0
 * OUTPUT_log = 1 : if you want to get the time output, else 0
 *
 * You better want to run IBIS on a directory than a single file since the initialization of
 * every mask could be time consuming in comparison with the processing itself.
 */

#ifndef IBIS_H
#define IBIS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include "utils.h"

// algo debug parameters
#define THREAD_count        0       // deactivated if <= 1
#define size_roi            9       // 9 25 49 : consider adjacent seeds for pixels assignation
#define MATLAB_lab          0       // 0=>RGB2LAB : l,a,b -> 0-255. 1=>RGB2LAB : l -> 0-100; a,b -> -127:127
#define MASK_chrono         0       // 0:1 provide more informations ( complexity, process burden repartition ) : slow down the process !
#define VISU                1       // show processed pixels for each iterations
#define VISU_all            0       // for mask by mask visu of the processing : THREAD_count = 0, very slow
#define OUTPUT_log          1       // 0:1 print output log

#define STEP                2

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define CUDA_C      16
#define CUDA_SP     32

// mask level max 6 => 49504 o / 65 Ko
__constant__ int __design_check_x[ 508 ];
__constant__ int __design_check_y[ 508 ];
__constant__ int __design_assign_x[ 247 ];
__constant__ int __design_assign_y[ 247 ];
__constant__ int __design_size[ 7 ];
__constant__ int __design_size_to_assign[ 7 ] = { 1, 3, 7, 15, 31, 63, 127 };
__constant__ int __design_size_to_check[ 7 ] = { 4, 8, 16, 32, 64, 128, 256 };
__constant__ int __design_check_start[ 7 ] = { 0, 4, 12, 28, 60, 124, 252 };
__constant__ int __design_assign_start[ 7 ] = { 0, 1, 4, 11, 26, 57, 120 };
__constant__ int __c_size;
__constant__ int __c_width;
__constant__ int __c_height;
__constant__ float __c_invwt;

#if STEP == 2 // 40 o
__constant__ int __last_px_x[ 5 ] = { 1, 0, 1, 2, 1 };
__constant__ int __last_px_y[ 5 ] = { 0, 1, 1, 1, 2 };
#define __count_last 5

#elif STEP == 3 // 96 o 
__constant__ int __last_px_x[ 12 ] = { 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2 };
__constant__ int __last_px_y[ 12 ] = { 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3 };
#define __count_last 12

#elif STEP == 4 // 168 o
__constant__ int __last_px_x[ 21 ] = { 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 1, 2, 3 };
__constant__ int __last_px_y[ 21 ] = { 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4 };
#define __count_last 21

#elif STEP == 5 // 256 o
__constant__ int __last_px_x[ 32 ] = { 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4 };
__constant__ int __last_px_y[ 32 ] = { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5 };
#define __count_last 32

#elif STEP == 6 // 360 o
__constant__ int __last_px_x[ 45 ] = { 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5 };
__constant__ int __last_px_y[ 45 ] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6 };
#define __count_last 45

#else
#define __count_last 0

#endif // more is actually quite ridiculous

typedef struct mask_design {
    int size;

    int size_to_assign;
    int size_to_check;
    
    int* to_assign_x;
    int* to_assign_y;
    
    int* to_check_x;
    int* to_check_y;

} mask_design;

typedef struct mask_apply {
    int size;
    int total_count;
    
    int* apply_x;
    int* apply_y;

} mask_apply;


void generate_mask( int k, int step, mask_design* masks, int* check_x, int* check_y, int* assign_x, int* assign_y );
void generate_coord_mask( mask_design* masks, mask_apply* masks_pos, int width, int height );

typedef struct __c_ibis {
    float* __c_px;
    float* __xs;
    float* __ys;
    float* __ls;
    float* __as;
    float* __bs;
    float* __xs_s;
    float* __ys_s;
    float* __ls_s;
    float* __as_s;
    float* __bs_s;
    
    float* __l_vec;
    float* __a_vec;
    float* __b_vec;
    
    int* __adj_sp;
    int* __c_adj;
    int* __init_repa;
    int* __proc;
    int* __labels;
    int* __t_labels;
    int* __to_fill;

} __c_ibis;

__device__ void RGB2XYZ( const int& sR, const int& sG, const int& sB, double& X, double& Y, double& Z);
	
__device__ void RGB2LAB( const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval );

__global__ void update_seeds( __c_ibis* ibis_data );

__global__ void assign_px( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y  );

__global__ void assign_last( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y );

__global__ void check_boundaries( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y );

__global__ void fill_mask( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int* __c_exec_count, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y, int* __prep_exec_list_x, int* __prep_exec_list_y );

class IBIS
{

public:
    friend class MASK;

    IBIS(int maxSPNum, int compacity);
    virtual ~IBIS();

    void process( cv::Mat* img );
    void init();
    void reset();

    int getMaxSPNumber() { return maxSPNumber;}
    int getActualSPNumber() { return SPNumber; }
    int* getLabels() { return labels; }

    float get_complexity();

protected:
    void getLAB(cv::Mat *img);
    void initSeeds();
    void mask_propagate_SP();
    void mean_seeds();

    double now_ms(void);
    void enforceConnectivity();

private:

    // input parameters
    int size;
    int width;
    int height;

    // internals buffer
    int* adjacent_sp;
    int* count_adjacent;
    int* initial_repartition;
    int* processed;
    int* x_vec;
    int* y_vec;
    int* labels;
    float* inv;
    float* Xseeds_Sum;
    float* Yseeds_Sum;
    float* lseeds_Sum;
    float* aseeds_Sum;
    float* bseeds_Sum;
    float* countPx;

    // inner parameter
    int count_mask;
    int start_xy;
    int index_mask;
    int y_limit;
    int x_limit;
    float invwt;
    int minSPSizeThreshold;

    int SPNumber;               // number of Sp actually created
    int SPTypicalLength;		// typical size of the width or height for a SP
    int compacity;              // compacity factor
    int maxSPNumber;            // number of Sp passed by user

    // seeds value
    float* Xseeds;
    float* Yseeds;
    float* lseeds;
    float* aseeds;
    float* bseeds;

    // initial seeds repartition
    float* Xseeds_init;
    float* Yseeds_init;

    // image data
    float* lvec;
    float* avec;
    float* bvec;
    
    // cuda buffer
    __c_ibis* __c_buffer;
    __c_ibis* __h_buffer;
    
    //
    mask_design* masks;
    mask_design* __c_masks;
    
    mask_apply* masks_pos;
    mask_apply* __c_masks_pos;
    
    int* __c_exec_list_x;
    int* __c_exec_list_y;
    
    int* __prep_exec_list_x;
    int* __prep_exec_list_y;
    
    int exec_count;
    int* __c_exec_count;

public:
    double st2, st3, st4;

};

#endif // IBIS_H
