/* -- Serge Bobbia : serge.bobbia@u-bourgogne.fr -- Le2i 2018
 * This work is distributed for non commercial use only,
 * it implements the IBIS method as described in the ICPR 2018 paper.
 *
 * Read the ibis.h file for options and benchmark instructions
 */

#include "ibis.cuh"

#define SAFE_KER(ans) { kernelAssert((ans), __FILE__, __LINE__); }
inline void kernelAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"%s//!!!\\Kernel Assert:%s %s %s %d\n", KRED, KNRM, cudaGetErrorString(code), file, line);
      
      if (abort)
        exit(code);

   }

}

#define SAFE_C(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
    if(err!=cudaSuccess) {
        fprintf(stderr,"%s%s\n->File: %s\n->Line Number: %d\n->Reason: %s%s\n",KRED,msg,file_name,line_number,cudaGetErrorString(err),KNRM);
        std::cin.get();
        exit(EXIT_FAILURE);

    }

}

void generate_mask( int k, int step, mask_design* masks, int* check_x, int* check_y, int* assign_x, int* assign_y ) {
    int size_to_assign[ 7 ] = { 1, 3, 7, 15, 31, 63, 127 };
    int size_to_check[ 7 ] = { 4, 8, 16, 32, 64, 128, 256 };
    int check_start[ 7 ] = { 0, 4, 12, 28, 60, 124, 252 };
    int assign_start[ 7 ] = { 0, 1, 4, 11, 26, 57, 120 };
    
    masks->size_to_assign = int( pow(2.0, k + 1) ) - 1;
    masks->size_to_check = 4 * int( pow( 2.0, k ) );
    
    cudaMalloc( (void**) &masks->to_assign_x, sizeof(int) * masks->size_to_assign );
    cudaMalloc( (void**) &masks->to_assign_y, sizeof(int) * masks->size_to_assign );
    
    int* to_assign_x = new int[ masks->size_to_assign ];
    int* to_assign_y = new int[ masks->size_to_assign ];
    
    cudaMalloc( (void**) &masks->to_check_x, sizeof(int) * masks->size_to_check );
    cudaMalloc( (void**) &masks->to_check_y, sizeof(int) * masks->size_to_check );
    
    int* to_check_x = new int[ masks->size_to_check ];
    int* to_check_y = new int[ masks->size_to_check ];
    
    masks->size = int( pow( 2.0, k ) ) * step;
    
    int count_var = 0;
    int count_assign = 0;
    
    for( int y_var=0; y_var <= masks->size; y_var+=step ) {
        for( int x_var=0; x_var <= masks->size; x_var+=step ) {
            if( y_var == 0 || y_var == masks->size ) {
                // top / bot row
                to_check_x[ count_var ] = x_var;
                to_check_y[ count_var ] = y_var;
                count_var++;
            
                if( y_var == masks->size && x_var > 0 ) {
                    to_assign_x[ count_assign ] = x_var;
                    to_assign_y[ count_assign ] = y_var;
                    count_assign++;
                    
                }
                
            }
            else {
                // column border
                if( x_var == 0 || x_var == masks->size ) {
                    to_check_x[ count_var ] = x_var;
                    to_check_y[ count_var ] = y_var;
                    count_var++;
                    
                    if( x_var == masks->size && y_var > 0 ) {
                        to_assign_x[ count_assign ] = x_var;
                        to_assign_y[ count_assign ] = y_var;
                        count_assign++;
                        
                    }
                    
                }
            
            }
            
        }

    }
    
    printf("step : %i\n", step);
    for( int i=0; i<masks->size_to_assign; i++)
        printf("( %i, %i ) ", to_assign_x[ i ], to_assign_y[ i ]);
        
    printf("\n");
    
    // printf(" generate masks : %i / %i ; %i / %i\n ", count_var, masks->size_to_check, count_assign, masks->size_to_assign );
    
    cudaMemcpy( masks->to_assign_x, to_assign_x, sizeof( int ) * masks->size_to_assign, cudaMemcpyHostToDevice );
    cudaMemcpy( masks->to_assign_y, to_assign_y, sizeof( int ) * masks->size_to_assign, cudaMemcpyHostToDevice );
    
    cudaMemcpy( masks->to_check_x, to_check_x, sizeof( int ) * masks->size_to_check, cudaMemcpyHostToDevice );
    cudaMemcpy( masks->to_check_y, to_check_y, sizeof( int ) * masks->size_to_check, cudaMemcpyHostToDevice );
    
    memcpy( check_x + check_start[ k ], to_check_x, sizeof( int ) * size_to_check[ k ] );
    memcpy( check_y + check_start[ k ], to_check_y, sizeof( int ) * size_to_check[ k ] );
    memcpy( assign_x + assign_start[ k ], to_assign_x, sizeof( int ) * size_to_assign[ k ] );
    memcpy( assign_y + assign_start[ k ], to_assign_y, sizeof( int ) * size_to_assign[ k ] );
    
    delete[] to_assign_x;
    delete[] to_assign_y;
    delete[] to_check_x;
    delete[] to_check_y;
    
}

void generate_coord_mask( mask_design* masks, mask_apply* masks_pos, int width, int height ) {
    int quantity_apply = ( int(width / masks->size) + 1 ) * ( int(height / masks->size) + 1 );
    
    int* apply_x = new int[ quantity_apply ];
    int* apply_y = new int[ quantity_apply ];
    
    cudaMalloc( (void**) &masks_pos->apply_x, sizeof(int) * quantity_apply );
    cudaMalloc( (void**) &masks_pos->apply_y, sizeof(int) * quantity_apply );
    
    masks_pos->size = quantity_apply;
    
    int ii=0;
    for( int y=-1; y<height; y+=masks->size ) {
        for( int x=-1; x<width; x+=masks->size ) {
            apply_x[ ii ] = x;
            apply_y[ ii ] = y;
            ii++;
        
        }
    
    }
    
    masks_pos->total_count = ii;
    
    cudaMemcpy( masks_pos->apply_x, apply_x, sizeof( int ) * masks_pos->size, cudaMemcpyHostToDevice );
    cudaMemcpy( masks_pos->apply_y, apply_y, sizeof( int ) * masks_pos->size, cudaMemcpyHostToDevice );
    
    delete[] apply_x;
    delete[] apply_y;
    
}

IBIS::IBIS(int _maxSPNum, int _compacity ) {
    labels = nullptr;
    maxSPNumber = _maxSPNum;
    compacity = _compacity;
    size = 0;

    // memory allocation
    /*Xseeds = new float[maxSPNumber];
    Yseeds = new float[maxSPNumber];
    lseeds = new float[maxSPNumber];
    aseeds = new float[maxSPNumber];
    bseeds = new float[maxSPNumber];*/
    cudaMallocHost( (void**)&Xseeds, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&Yseeds, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&lseeds, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&aseeds, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&bseeds, sizeof(float) * maxSPNumber );
    
    Xseeds_init = new float[maxSPNumber];
    Yseeds_init = new float[maxSPNumber];

    /*Xseeds_Sum = new float[maxSPNumber];
    Yseeds_Sum = new float[maxSPNumber];
    lseeds_Sum = new float[maxSPNumber];
    aseeds_Sum = new float[maxSPNumber];
    bseeds_Sum = new float[maxSPNumber];*/
    cudaMallocHost( (void**)&Xseeds_Sum, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&Yseeds_Sum, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&lseeds_Sum, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&aseeds_Sum, sizeof(float) * maxSPNumber );
    cudaMallocHost( (void**)&bseeds_Sum, sizeof(float) * maxSPNumber );
    
    //countPx = new float[maxSPNumber];
    cudaMallocHost( (void**)&countPx, sizeof(float) * maxSPNumber );
    
    inv = new float[maxSPNumber];
    adjacent_sp = new int[size_roi*maxSPNumber];
    count_adjacent = new int[maxSPNumber];
    
    // cuda memory
    cudaMalloc( (void**) &__c_buffer, sizeof( __c_ibis ) );
    __h_buffer = (__c_ibis*)malloc( sizeof( __c_ibis ) );
    
    cudaMalloc( (void**) &__h_buffer->__c_px, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__xs, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ys, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ls, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__as, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__bs, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__xs_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ys_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ls_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__as_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__bs_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__adj_sp, size_roi*maxSPNumber*sizeof(int) );
    cudaMalloc( (void**) &__h_buffer->__c_adj, maxSPNumber*sizeof(int) );
    
}

IBIS::~IBIS() {
    delete[] labels;
    delete[] avec;
    delete[] lvec;
    delete[] bvec;
    delete[] inv;

    // allocated in constructor
    /*delete[] Xseeds;
    delete[] Yseeds;
    delete[] lseeds;
    delete[] aseeds;
    delete[] bseeds;*/
    SAFE_C( cudaFreeHost( Xseeds ), "free Xseeds" );
    SAFE_C( cudaFreeHost( Yseeds ), "free Yseeds" );
    SAFE_C( cudaFreeHost( lseeds ), "free lseeds" );
    SAFE_C( cudaFreeHost( aseeds ), "free aseeds" );
    SAFE_C( cudaFreeHost( bseeds ), "free bseeds" );
    
    delete[] Xseeds_init;
    delete[] Yseeds_init;

    /*delete[] Xseeds_Sum;
    delete[] Yseeds_Sum;
    delete[] lseeds_Sum;
    delete[] aseeds_Sum;
    delete[] bseeds_Sum;*/
    SAFE_C( cudaFreeHost( Xseeds_Sum ), "free Xseeds_Sum" );
    SAFE_C( cudaFreeHost( Yseeds_Sum ), "free Yseeds_Sum" );
    SAFE_C( cudaFreeHost( lseeds_Sum ), "free lseeds_Sum" );
    SAFE_C( cudaFreeHost( aseeds_Sum ), "free aseeds_Sum" );
    SAFE_C( cudaFreeHost( bseeds_Sum ), "free bseeds_Sum" );

    //delete[] countPx;
    SAFE_C( cudaFreeHost( countPx ), "free countPx" );

    //delete[] mask_size;
    delete[] x_vec;
    delete[] y_vec;

    delete[] adjacent_sp;
    delete[] count_adjacent;
    delete[] initial_repartition;
    delete[] processed;
    
    // cuda
    cudaFree( __h_buffer->__c_px );
    cudaFree( __h_buffer->__xs );
    cudaFree( __h_buffer->__ys );
    cudaFree( __h_buffer->__ls );
    cudaFree( __h_buffer->__as );
    cudaFree( __h_buffer->__bs );
    cudaFree( __h_buffer->__xs_s );
    cudaFree( __h_buffer->__ys_s );
    cudaFree( __h_buffer->__ls_s );
    cudaFree( __h_buffer->__as_s );
    cudaFree( __h_buffer->__bs_s );
    cudaFree( __h_buffer->__adj_sp );
    cudaFree( __h_buffer->__c_adj );
    cudaFree( __h_buffer->__init_repa );
    cudaFree( __h_buffer->__proc );
    cudaFree( __h_buffer->__labels );
    cudaFree( __h_buffer->__t_labels );
    cudaFree( __h_buffer->__to_fill );
    cudaFree( __h_buffer->__to_fill_x );
    cudaFree( __h_buffer->__to_fill_y );
    cudaFree( __h_buffer->__to_split_x );
    cudaFree( __h_buffer->__to_split_y );
    
    cudaFree( __h_buffer->__l_vec );
    cudaFree( __h_buffer->__a_vec );
    cudaFree( __h_buffer->__b_vec );
    
    cudaFree( __c_exec_count );
    cudaFree( __c_fill );
    cudaFree( __c_split );
    
    cudaFree( __c_exec_list_x );
    cudaFree( __c_exec_list_y );
    
    cudaFree( __prep_exec_list_x );
    cudaFree( __prep_exec_list_y );
    
    //cudaFree( __c_buffer );
    //free( __h_buffer );

    //free( global_buffer );
    
    cudaDeviceReset();

}

void IBIS::initSeeds() {
    int n;
    int xstrips, ystrips;
    int xerr, yerr;
    double xerrperstrip, yerrperstrip;
    int xoff, yoff;
    int x, y;
    int xe, ye, xe_1, ye_1;
    int start_y, final_y, start_x, final_x;
    int* tmp_adjacent_sp = new int[size_roi*maxSPNumber];
    int* tmp_count_adjacent = new int[maxSPNumber];

    xstrips = width / SPTypicalLength;
    ystrips = height / SPTypicalLength;

    xerr = width - SPTypicalLength * xstrips;
    yerr = height - SPTypicalLength * ystrips;

    xerrperstrip = (double)xerr / xstrips;
    yerrperstrip = (double)yerr / ystrips;

    xoff = SPTypicalLength / 2;
    yoff = SPTypicalLength / 2;

    n = 0;
    for (y = 0; y < ystrips; y++)
    {
        ye = (int)(y*yerrperstrip);
        ye_1 = (int)((y+1)*yerrperstrip);

        int seedy = y * SPTypicalLength + yoff + ye;

        if( y == 0 ) {
            start_y = 0;
            final_y = SPTypicalLength + ye_1;

        }
        else {
            start_y = y * SPTypicalLength + ye;
            final_y = ( (y + 1) * SPTypicalLength + ye_1 >= height ) ? height-1 : (y + 1) * SPTypicalLength + ye_1;

        }

        for (x = 0; x < xstrips; x++)
        {
            int seedx;
            xe = (int)(x*xerrperstrip);
            xe_1 = (int)((x+1)*xerrperstrip);
            seedx = x * SPTypicalLength + xoff + xe;

            if( x == 0 ) {
                start_x = 0;
                final_x = SPTypicalLength + xe_1;

            }
            else {
                start_x = x * SPTypicalLength + xe;

                final_x = ( (x + 1) * SPTypicalLength + xe_1 > width ) ? width : (x + 1) * SPTypicalLength + xe_1;

            }

            Xseeds_init[n] = (float) seedx;
            Yseeds_init[n] = (float) seedy;

            // fill line by line
            for( int index_y=start_y; index_y<=final_y; index_y++ ) {
                std::fill( initial_repartition + index_y*width + start_x, initial_repartition + index_y*width + final_x, n );

            }

            // list adjacents seeds
            tmp_count_adjacent[n] = 0;
            for( int roi_y=-(sqrt(size_roi) - 1)/2; roi_y <= (sqrt(size_roi) - 1)/2; roi_y++ ) {
                for( int roi_x=-(sqrt(size_roi) - 1)/2; roi_x <= (sqrt(size_roi) - 1)/2; roi_x++ ) {
                    if( !( y + roi_y < 0 || y + roi_y >= ystrips || x + roi_x < 0 || x + roi_x >= xstrips ) ) {
                        tmp_adjacent_sp[size_roi*n+tmp_count_adjacent[n]] = n + roi_y*xstrips + roi_x;
                        tmp_count_adjacent[n]++;

                    }

                }

            }

            n++;
        }
    }
    SPNumber = n;

    for(int i=0; i<SPNumber; i++) {
        count_adjacent[i] = 0;

        for(int j=0; j<tmp_count_adjacent[i]; j++) {
            if( tmp_adjacent_sp[size_roi*i+j] >= 0 && tmp_adjacent_sp[size_roi*i+j] < SPNumber ) {
                adjacent_sp[size_roi*i+count_adjacent[i]] = tmp_adjacent_sp[size_roi*i+j];
                count_adjacent[i]++;

            }

        }

    }

    delete[] tmp_adjacent_sp;
    delete[] tmp_count_adjacent;

}

void IBIS::mean_seeds() {

    for( int i=0; i<SPNumber; i++ ) {
        inv[ i ] = 1.f / countPx[ i ];
    }

    for( int i=0; i<SPNumber; i++ ) {
        if( countPx[ i ] > 0 ) {
            Xseeds[ i ] = Xseeds_Sum[ i ] * inv[ i ];
            Yseeds[ i ] = Yseeds_Sum[ i ] * inv[ i ];
            lseeds[ i ] = lseeds_Sum[ i ] * inv[ i ];
            aseeds[ i ] = aseeds_Sum[ i ] * inv[ i ];
            bseeds[ i ] = bseeds_Sum[ i ] * inv[ i ];

        }

    }

}

float IBIS::get_complexity() {
    int count_px_processed = 0;
    for( int i=0; i<size; i++ ) {
        if( processed[ i ] )
            count_px_processed++;

    }

    return float(count_px_processed) / float(size);

}

double IBIS::now_ms(void)
{
    double milliseconds_since_epoch = (double) (std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    return milliseconds_since_epoch;

}

// return the current actual SP number (the method may reduce the actual Sp number).
void IBIS::enforceConnectivity()
{
    //local var
    int label = 0;
    int i, j, k;
    int n, c, count;
    int x, y;
    int ind;
    int oindex, adjlabel;
    int nindex;
    const int dx4[4] = { -1,  0,  1,  0 };
    const int dy4[4] = { 0, -1,  0,  1 };
    int* nlabels = new int[ size ];
    std::fill( nlabels, nlabels + size, -1 );

    oindex = 0;
    adjlabel = 0;

    for (j = 0; j < height; j++)
    {
        for (k = 0; k < width; k++)
        {
            if (nlabels[oindex] < 0)
            {
                nlabels[oindex] = label;// !! labels[oindex] --> label

                x_vec[0] = k;
                y_vec[0] = j;

                for (n = 0; n < 4; n++)
                {
                    x = x_vec[0] + dx4[n];
                    y = y_vec[0] + dy4[n];

                    if ((x >= 0 && x < width) && (y >= 0 && y < height))
                    {
                        nindex = y*width + x;

                        if (nlabels[nindex] >= 0)
                            adjlabel = nlabels[nindex];
                    }
                }

                count = 1;
                for (c = 0; c < count; c++)
                {
                    for (n = 0; n < 4; n++)
                    {
                        x = x_vec[c] + dx4[n];
                        y = y_vec[c] + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            nindex = y*width + x;

                            if (nlabels[nindex] < 0 && labels[oindex] == labels[nindex])
                            {
                                x_vec[count] = x;
                                y_vec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }
                    }
                }

                if (count <= minSPSizeThreshold)
                {
                    for (c = 0; c < count; c++)
                    {
                        ind = y_vec[c] * width + x_vec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }

    for (i = 0; i < size; i++)
        labels[i] = nlabels[i];

    delete[] nlabels;

}

void IBIS::init() {
    // image lab buffer
    avec = new float[size];
    bvec = new float[size];
    lvec = new float[size];

    //store mean distance between 2 seeds info
    SPTypicalLength = (int)(std::sqrt((float)(size) / (float)(maxSPNumber))) + 1;

    // compacity weight
    invwt = (float)SPTypicalLength / compacity;
    invwt = 1.0f / (invwt * invwt);

    // inferior limit for superpixels size
    minSPSizeThreshold = (size / maxSPNumber) / 4;

    // set the top mask size
    index_mask = 1;
    while( SPTypicalLength > ( pow( 2.0, index_mask - 1 ) * STEP ) )
        index_mask++;
    
    index_mask--;
    
    if( index_mask > 6 )
        index_mask = 6;
    
    // mask construction
    masks = new mask_design[ index_mask ];
    SAFE_C( cudaMalloc( (void**) &__c_masks, sizeof( mask_design ) * index_mask ), "cudaMalloc");
    
    masks_pos = new mask_apply[ index_mask ];
    SAFE_C( cudaMalloc( (void**) &__c_masks_pos, sizeof( mask_apply ) * index_mask ), "cudaMalloc");
    
    // set constant memory
    int check_x[ 508 ];
    int check_y[ 508 ];
    int assign_x[ 247 ];
    int assign_y[ 247 ];
    
    int design_size[ 7 ] = {0};
    for( int k=0; k<7; k++ )
        design_size[ k ] = int( pow( 2.0, k ) ) * STEP;
    
    SAFE_C( cudaMemcpyToSymbol( __design_size, design_size, sizeof(int) * 7 ), "constant");
    
    for( int k=0; k<index_mask; k++ ) {
        // design masks
        generate_mask( k, STEP, &masks[k], check_x, check_y, assign_x, assign_y );
        
        // position to apply masks
        generate_coord_mask( &masks[k], &masks_pos[k], width, height );
        
    }
    
    SAFE_C( cudaMemcpyToSymbol( __design_check_x, check_x, sizeof(int) * 508 ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __design_check_y, check_y, sizeof(int) * 508 ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __design_assign_x, assign_x, sizeof(int) * 247 ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __design_assign_y, assign_y, sizeof(int) * 247 ), "constant");
    
    // save image params in constant memory
    SAFE_C( cudaMemcpyToSymbol( __c_invwt, &invwt, sizeof(float) ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __c_size, &size, sizeof(int) ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __c_width, &width, sizeof(int) ), "constant");
    SAFE_C( cudaMemcpyToSymbol( __c_height, &height, sizeof(int) ), "constant");
    
    SAFE_C( cudaMemcpy( __c_masks_pos, masks_pos, sizeof( mask_apply ) * index_mask, cudaMemcpyHostToDevice ), "cudaMemcpy");
    SAFE_C( cudaMemcpy( __c_masks, masks, sizeof( mask_design ) * index_mask, cudaMemcpyHostToDevice ), "cudaMemcpy");
    
    // enforce connectivity buffer
    x_vec = new int[ size ];
    y_vec = new int[ size ];

    // visu the processed pixels
    processed = new int[ size ];
    std::fill( processed, processed + size, 0 );

    // output labels buffer
    labels = new int[size];

    // repartition of pixels at start
    initial_repartition = new int[size];

    // cuda alloc init data
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__labels, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__t_labels, sizeof( int ) * size ), "cudaMalloc");
    
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__to_fill, sizeof( int ) * maxSPNumber ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__to_fill_x, sizeof( int ) * maxSPNumber ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__to_fill_y, sizeof( int ) * maxSPNumber ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__to_split_x, sizeof( int ) * maxSPNumber ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__to_split_y, sizeof( int ) * maxSPNumber ), "cudaMalloc");
    
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__proc, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__init_repa, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__l_vec, sizeof( float ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__a_vec, sizeof( float ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__h_buffer->__b_vec, sizeof( float ) * size ), "cudaMalloc");
    
    SAFE_C(  cudaMalloc( (void**) &__c_exec_list_x, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__c_exec_list_y, sizeof( int ) * size ), "cudaMalloc");
    
    SAFE_C(  cudaMalloc( (void**) &__prep_exec_list_x, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C(  cudaMalloc( (void**) &__prep_exec_list_y, sizeof( int ) * size ), "cudaMalloc");
    
    SAFE_C(  cudaMemcpy( __c_buffer, __h_buffer, sizeof( __c_ibis ), cudaMemcpyHostToDevice ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__c_exec_count, sizeof( int ) ), "cudaMalloc" );
    SAFE_C( cudaMalloc( (void**) &__c_fill, sizeof( int ) ), "cudaMalloc" );
    SAFE_C( cudaMalloc( (void**) &__c_split, sizeof( int ) ), "cudaMalloc" );
    
}

void IBIS::reset() {
    int index_xy;

    st4 = 0;
    st2 = 0;
    st3 = 0;

    std::fill( countPx, countPx + maxSPNumber, 0 );
    std::fill( labels, labels + size, -1 );

    for( int i=0; i < SPNumber; i++ ) {
        Xseeds[ i ] = Xseeds_init[ i ];
        Yseeds[ i ] = Yseeds_init[ i ];

        index_xy = (int) Yseeds[ i ] * width + Xseeds[ i ];

        lseeds[ i ] = lvec[ index_xy ];
        aseeds[ i ] = avec[ index_xy ];
        bseeds[ i ] = bvec[ index_xy ];
    }

    memset( lseeds_Sum, 0, sizeof( float ) * maxSPNumber );
    memset( aseeds_Sum, 0, sizeof( float ) * maxSPNumber );
    memset( bseeds_Sum, 0, sizeof( float ) * maxSPNumber );
    memset( Xseeds_Sum, 0, sizeof( float ) * maxSPNumber );
    memset( Yseeds_Sum, 0, sizeof( float ) * maxSPNumber );

    /*for( int i=0; i<count_mask; i++ )
        mask_buffer[ i ].reset();*/

}

void IBIS::getLAB( cv::Mat* img ) {
    cv::Mat lab_image;
    cv::cvtColor(*img, lab_image, CV_BGR2Lab, 0);

    int ii = 0;
    for (int i = 0; i < size * 3; i += 3) {
#if MATLAB_lab
        lvec[ii] = lab_image.ptr()[i] * 100 / 255;
        avec[ii] = lab_image.ptr()[i + 1] - 128;
        bvec[ii] = lab_image.ptr()[i + 2] - 128;
#else
        lvec[ii] = lab_image.ptr()[i];
        avec[ii] = lab_image.ptr()[i + 1];
        bvec[ii] = lab_image.ptr()[i + 2];
#endif
        ii++;
    }

}

void IBIS::process( cv::Mat* img ) {

    double lap;

    if( size == 0 ) {
        size = img->cols * img->rows;
        width = img->cols;
        height = img->rows;
        
        // initialise all the buffer and inner parameters
        init();

        // STEP 1 : initialize with fix grid seeds value
        initSeeds();

    }

    // convert to Lab
    getLAB( img );

    // prepare value to compute a picture
    reset();
    
    cudaMemcpy( __h_buffer->__xs, Xseeds, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__ys, Yseeds, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__ls, lseeds, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__as, aseeds, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__bs, bseeds, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    
    cudaMemcpy( __h_buffer->__c_px, countPx, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__xs_s, Xseeds_Sum, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__ys_s, Yseeds_Sum, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__ls_s, lseeds_Sum, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__as_s, aseeds_Sum, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__bs_s, bseeds_Sum, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__adj_sp, adjacent_sp, maxSPNumber * size_roi * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__c_adj, count_adjacent, maxSPNumber * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__init_repa, initial_repartition, size * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__proc, processed, size * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__labels, labels, size * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__t_labels, labels, size * sizeof( int ), cudaMemcpyHostToDevice );
    
    cudaMemcpy( __h_buffer->__l_vec, lvec, size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__a_vec, avec, size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( __h_buffer->__b_vec, bvec, size * sizeof( float ), cudaMemcpyHostToDevice );
    
    // STEP 2 : process IBIS
#if OUTPUT_log
    lap = now_ms();
#endif
    mask_propagate_SP();

#if OUTPUT_log
    st3 = now_ms() - lap;
#endif
    
    // STEP 3 : post processing
#if OUTPUT_log
    lap = now_ms();
#endif

    enforceConnectivity();

#if OUTPUT_log
    st4 = now_ms() - lap;
#endif

    // output log
#if OUTPUT_log
    printf("-----------------\n");
    printf("PERF_T %lf\n", st3+st4);
    printf("IBIS.process\t\t%lf\t ms\n", st3);
    printf("Kernel exec\t\t%lf\t ms\n", st2);
    printf("IBIS.post_process\t%lf\t ms\n", st4);

    #if MASK_chrono
    float chrono[4] = { 0.f };
    for( int i=0; i < count_mask; i++ )
        mask_buffer[i].get_chrono( chrono );

    float total_chrono = chrono[0] + chrono[1] + chrono[2] + st2;

    printf("-----------------------------------\n");
    printf("Pixels processed:\t%lf\t\%\n", get_complexity()*100 );
    printf("-----------------\n");
    printf("\tMASK.angular_assign()\t\t%lf \%\n", chrono[0]/total_chrono);
    printf("\tMASK.fill_mask()\t\t%lf \%\n", chrono[1]/total_chrono);
    printf("\tMASK.assign_last()\t\t%lf \%\n", chrono[2]/total_chrono);
    printf("\tIBIS.mean_seeds()\t\t%lf \%\n", st2/total_chrono);

    #if THREAD_count > 1
    printf("-----------------\n");
    printf("multi-thread accel:\t\t\t%lf times\n", total_chrono/st3);
    printf("-----------------\n");
    #endif

    #endif

#endif

}

//--------------------------------------------------------------------------------CUDA

__global__ void assign_last( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y ) {
    int index = blockIdx.x * CUDA_C + threadIdx.x;
    int px_to_compute = threadIdx.y;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_exec_list_x[ index ];
    int y_ref = __c_exec_list_y[ index ];
    
    if( x_ref == 0 || y_ref == 0 )
        return;
    
    int x = x_ref + __last_px_x[ px_to_compute ];
    int y = y_ref + __last_px_y[ px_to_compute ];
    
    if( x >= 0 && x < __c_width && y >= 0 && y < __c_height ) {
        int index_xy = y * __c_width + x;
        
        int init_repart = __c_buffer->__init_repa[ index_xy ];
        
        float l = __c_buffer->__l_vec[ index_xy ];
        float a = __c_buffer->__a_vec[ index_xy ];
        float b = __c_buffer->__b_vec[ index_xy ];
        
        int best_sp = 0;
        int index_sp;
        float dist_lab, dist_xy;
        float D = -1.f;
        float total_dist;
        
        for(int i=0; i<__c_buffer->__c_adj[ init_repart ]; i++) {
            index_sp = __c_buffer->__adj_sp[ size_roi * init_repart + i ];
            
            dist_lab = ( l - __c_buffer->__ls[ index_sp ]) * ( l - __c_buffer->__ls[ index_sp ]) +
                       ( a - __c_buffer->__as[ index_sp ]) * ( a - __c_buffer->__as[ index_sp ]) +
                       ( b - __c_buffer->__bs[ index_sp ]) * ( b - __c_buffer->__bs[ index_sp ]);

            dist_xy = ( x - __c_buffer->__xs[ index_sp ] ) * ( x - __c_buffer->__xs[ index_sp ] ) +
                      ( y - __c_buffer->__ys[ index_sp ] ) * ( y - __c_buffer->__ys[ index_sp ] );

            total_dist = dist_lab + dist_xy * __c_invwt;

            if( total_dist < D || D < 0) {
                best_sp = index_sp;
                D = total_dist;

            }
            
        }
        
        // assign labels
        __c_buffer->__labels[ index_xy ] = best_sp;

#if VISU
        __c_buffer->__proc[ index_xy ] = 10;
#endif
    }
    
}

__global__ void update_seeds( __c_ibis* __c_buffer ) {
    int i = CUDA_SP * blockIdx.x + threadIdx.x;
    float inv;
    
    if( __c_buffer->__c_px[ i ] > 0 ) {
        inv = 1.f / __c_buffer->__c_px[ i ];
        
        __c_buffer->__xs[ i ] = __c_buffer->__xs_s[ i ] * inv;
        __c_buffer->__ys[ i ] = __c_buffer->__ys_s[ i ] * inv;
        __c_buffer->__ls[ i ] = __c_buffer->__ls_s[ i ] * inv;
        __c_buffer->__as[ i ] = __c_buffer->__as_s[ i ] * inv;
        __c_buffer->__bs[ i ] = __c_buffer->__bs_s[ i ] * inv;
        
    }

}

__global__ void assign_px( int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int px_to_compute = threadIdx.y;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_exec_list_x[ index ];
    int y_ref = __c_exec_list_y[ index ];
    
    if( x_ref == 0 || y_ref == 0 )
        return;
    
    /*if( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
        printf("k : %i , step : %i\n", k, STEP);
        for( int i=0; i<__design_size_to_assign[k]; i++)
            printf("( %i, %i ) ", 
            __design_assign_x[ __design_assign_start[ k ] + i ], 
            __design_assign_y[ __design_assign_start[ k ] + i ]);
            
        printf("\n\n");
        
    }*/
    
    int x = x_ref + __design_assign_x[ __design_assign_start[ k ] + px_to_compute ];
    int y = y_ref + __design_assign_y[ __design_assign_start[ k ] + px_to_compute ];
    
    if( x >= 0 && x < __c_width && y >= 0 && y < __c_height ) {
        int index_xy = y * __c_width + x;
        
        int init_repart = __c_buffer->__init_repa[ index_xy ];
        
        float l = __c_buffer->__l_vec[ index_xy ];
        float a = __c_buffer->__a_vec[ index_xy ];
        float b = __c_buffer->__b_vec[ index_xy ];
        
        int best_sp = 0;
        int index_sp;
        float dist_lab, dist_xy;
        float D = -1.f;
        float total_dist;
        
        // prepare shared buffer
        int nb_adj = __c_buffer->__c_adj[ init_repart ];
        //int __adj_sp[ size_roi ];
        //memcpy( __adj_sp, __c_buffer->__adj_sp + size_roi * init_repart, sizeof(int) * size_roi );
        
        /*extern __shared__ float __s_seeds_x[];
        extern __shared__ float __s_seeds_y[];
        extern __shared__ float __s_seeds_l[];
        extern __shared__ float __s_seeds_a[];
        extern __shared__ float __s_seeds_b[];
        
        int __adj_sp[ size_roi ];
        memcpy( __adj_sp, __c_buffer->__adj_sp + size_roi * init_repart, sizeof(int) * size_roi );
        
        int index_seeds = size_roi * threadIdx.x + threadIdx.y;
        __s_seeds_x[ index_seeds ] = x;
        __s_seeds_y[ index_seeds ] = y;
        __s_seeds_l[ index_seeds ] = __c_buffer->__ls[ __adj_sp[ threadIdx.y ] ];
        __s_seeds_a[ index_seeds ] = __c_buffer->__as[ __adj_sp[ threadIdx.y ] ];
        __s_seeds_b[ index_seeds ] = __c_buffer->__bs[ __adj_sp[ threadIdx.y ] ];
        
        __syncthreads();
        
        for(int i=0; i<nb_adj; i++) {
            index_sp = __adj_sp[ i ];
            
            dist_lab = ( l - __s_seeds_l[ size_roi * threadIdx.x + i ]) * ( l - __s_seeds_l[ size_roi * threadIdx.x + i ]) +
                       ( a - __s_seeds_a[ size_roi * threadIdx.x + i ]) * ( a - __s_seeds_a[ size_roi * threadIdx.x + i ]) +
                       ( b - __s_seeds_b[ size_roi * threadIdx.x + i ]) * ( b - __s_seeds_b[ size_roi * threadIdx.x + i ]);

            dist_xy = ( x - __s_seeds_x[ size_roi * threadIdx.x + i ] ) * ( x - __s_seeds_x[ size_roi * threadIdx.x + i ] ) +
                      ( y - __s_seeds_y[ size_roi * threadIdx.x + i ] ) * ( y - __s_seeds_y[ size_roi * threadIdx.x + i ] );

            total_dist = dist_lab + dist_xy * __c_invwt;

            if( total_dist < D || D < 0) {
                best_sp = index_sp;
                D = total_dist;

            }
            
        }*/
        
        for(int i=0; i<nb_adj; i++) {
            index_sp = __c_buffer->__adj_sp[ size_roi * init_repart + i ];
            //index_sp = __adj_sp[ i ];
            
            dist_lab = ( l - __c_buffer->__ls[ index_sp ]) * ( l - __c_buffer->__ls[ index_sp ]) +
                       ( a - __c_buffer->__as[ index_sp ]) * ( a - __c_buffer->__as[ index_sp ]) +
                       ( b - __c_buffer->__bs[ index_sp ]) * ( b - __c_buffer->__bs[ index_sp ]);

            dist_xy = ( x - __c_buffer->__xs[ index_sp ] ) * ( x - __c_buffer->__xs[ index_sp ] ) +
                      ( y - __c_buffer->__ys[ index_sp ] ) * ( y - __c_buffer->__ys[ index_sp ] );

            total_dist = dist_lab + dist_xy * __c_invwt;

            if( total_dist < D || D < 0) {
                best_sp = index_sp;
                D = total_dist;

            }
            
        }
        
        // assign temporary labels
        __c_buffer->__t_labels[ index_xy ] = best_sp;
        
        if( k == 0 )
            __c_buffer->__labels[ index_xy ] = best_sp;

#if VISU
        __c_buffer->__proc[ index_xy ] = 10;
#endif
    }
    
}

__global__ void check_boundaries( int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y, int* __c_split, int* __c_fill ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_exec_list_x[ index ];
    int y_ref = __c_exec_list_y[ index ];
    
    if( x_ref == 0 || y_ref == 0 )
        return;
    
    int x;
    int y;
    int index_xy;
    int ref=-1;
    int ii=0;
    int current_sp;
    bool state = true;
    
    for( int i=0; i<__design_size_to_check[k]; i++ ) {
        x = x_ref + __design_check_x[ __design_check_start[ k ] + i ];
        y = y_ref + __design_check_y[ __design_check_start[ k ] + i ];
        
        if( x >= 0 && x < __c_width && y >= 0 && y < __c_height ) {
            index_xy = y*__c_width + x;
            
            if( ii == 0 ) {
                ref = __c_buffer->__t_labels[ index_xy ];
                
            }
            else {
                current_sp = __c_buffer->__t_labels[ index_xy ];
                if( ref != current_sp || current_sp < 0 ) {
                    state = false;
                    break;
                    
                }
            
            }
            
            ii++;
            //__c_buffer->__proc[ index_xy ] = 10;
        
        }
    
    }
    
    if( !state || ref < 0 ) {
        int base = atomicAdd( __c_split, 1 );
        __c_buffer->__to_split_x[ base ] = x_ref;
        __c_buffer->__to_split_y[ base ] = y_ref;
        if( x_ref == -1 && y_ref == -1 )
            printf(" to split \n ");
    }
    else {
        int base = atomicAdd( __c_fill, 1 );
        __c_buffer->__to_fill_x[ base ] = x_ref;
        __c_buffer->__to_fill_y[ base ] = y_ref;
        __c_buffer->__to_fill[ base ] = ref;
        if( x_ref == -1 && y_ref == -1 )
            printf(" to fill \n ");
        
    }
    
}

__global__ void fill_mask( int k, __c_ibis* __c_buffer, int exec_count ) {
    int index = blockIdx.x * CUDA_C + threadIdx.x;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_buffer->__to_fill_x[ index ];
    int y_ref = __c_buffer->__to_fill_y[ index ];
    int ref = __c_buffer->__to_fill[ index ];

    int x;
    int y;
    int index_xy;
    float __tmp_l = 0.f;
    float __tmp_a = 0.f;
    float __tmp_b = 0.f;
    float __tmp_x = 0.f;
    float __tmp_y = 0.f;
    
    // fill labels
    int ii = 0;
    for( int i=0; i<=__design_size[ k ]; i++ ) {
        y = y_ref + i;
        
        if( y >= 0 && y < __c_height ) {
            for( int j=0; j<=__design_size[ k ]; j++ ) {
                x = x_ref + j;
                
                if( x >= 0 && x < __c_width ) {
                    index_xy = y*__c_width + x;
                    __c_buffer->__labels[ index_xy ] = ref;
                    
                    if( ref >= 0 ) {
                        __tmp_l += __c_buffer->__l_vec[ index_xy ];
                        __tmp_a += __c_buffer->__a_vec[ index_xy ];
                        __tmp_b += __c_buffer->__b_vec[ index_xy ];
                        __tmp_x += x;
                        __tmp_y += y;
                        ii++;
                        
                    }
                
                }
            
            }
            
        }
    
    }
    
    // add tmp value to seeds sum
    atomicAdd( &__c_buffer->__c_px[ ref ], ii );
    atomicAdd( &__c_buffer->__xs_s[ ref ], __tmp_x );
    atomicAdd( &__c_buffer->__ys_s[ ref ], __tmp_y );
    atomicAdd( &__c_buffer->__ls_s[ ref ], __tmp_l );
    atomicAdd( &__c_buffer->__as_s[ ref ], __tmp_a );
    atomicAdd( &__c_buffer->__bs_s[ ref ], __tmp_b );

}

__global__ void split_mask( int k, __c_ibis* __c_buffer, int* __c_exec_count, int exec_count, int* __prep_exec_list_x, int* __prep_exec_list_y ) {
    int index = blockIdx.x * CUDA_C + threadIdx.x;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_buffer->__to_split_x[ index ];
    int y_ref = __c_buffer->__to_split_y[ index ];
    
    if( x_ref == -1 && y_ref == -1 )
        printf(" index: %i, split x: %i, y: %i \n", index, x_ref, y_ref);
    
    if( k > 0 ) {
        int base = atomicAdd( __c_exec_count, 4 );
        
        __prep_exec_list_y[ base + 0 ] = y_ref;
        __prep_exec_list_y[ base + 1 ] = y_ref;
        __prep_exec_list_y[ base + 2 ] = y_ref + __design_size[ k - 1 ];
        __prep_exec_list_y[ base + 3 ] = y_ref + __design_size[ k - 1 ];
        
        __prep_exec_list_x[ base + 0 ] = x_ref;
        __prep_exec_list_x[ base + 1 ] = x_ref + __design_size[ k - 1 ];
        __prep_exec_list_x[ base + 2 ] = x_ref;
        __prep_exec_list_x[ base + 3 ] = x_ref + __design_size[ k - 1 ];
        
    }
    else {
        int base = atomicAdd( __c_exec_count, 1 );
        __prep_exec_list_y[ base + 0 ] = y_ref;
        __prep_exec_list_x[ base + 0 ] = x_ref;
        
    }
    
}

//--------------------------------------------------------------------------------CUDA

void set_grid_block_dim( int* __g_dim, int* __t_dim, int ref_t, int value ) {
    *__g_dim = int( value / ref_t );
    *__t_dim = value % ref_t;
        
    if( *__g_dim > 0 ) {
        if( *__t_dim > 0 )
            *__g_dim++;
            
        *__t_dim = ref_t;
        
    }
    else {
        *__g_dim = 1;
        *__t_dim = value;
        
    }
    
}

void IBIS::mask_propagate_SP() {
#if OUTPUT_log
    double lap;
#endif
    st2=0;
    
    int __g_dim_assign;
    int __t_dim_assign;
    
    int __g_dim_fill;
    int __t_dim_fill;
    
    int __g_dim_split;
    int __t_dim_split;
    
    int __g_dim_sp;
    int __t_dim_sp;
    
    set_grid_block_dim( &__g_dim_sp, &__t_dim_sp, CUDA_SP, SPNumber );
    
    // main loop
    for( int k=index_mask-1; k>=0; k--) {
        
        printf( " |-> ---- ---- ---- <-| \n" );
        
#if OUTPUT_log
        lap = now_ms();
#endif
        
        //int cuda_c[8] = { 512, 171, 74, 35, 9, 5, 3, 2 };
        if( k == index_mask-1 ) {
            exec_count = masks_pos[ k ].total_count;
            cudaMemcpy( __c_exec_list_x, masks_pos[k].apply_x, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            cudaMemcpy( __c_exec_list_y, masks_pos[k].apply_y, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            
        }
        else {
            cudaMemcpy( &exec_count, __c_exec_count, sizeof( int ), cudaMemcpyDeviceToHost );
            cudaMemcpy( __c_exec_list_x, __prep_exec_list_x, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            cudaMemcpy( __c_exec_list_y, __prep_exec_list_y, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            
        }
        
        SAFE_C( cudaMemset( __c_exec_count, 0, sizeof(int) ), "cudaMemset" );
        SAFE_C( cudaMemset( __c_fill, 0, sizeof(int) ), "cudaMemset" );
        SAFE_C( cudaMemset( __c_split, 0, sizeof(int) ), "cudaMemset" );
        
        set_grid_block_dim( &__g_dim_assign, &__t_dim_assign, CUDA_C, exec_count );
        printf(" || -- iteration %i : %i / %i = %f \n", k, exec_count, masks_pos[k].total_count, float(exec_count)/float(masks_pos[k].total_count)*100);
        printf(" |-> assign_px (%i ; %i,%i) => %i \n", __g_dim_assign, __t_dim_assign, masks[ k ].size_to_assign, __t_dim_assign * masks[ k ].size_to_assign );

        // assign px
        assign_px <<< __g_dim_assign, dim3( __t_dim_assign, masks[ k ].size_to_assign ) >>> ( k, __c_buffer, exec_count, __c_exec_list_x, __c_exec_list_y );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );
        
        // check borders
        check_boundaries <<< __g_dim_assign, __t_dim_assign >>> ( k, __c_buffer, exec_count, __c_exec_list_x, __c_exec_list_y, __c_split, __c_fill );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );
        
        cudaMemcpy( &split_count, __c_split, sizeof( int ), cudaMemcpyDeviceToHost );
        cudaMemcpy( &fill_count, __c_fill, sizeof( int ), cudaMemcpyDeviceToHost );
        
        printf( " |-> ---- to_fill : %i ; ---- to split : %i \n", fill_count, split_count );
        
        set_grid_block_dim( &__g_dim_fill, &__t_dim_fill, CUDA_C, fill_count );
        set_grid_block_dim( &__g_dim_split, &__t_dim_split, CUDA_C, split_count );
        
        // fill labels
        fill_mask <<< __g_dim_fill, __t_dim_fill >>> ( k, __c_buffer, fill_count );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );
        
        // fill labels
        split_mask <<< __g_dim_split, __t_dim_split >>> ( k, __c_buffer, __c_exec_count, split_count, __prep_exec_list_x, __prep_exec_list_y );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );
        
        /*// check borders
        check_boundaries <<< __g_dim, __t_dim >>> ( __c_masks_pos, k, __c_buffer, exec_count, __c_exec_list_x, __c_exec_list_y );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );
        
        // fill labels
        fill_mask <<< __g_dim, __t_dim >>> ( __c_masks_pos, k, __c_buffer, __c_exec_count, exec_count, __c_exec_list_x, __c_exec_list_y, __prep_exec_list_x, __prep_exec_list_y );
        SAFE_KER( cudaPeekAtLastError() );
        SAFE_KER( cudaDeviceSynchronize() );*/
        
#if STEP > 1
        if( k == 0 ) {
#if VISU
            cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );
            cudaMemcpy( processed, __h_buffer->__proc, size * sizeof( int ), cudaMemcpyDeviceToHost );
            
            imagesc( std::string("labels"), labels, width, height );
            imagesc( std::string("processed"), processed, width, height );
            
            cvWaitKey( 0 );
#endif
            
            cudaMemcpy( &exec_count, __c_exec_count, sizeof( int ), cudaMemcpyDeviceToHost );
            cudaMemcpy( __c_exec_list_x, __prep_exec_list_x, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            cudaMemcpy( __c_exec_list_y, __prep_exec_list_y, sizeof( int ) * exec_count, cudaMemcpyDeviceToDevice );
            
            set_grid_block_dim( &__g_dim_assign, &__t_dim_assign, CUDA_C, exec_count );
            printf(" || -- iteration %i : %i \n", k, exec_count);
            printf(" |-> assign_last (%i ; %i,%i) \n", __g_dim_assign, __t_dim_assign, __count_last );
            
            // assign last
            assign_last <<< __g_dim_assign, dim3( __t_dim_assign, __count_last ) >>> ( __c_masks_pos, k, __c_buffer, exec_count, __c_exec_list_x, __c_exec_list_y );
            
            SAFE_KER( cudaPeekAtLastError() );
            SAFE_KER( cudaDeviceSynchronize() );
            
        }
#endif

        if(k > 0) {
            // reset __t_labels
            cudaMemcpy( __h_buffer->__t_labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToDevice );
            
            // update seeds
            update_seeds <<< __g_dim_sp, __t_dim_sp >>> ( __c_buffer );
            SAFE_KER( cudaPeekAtLastError() );
            SAFE_KER( cudaDeviceSynchronize() );
        
        }

#if OUTPUT_log
        st2 += now_ms() - lap;
#endif

#if VISU
        cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );
        cudaMemcpy( processed, __h_buffer->__proc, size * sizeof( int ), cudaMemcpyDeviceToHost );
        
        imagesc( std::string("labels"), labels, width, height );
        imagesc( std::string("processed"), processed, width, height );
        
        cvWaitKey( 0 );
#endif
        
        printf( " |-> ---- ---- ---- <-| \n" );
        printf( " \n" );
        
    }
    
    cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );
    
}
