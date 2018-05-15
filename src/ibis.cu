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
    
    /*printf("step : %i\n", step);
    for( int i=0; i<masks->size_to_assign; i++)
        printf("( %i, %i ) ", to_assign_x[ i ], to_assign_y[ i ]);
        
    printf("\n");*/
    
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

void generate_coord_mask( mask_design* masks, mask_apply* masks_pos, int width, int height) {
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

    cudaMalloc( (void**) &__c_Xseeds_init, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__c_Yseeds_init, maxSPNumber*sizeof(float) );

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
    
    cudaMalloc( (void**) &__h_buffer->__seeds, maxSPNumber*sizeof(float) * 5 );
    cudaMalloc( (void**) &__h_buffer->__seeds_s, maxSPNumber*sizeof(float) * 6 );
    
    cudaMalloc( (void**) &__h_buffer->__xs_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ys_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__ls_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__as_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__bs_s, maxSPNumber*sizeof(float) );
    cudaMalloc( (void**) &__h_buffer->__adj_sp, size_roi*maxSPNumber*sizeof(int) );
    cudaMalloc( (void**) &__h_buffer->__c_adj, maxSPNumber*sizeof(int) );
    
}

IBIS::~IBIS() {
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

    cudaFree( __c_Xseeds_init );
    cudaFree( __c_Yseeds_init );
    
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
    
    SAFE_C( cudaFreeHost( __h_RGB ), "free RGB" );
    SAFE_C( cudaFreeHost( labels ), "free labels" );
    
    SAFE_C( cudaFreeHost( sp_count ), "free sp_count" );
    SAFE_C( cudaFreeHost( p_exec_count ), "free sp_count" );
    
    // cuda
    cudaFree( __h_buffer->__c_px );
    cudaFree( __h_buffer->__xs );
    cudaFree( __h_buffer->__ys );
    cudaFree( __h_buffer->__ls );
    cudaFree( __h_buffer->__as );
    cudaFree( __h_buffer->__bs );
    
    cudaFree( __h_buffer->__lab );
    cudaFree( __h_buffer->__seeds );
    cudaFree( __h_buffer->__seeds_s );
    
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
    cudaFree( __c_sp );
    
    cudaFree( __c_exec_list_x );
    cudaFree( __c_exec_list_y );
    
    cudaFree( __prep_exec_list_x );
    cudaFree( __prep_exec_list_y );
    
    cudaFree( __c_RGB );
    
    cudaFree( __prep_exec_list_y );
    cudaFree( __prep_exec_list_y );
    
    cudaStreamDestroy( stream1 );
    cudaStreamDestroy( stream2 );
    
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

                            if (nlabels[nindex] < 0 && initial_repartition[oindex] == initial_repartition[nindex])
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
        initial_repartition[i] = nlabels[i];

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
    
    //index_mask = 1;
    
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
    cudaMallocHost( (void**)&labels, sizeof(int) * size );

    // repartition of pixels at start
    initial_repartition = new int[size];

    // cuda alloc init data
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__labels, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__t_labels, sizeof( int ) * size ), "cudaMalloc");
    
    int fill_split_size = int( size / ( STEP * STEP ) );
    
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__to_fill, sizeof( int ) * fill_split_size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__to_fill_x, sizeof( int ) * fill_split_size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__to_fill_y, sizeof( int ) * fill_split_size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__to_split_x, sizeof( int ) * fill_split_size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__to_split_y, sizeof( int ) * fill_split_size ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__proc, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__init_repa, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__l_vec, sizeof( float ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__a_vec, sizeof( float ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__b_vec, sizeof( float ) * size ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__h_buffer->__lab, sizeof( float ) * size * 3 ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__c_exec_list_x, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__c_exec_list_y, sizeof( int ) * size ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__prep_exec_list_x, sizeof( int ) * size ), "cudaMalloc");
    SAFE_C( cudaMalloc( (void**) &__prep_exec_list_y, sizeof( int ) * size ), "cudaMalloc");
    
    SAFE_C( cudaMemcpy( __c_buffer, __h_buffer, sizeof( __c_ibis ), cudaMemcpyHostToDevice ), "cudaMalloc");
    
    SAFE_C( cudaMalloc( (void**) &__c_exec_count, sizeof( int ) ), "cudaMalloc" );
    SAFE_C( cudaMalloc( (void**) &__c_fill, sizeof( int ) ), "cudaMalloc" );
    SAFE_C( cudaMalloc( (void**) &__c_split, sizeof( int ) ), "cudaMalloc" );
    SAFE_C( cudaMalloc( (void**) &__c_sp, sizeof( int ) * 2 ), "cudaMalloc" );
    
    // RGB to LAB
    cudaMallocHost( (void**)&__h_RGB, sizeof(float) * size * 3 );
    
    cudaMallocHost( (void**)&sp_count, sizeof(int) * 2 );
    cudaMallocHost( (void**)&p_exec_count, sizeof(int) );
    
    SAFE_C( cudaMalloc( (void**) &__c_RGB, sizeof( float ) * size * 3 ), "cudaMalloc" );
    
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
    
    //int index = 26894;
    //printf(" -- ref -- (%i, %i, %i) = (%f, %f, %f)\n", img->ptr()[3*index + 2], img->ptr()[3*index + 1], img->ptr()[3*index], lvec[index], avec[index], bvec[index]);
    
}

void IBIS::process( cv::Mat* img ) {

#if OUTPUT_log
    double lap;
#endif
    int g_dim, t_dim;

    if( size == 0 ) {
        size = img->cols * img->rows;
        width = img->cols;
        height = img->rows;
        
        // initialise all the buffer and inner parameters
        init();

        // STEP 1 : initialize with fix grid seeds value
        initSeeds();
        
        cudaMemcpy( __h_buffer->__adj_sp, adjacent_sp, maxSPNumber * size_roi * sizeof( int ), cudaMemcpyHostToDevice );
        cudaMemcpy( __h_buffer->__c_adj, count_adjacent, maxSPNumber * sizeof( int ), cudaMemcpyHostToDevice );
        cudaMemcpy( __h_buffer->__init_repa, initial_repartition, size * sizeof( int ), cudaMemcpyHostToDevice );
        
        cudaMemcpy( __c_Xseeds_init, Xseeds_init, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
        cudaMemcpy( __c_Yseeds_init, Yseeds_init, maxSPNumber * sizeof( float ), cudaMemcpyHostToDevice );
        
        cudaStreamCreate( &stream1 );
        cudaStreamCreate( &stream2 );
        std::fill( labels, labels + size, -1 );
        
    }
    
    st2 = 0;
    st3 = 0;
    st4 = 0;
    
#if OUTPUT_log
    lap = now_ms();
#endif

    for (int i = 0; i < size * 3; i++)
        __h_RGB[ i ] = float( img->ptr()[ i ] );
    
    cudaMemcpy( __c_RGB, __h_RGB, sizeof(float) * size * 3, cudaMemcpyHostToDevice );
    
    set_grid_block_dim( &g_dim, &t_dim, 64, size );
    RGB2LAB <<< g_dim, t_dim, 0, stream1 >>> ( __c_RGB, __c_buffer, size );
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
    
    cudaMemcpyAsync( __h_buffer->__labels, labels, size * sizeof( int ), cudaMemcpyHostToDevice, stream2 );
    
    cudaMemcpyAsync( __h_buffer->__t_labels, labels, size * sizeof( int ), cudaMemcpyHostToDevice, stream2 );
    
    SAFE_KER( cudaStreamSynchronize( stream1 ) );
    SAFE_KER( cudaStreamSynchronize( stream2 ) );
    
#if VISU
    cudaMemsetAsync( __h_buffer->__proc, 0, size * sizeof( int ) );
#endif

    // reset var
    set_grid_block_dim( &g_dim, &t_dim, 64, SPNumber );
    __c_reset <<< g_dim, t_dim >>> ( __c_Xseeds_init, __c_Yseeds_init, __c_buffer, SPNumber, __c_sp, __c_exec_count );
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
    
#if OUTPUT_log
    st2 = now_ms() - lap;
    lap = now_ms();
#endif

    // STEP 2 : process IBIS
    mask_propagate_SP();
    
    cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );

#if OUTPUT_log
    st3 = now_ms() - lap;
    lap = now_ms();
#endif
    // STEP 3 : post processing
    memcpy( initial_repartition, labels, sizeof(int) * size );
    enforceConnectivity();

#if OUTPUT_log
    st4 = now_ms() - lap;
#endif

    // output log
#if OUTPUT_log
    printf("-----------------\n");
    printf("PERF_T %lf\n", st3+st4+st2);
    printf("gpu prec\t\t%lf\t ms\n", st2);
    printf("gpu exec\t\t%lf\t ms\n", st3);
    printf("cpu posc\t\t%lf\t ms\n", st4);
#endif

}

//--------------------------------------------------------------------------------CUDA

__global__ void __c_reset( float* __c_Xseeds_init, float* __c_Yseeds_init, __c_ibis* __c_buffer, int SPNumber, int* __c_sp, int* __c_exec_count ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if( index >= SPNumber )
        return;
    
    __c_buffer->__seeds[ 5*index ] = __c_Xseeds_init[ index ];
    __c_buffer->__seeds[ 5*index + 1 ] = __c_Yseeds_init[ index ];
    
    int index_xy = (int) __c_Yseeds_init[ index ] * __c_width + __c_Xseeds_init[ index ];
    
    __c_buffer->__seeds[ 5*index + 2 ] = __c_buffer->__lab[ 3*index_xy ];
    __c_buffer->__seeds[ 5*index + 3 ] = __c_buffer->__lab[ 3*index_xy + 1 ];
    __c_buffer->__seeds[ 5*index + 4 ] = __c_buffer->__lab[ 3*index_xy + 2 ];
    
    //reset seeds_s
    __c_buffer->__seeds_s[ 5*index + 0 ] = 0.f;
    __c_buffer->__seeds_s[ 5*index + 1 ] = 0.f;
    __c_buffer->__seeds_s[ 5*index + 2 ] = 0.f;
    __c_buffer->__seeds_s[ 5*index + 3 ] = 0.f;
    __c_buffer->__seeds_s[ 5*index + 4 ] = 0.f;
    __c_buffer->__seeds_s[ 5*index + 5 ] = 0.f;
    
    if( index == 0 ) {
        __c_sp[ 0 ] = 0;
        __c_sp[ 1 ] = 0;
        
        *__c_exec_count = 0;
        
    }
    
}

__global__ void assign_last( mask_apply* __c_masks_pos, int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int px_to_compute = blockIdx.y;
    
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
        
        if( __c_buffer->__labels[ index_xy ] >= 0 )
            return;
        
        int init_repart = __c_buffer->__init_repa[ index_xy ];
        
        float l = __c_buffer->__lab[ 3*index_xy ];
        float a = __c_buffer->__lab[ 3*index_xy + 1 ];
        float b = __c_buffer->__lab[ 3*index_xy + 2 ];
        
        int best_sp = 0;
        int index_sp;
        float dist_lab, dist_xy;
        float D = -1.f;
        float total_dist;
        
        for(int i=0; i<__c_buffer->__c_adj[ init_repart ]; i++) {
            index_sp = __c_buffer->__adj_sp[ size_roi * init_repart + i ];
            float sx = __c_buffer->__seeds[ 5*index_sp + 0 ];
            float sy = __c_buffer->__seeds[ 5*index_sp + 1 ];
            float sl = __c_buffer->__seeds[ 5*index_sp + 2 ];
            float sa = __c_buffer->__seeds[ 5*index_sp + 3 ];
            float sb = __c_buffer->__seeds[ 5*index_sp + 4 ];
            
            dist_lab = ( l - sl ) * ( l - sl ) +
                       ( a - sa ) * ( a - sa ) +
                       ( b - sb ) * ( b - sb );

            dist_xy = ( x - sx ) * ( x - sx ) +
                      ( y - sy ) * ( y - sy );

            total_dist = dist_lab + dist_xy * __c_invwt;

            if( total_dist < D || D < 0) {
                best_sp = index_sp;
                D = total_dist;

            }
            
        }
        
        // assign labels
        __c_buffer->__labels[ index_xy ] = best_sp;

#if VISU
        __c_buffer->__proc[ index_xy ] += 10;
#endif
    }
    
}

__global__ void update_seeds( int k, __c_ibis* __c_buffer, int* __c_sp, int* __c_exec_count ) {
    int i = CUDA_SP * blockIdx.x + threadIdx.x;
    
    if( i == 0 && k > 0 ) {
        __c_sp[0] = 0;
        __c_sp[1] = 0;
        
        *__c_exec_count = 0;
        
    }
    
    float inv;
    float count_px = __c_buffer->__seeds_s[ 6*i ];
    
    if( count_px > 0 ) {
        inv = 1.f / count_px;
        
        __c_buffer->__seeds[ 5*i + 0 ] = __c_buffer->__seeds_s[ 6*i + 1 ] * inv;
        __c_buffer->__seeds[ 5*i + 1 ] = __c_buffer->__seeds_s[ 6*i + 2 ] * inv;
        __c_buffer->__seeds[ 5*i + 2 ] = __c_buffer->__seeds_s[ 6*i + 3 ] * inv;
        __c_buffer->__seeds[ 5*i + 3 ] = __c_buffer->__seeds_s[ 6*i + 4 ] * inv;
        __c_buffer->__seeds[ 5*i + 4 ] = __c_buffer->__seeds_s[ 6*i + 5 ] * inv;
        
    }

}

__global__ void assign_px( int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int px_to_compute = blockIdx.y;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_exec_list_x[ index ];
    int y_ref = __c_exec_list_y[ index ];
    
    if( x_ref == 0 || y_ref == 0 )
        return;
    
    int x = x_ref + __design_assign_x[ __design_assign_start[ k ] + px_to_compute ];
    int y = y_ref + __design_assign_y[ __design_assign_start[ k ] + px_to_compute ];
    
    if( x >= 0 && x < __c_width && y >= 0 && y < __c_height ) {
        int index_xy = y * __c_width + x;
        
        int init_repart = __c_buffer->__init_repa[ index_xy ];
        
        float l = __c_buffer->__lab[ 3*index_xy ];
        float a = __c_buffer->__lab[ 3*index_xy + 1 ];
        float b = __c_buffer->__lab[ 3*index_xy + 2 ];
        
        int best_sp = 0;
        int index_sp;
        float dist_lab, dist_xy;
        float D = -1.f;
        float total_dist;
        
        // prepare shared buffer
        int nb_adj = __c_buffer->__c_adj[ init_repart ];
       
        for(int i=0; i<nb_adj; i++) {
            index_sp = __c_buffer->__adj_sp[ size_roi * init_repart + i ];
            
            float sx = __c_buffer->__seeds[ 5*index_sp + 0 ];
            float sy = __c_buffer->__seeds[ 5*index_sp + 1 ];
            float sl = __c_buffer->__seeds[ 5*index_sp + 2 ];
            float sa = __c_buffer->__seeds[ 5*index_sp + 3 ];
            float sb = __c_buffer->__seeds[ 5*index_sp + 4 ];
            
            dist_lab = ( l - sl ) * ( l - sl ) +
                       ( a - sa ) * ( a - sa ) +
                       ( b - sb ) * ( b - sb );

            dist_xy = ( x - sx ) * ( x - sx ) +
                      ( y - sy ) * ( y - sy );

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
        __c_buffer->__proc[ index_xy ] += 10;
#endif
    }
    
}

__global__ void check_boundaries( int k, __c_ibis* __c_buffer, int exec_count, int* __c_exec_list_x, int* __c_exec_list_y, int* __c_split, int* __c_sp ) {
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
        int base = atomicAdd( &__c_sp[0], 1 );
        __c_buffer->__to_split_x[ base ] = x_ref;
        __c_buffer->__to_split_y[ base ] = y_ref;
        
        //printf(" split[ %i ] = ( %i, %i ) \n ", base, x_ref, y_ref);
    }
    else {
        int base = atomicAdd( &__c_sp[1], 1 );
        __c_buffer->__to_fill_x[ base ] = x_ref;
        __c_buffer->__to_fill_y[ base ] = y_ref;
        __c_buffer->__to_fill[ base ] = ref;
        
        //printf(" fill[ %i ] = ( %i, %i ): %i\n", base, x_ref, y_ref, ref);
        
    }
    
}

__global__ void fill_mask_assign( int k, __c_ibis* __c_buffer, int fill_count ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index >= fill_count )
        return;
    
    int x_ref = __c_buffer->__to_fill_x[ index ];
    int y_ref = __c_buffer->__to_fill_y[ index ];
    int ref = __c_buffer->__to_fill[ index ];
    
    // fill limit
    int i_max = __design_size[ k ];
    int j_max = __design_size[ k ];
    int index_y;
    
    if( y_ref + i_max >= __c_height )
        i_max = __c_height - y_ref;
    
    if( x_ref + j_max >= __c_width )
        j_max = __c_width - x_ref;
    
     for( int i=y_ref; i<=y_ref+i_max; i++ ) {
        index_y = i * __c_width;
        
        for( int j=x_ref; j<=x_ref+j_max; j++ )
            __c_buffer->__labels[ index_y + j ] = ref;
        
    }
    
}

__global__ void fill_mask( int k, __c_ibis* __c_buffer, int exec_count ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_buffer->__to_fill_x[ index ];
    int y_ref = __c_buffer->__to_fill_y[ index ];
    int ref = __c_buffer->__to_fill[ index ];

    //printf(" <----- to fill[ %i ] = ( %i, %i ): %i\n", index, x_ref, y_ref, ref);

    int index_xy;
    float __tmp_l = 0.f;
    float __tmp_a = 0.f;
    float __tmp_b = 0.f;
    float __tmp_x = 0.f;
    float __tmp_y = 0.f;
    
    // fill limit
    int ii = 0;
    int i_max = __design_size[ k ];
    int j_max = __design_size[ k ];
    int index_y;
    
    if( y_ref + i_max >= __c_height )
        i_max = __c_height - y_ref;
    
    if( x_ref + j_max >= __c_width )
        j_max = __c_width - x_ref;
    
    for( int i=y_ref; i<=y_ref+i_max; i++ ) {
        index_y = i * __c_width;
        
        for( int j=x_ref; j<=x_ref+j_max; j++ ) {
            index_xy = index_y + j;
            
            //__c_buffer->__labels[ index_y + j ] = ref;
            __tmp_l += __c_buffer->__lab[ 3*index_xy ];
            __tmp_a += __c_buffer->__lab[ 3*index_xy + 1 ];
            __tmp_b += __c_buffer->__lab[ 3*index_xy + 2 ];
            __tmp_x += j;
            __tmp_y += i;
            ii++;
            
        }
    
    }

    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 0 ], ii );
    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 1 ], __tmp_x );
    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 2 ], __tmp_y );
    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 3 ], __tmp_l );
    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 4 ], __tmp_a );
    atomicAdd( &__c_buffer->__seeds_s[ 6*ref + 5 ], __tmp_b );

}

__global__ void split_mask( int k, __c_ibis* __c_buffer, int* __c_exec_count, int exec_count, int* __prep_exec_list_x, int* __prep_exec_list_y ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( index >= exec_count )
        return;
    
    int x_ref = __c_buffer->__to_split_x[ index ];
    int y_ref = __c_buffer->__to_split_y[ index ];
    
    //printf(" <----- to split[ %i / %i ] = ( %i, %i ) \n ", index, exec_count, x_ref, y_ref);
    
    //if( x_ref == -1 && y_ref == -1 )
    //    printf(" index: %i, split x: %i, y: %i \n", index, x_ref, y_ref);
    
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
            *__g_dim = *__g_dim + 1;
            
        *__t_dim = ref_t;
        
    }
    else {
        *__g_dim = 1;
        *__t_dim = value;
        
    }
    
}

void IBIS::mask_propagate_SP() {
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
    
#if KERNEL_log
        printf( " |-> ---- ---- ---- <-| \n" );
#endif

        if( k == index_mask-1 )
            exec_count = masks_pos[k].total_count;
        else
            exec_count = split_count * 4;
        
        set_grid_block_dim( &__g_dim_assign, &__t_dim_assign, 128, exec_count );
        
#if KERNEL_log
        printf(" || -- iteration %i : %i / %i = %f \n", k, exec_count, masks_pos[k].total_count, float(exec_count)/float(masks_pos[k].total_count)*100);
        printf("  |-> assign_px (%i, %i ; %i) => %i \n", __g_dim_assign, masks[ k ].size_to_assign, __t_dim_assign, __t_dim_assign * masks[ k ].size_to_assign );
#endif

        // assign px
        if( k == index_mask-1 )
            assign_px <<< dim3(__g_dim_assign, masks[ k ].size_to_assign), __t_dim_assign >>> ( k, __c_buffer, exec_count, masks_pos[k].apply_x, masks_pos[k].apply_y );
        else
            assign_px <<< dim3(__g_dim_assign, masks[ k ].size_to_assign), __t_dim_assign >>> ( k, __c_buffer, exec_count, __prep_exec_list_x, __prep_exec_list_y );
        
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
        
        // check borders        
#if KERNEL_log
        printf("  |-> check_boundaries (%i, %i) => %i \n", __g_dim_assign, __t_dim_assign, __t_dim_assign * __g_dim_assign );
#endif
        
        set_grid_block_dim( &__g_dim_assign, &__t_dim_assign, 256, exec_count );
        if( k == index_mask-1 )
            check_boundaries <<< __g_dim_assign, __t_dim_assign >>> ( k, __c_buffer, exec_count, masks_pos[k].apply_x, masks_pos[k].apply_y, __c_split, __c_sp );
        else
            check_boundaries <<< __g_dim_assign, __t_dim_assign >>> ( k, __c_buffer, exec_count, __prep_exec_list_x, __prep_exec_list_y, __c_split, __c_sp );
        
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
        cudaMemcpy( sp_count, __c_sp, sizeof( int )*2, cudaMemcpyDeviceToHost );
        split_count = sp_count[0];
        fill_count = sp_count[1];
        
        set_grid_block_dim( &__g_dim_fill, &__t_dim_fill, 16, fill_count );
        set_grid_block_dim( &__g_dim_split, &__t_dim_split, 256, split_count );
        
        // fill labels
#if KERNEL_log
        printf("  |-> fill_mask (%i ; %i) => %i \n", __g_dim_fill, __t_dim_fill, __t_dim_fill * __g_dim_fill );
#endif
        
        fill_mask_assign <<< __g_dim_fill, __t_dim_fill >>> ( k, __c_buffer, fill_count );
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
        
        // split masks
#if KERNEL_log
        printf("  |-> split_mask (%i ; %i) => %i \n", __g_dim_split, __t_dim_split, __t_dim_split * __g_dim_split );
#endif

        split_mask <<< __g_dim_split, __t_dim_split >>> ( k, __c_buffer, __c_exec_count, split_count, __prep_exec_list_x, __prep_exec_list_y );
        
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
        
        // sum seeds
        fill_mask <<< __g_dim_fill, __t_dim_fill >>> ( k, __c_buffer, fill_count );
#if KERNEL_log
        SAFE_KER( cudaPeekAtLastError() );
#endif
        
        //SAFE_KER( cudaStreamSynchronize( stream1 ) );
        //SAFE_KER( cudaStreamSynchronize( stream2 ) );
        
#if STEP > 1
        if( k == 0 ) {
#if VISU
            cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );
            cudaMemcpy( processed, __h_buffer->__proc, size * sizeof( int ), cudaMemcpyDeviceToHost );
            
            imagesc( std::string("labels"), labels, width, height );
            imagesc( std::string("processed"), processed, width, height );
            
            cvWaitKey( 0 );
#endif

            update_seeds <<< __g_dim_sp, __t_dim_sp >>> ( k, __c_buffer, __c_sp, __c_exec_count );
#if KERNEL_log
            SAFE_KER( cudaPeekAtLastError() );
#endif
            
            //cudaMemcpy( &exec_count, __c_exec_count, sizeof( int ), cudaMemcpyDeviceToHost );
            
            set_grid_block_dim( &__g_dim_assign, &__t_dim_assign, 64, split_count );
#if KERNEL_log
            printf("  |-> assign_last (%i ; %i,%i) \n", __g_dim_assign, __t_dim_assign, __count_last );
#endif
            
            // assign last
            assign_last <<< dim3(__g_dim_assign, __count_last ), __t_dim_assign >>> ( __c_masks_pos, k, __c_buffer, split_count, __prep_exec_list_x, __prep_exec_list_y );
            
#if KERNEL_log
            SAFE_KER( cudaPeekAtLastError() );
            //SAFE_KER( cudaDeviceSynchronize() );
#endif
            
        }
#endif
        if(k > 0) {
            // reset __t_labels
            cudaMemcpy( __h_buffer->__t_labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToDevice );
            
            // sync
            //SAFE_KER( cudaStreamSynchronize( stream1 ) );
            //SAFE_KER( cudaStreamSynchronize( stream2 ) );
            
            // update seeds
            update_seeds <<< __g_dim_sp, __t_dim_sp >>> ( k, __c_buffer, __c_sp, __c_exec_count );
#if KERNEL_log
            SAFE_KER( cudaPeekAtLastError() );
#endif
            
        }
        
#if VISU
        cudaMemcpy( labels, __h_buffer->__labels, size * sizeof( int ), cudaMemcpyDeviceToHost );
        cudaMemcpy( processed, __h_buffer->__proc, size * sizeof( int ), cudaMemcpyDeviceToHost );
        
        imagesc( std::string("labels"), labels, width, height );
        imagesc( std::string("processed"), processed, width, height );
        
        cvWaitKey( 0 );
#endif
        
#if KERNEL_log
        printf( " |-> ---- ---- ---- <-| \n" );
        printf( " \n" );
#endif
        
    }
    
}
