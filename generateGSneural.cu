#include  <stdexcept>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/AccumulateType.h>

#include <assert.h>
#include <vector>
#include <torch/torch.h>


# define blockDIM 256
# define debug 0
# define PI 3.14159

template <typename scalar_t>
__global__ void simpleIdx_kernel(
    const int width, //输出的宽度
    const uint64_t num_elements,
    scalar_t *indata,
    int64_t *maskIdx,
    scalar_t *outdata){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements){
        int out_row = idx / width;
        int col = idx % width;
        long in_row = maskIdx[out_row];
        outdata[out_row * width + col] = indata[in_row * width + col];
    }

}

void simpleIdx(torch::Tensor indata, 
                torch::Tensor maskIdx,
                torch::Tensor outdata){
        const int width = indata.size(1);
        const int height = maskIdx.size(0);
        const uint64_t num_elements = width * height;
        const uint32_t gridDIM = (height*width + blockDIM-1)/blockDIM;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
         indata.scalar_type(), "simpleIdx_kernel", ([&] {
                simpleIdx_kernel<scalar_t><<<gridDIM, blockDIM>>>
                (
                width, //输出的宽度
                num_elements,
                indata.data_ptr<scalar_t>(),
                maskIdx.data_ptr<int64_t>(),
                outdata.data_ptr<scalar_t>());
     }));
}





torch::Tensor simpleMask(torch::Tensor indata, 
                torch::Tensor maskIdx){
                
        const int width = indata.numel()/indata.size(0);
        const int height = maskIdx.size(0);
        
        // Get Input Tensor Shape info
        auto orig_shape = indata.sizes(); // auto -> c10::IntArrayRef

        /* Clone original shape details into another instance*/ 
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
        /* Modify first entry i.e. 0th position item */ 
        out_shape[0] = height;
        torch::Tensor outdata = torch::empty({out_shape}, indata.options());

        const uint64_t num_elements = width * height;

        const uint32_t gridDIM = (height*width + blockDIM-1)/blockDIM;
        AT_DISPATCH_ALL_TYPES_AND_HALF(
         indata.scalar_type(), "simpleIdx_kernel", ([&] {
                simpleIdx_kernel<scalar_t><<<gridDIM, blockDIM>>>
                (
                width, //输出的宽度
                num_elements,
                indata.data_ptr<scalar_t>(),
                maskIdx.data_ptr<int64_t>(),
                outdata.data_ptr<scalar_t>());
     }));
     return outdata;
}

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, const int pack_size>
union Pack {
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

__global__ void ob_property_kernel(const int rows,
                                const int width,
                                float* indata,
                                float* anchors,
                               float* obview,
                               int64_t *maskIdx,
                               float* ob_dist,
                               float* center){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < rows){
        int out_row = idx;
        long in_row = maskIdx[out_row];

        float anchor[3];
        float view[3];
        float dist = 0;
        for (int i = 0; i < width; i++){
            anchor[i] = indata[in_row * width + i];
            view[i] = anchor[i] - center[i];
            dist += view[i] * view[i];
        }
        dist = sqrtf(dist);
        for (int i = 0; i < width; i++){
            view[i] /= dist;
            anchors[out_row*width + i] = anchor[i];
            obview[out_row*width + i] = view[i]; 
        }
       
        ob_dist[idx] = dist;
    }
}

/*
equals to:
    anchor = fusedKernels.simpleMask(anchor, visible_idx)
    # get view properties for anchor
    ob_view = anchor - camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist
*/
std::vector<torch::Tensor>  ob_property(torch::Tensor indata, torch::Tensor maskIdx, torch::Tensor center, int dim){

    if(dim != 1){
      throw std::invalid_argument("Dimension normalization is only supported for a value of 1.");
    }

    const int width = indata.numel()/indata.size(0);
    const int height = maskIdx.size(0);

    if(width != 3){
      throw std::invalid_argument("only support width of 3.");
    }
    
    // Get Input Tensor Shape info
    auto orig_shape = indata.sizes(); // auto -> c10::IntArrayRef

    /* Clone original shape details into another instance*/ 
    std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end());
    /* Modify first entry i.e. 0th position item */ 
    out_shape[0] = height;
    torch::Tensor anchor = torch::empty({out_shape}, indata.options());
    torch::Tensor ob_view = torch::empty({out_shape}, indata.options());
    out_shape[1] = 1;
    torch::Tensor ob_dist = torch::empty({out_shape}, indata.options());

    const uint32_t gridDIM = (height + blockDIM-1)/blockDIM;

    ob_property_kernel<<<gridDIM, blockDIM>>>(
        height,
        width,
        reinterpret_cast<float*>(indata.data_ptr()),
        reinterpret_cast<float*>(anchor.data_ptr()), 
        reinterpret_cast<float*>(ob_view.data_ptr()),
        maskIdx.data_ptr<int64_t>(),
        ob_dist.data_ptr<float>(),
        center.data_ptr<float>());


    return {anchor, ob_view, ob_dist};

}

template <typename scalar_t>
__global__ void catRepeatMaskSplit_kernel(
    const int width1, //输出1的宽度
    const int width2, //输出2的宽度
    const uint64_t num_elements,
    const int repeat,
    scalar_t *in1,
    scalar_t *in2,
    int64_t *maskIdx,
    scalar_t *out1,
    scalar_t *out2){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements){
        int out_row = idx / (width1 + width2);
        int col = idx % (width1 + width2);
        long in_row = maskIdx[out_row]/repeat;
        if (col < width1){
          out1[out_row * width1 + col] = in1[in_row * width1 + col];
        }
        else{
          out2[out_row * width2 + (col-width1)] = in2[in_row * width2 + (col-width1)];
        }
    }

}

void catRepeatMaskSplit(
    torch::Tensor in1, 
    torch::Tensor in2,
    int repeat,
    torch::Tensor maskIdx,
    torch::Tensor out1,
    torch::Tensor out2){

    const int width1 = in1.size(1);
    const int width2 = in2.size(1);
    const int height = maskIdx.size(0);
    const uint64_t num_elements = (width1 + width2) * height;
    const uint32_t gridDIM = (num_elements + blockDIM-1)/blockDIM;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in1.scalar_type(), "catRepeatMaskSplit_kernel", ([&] {
            catRepeatMaskSplit_kernel<scalar_t><<<gridDIM, blockDIM>>>
            (
            width1, 
            width2, 
            num_elements,
            repeat,
            in1.data_ptr<scalar_t>(),
            in2.data_ptr<scalar_t>(),
            maskIdx.data_ptr<int64_t>(),
            out1.data_ptr<scalar_t>(),
            out2.data_ptr<scalar_t>());
     }));
}


template <typename scalar_t>
__global__ void RepeatMask_kernel(
    const int width1, //输出1的宽度
    const uint64_t num_elements,
    const int repeat,
    scalar_t *in1,
    int64_t *maskIdx,
    scalar_t *out1){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements){
        int out_row = idx / (width1);
        int col = idx % (width1);
        long in_row = maskIdx[out_row]/repeat;

        out1[out_row * width1 + col] = in1[in_row * width1 + col];
    }

}

/*
EQUALS TO:
  concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
  masked = concatenated_repeated[mask]
*/
void RepeatMask(
    torch::Tensor in1, 
    int repeat,
    torch::Tensor maskIdx,
    torch::Tensor out1){

    const int width1 = in1.size(1);
    const int height = maskIdx.size(0);
    const uint64_t num_elements = (width1) * height;
    const uint32_t gridDIM = (num_elements + blockDIM-1)/blockDIM;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in1.scalar_type(), "RepeatMask_kernel", ([&] {
            RepeatMask_kernel<scalar_t><<<gridDIM, blockDIM>>>
            (
            width1, 
            num_elements,
            repeat,
            in1.data_ptr<scalar_t>(),
            maskIdx.data_ptr<int64_t>(),
            out1.data_ptr<scalar_t>());
     }));
}


template <typename scalar_t>
__global__ void MaskPostProcessColor_kernel(
    const int width1, //输出1的宽度
    const uint64_t num_elements,
    const float plus,
    const float multiply,
    scalar_t *in1,
    int64_t *maskIdx,
    scalar_t *out1){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements){
        int out_row = idx / (width1);
        int col = idx % (width1);
        long in_row = maskIdx[out_row];
        // override_color_old = (color - 0.2)*1.5
        out1[out_row * width1 + col] = (in1[in_row * width1 + col]+plus)*multiply;
    }

}

/*
EQUALS TO:
  color = color[mask]
  override_color_old = (color - 0.2)*1.5
*/
void MaskPostProcessColor(
    torch::Tensor in1, 
    torch::Tensor maskIdx,
    torch::Tensor out1,
    const float plus,
    const float multiply){

    const int width1 = in1.size(1);
    const int height = maskIdx.size(0);
    const uint64_t num_elements = (width1) * height;
    const uint32_t gridDIM = (num_elements + blockDIM-1)/blockDIM;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      in1.scalar_type(), "MaskPostProcessColor_kernel", ([&] {
            MaskPostProcessColor_kernel<scalar_t><<<gridDIM, blockDIM>>>
            (
            width1, 
            num_elements,
            plus,
            multiply,
            in1.data_ptr<scalar_t>(),
            maskIdx.data_ptr<int64_t>(),
            out1.data_ptr<scalar_t>());
     }));
}

template <typename scalar_t>
__global__ void RepeatMaskPostProcessOffsets_kernel(
    const int repeat,
    const int w_out,
    const int w_scale,
    const uint64_t num_elements,
    scalar_t *grid_xyz,
    scalar_t *offsets,
    scalar_t *scaling_repeat,
    int64_t *maskIdx,
    scalar_t *xyz){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_elements){
        int out_row = idx / (w_out);
        int col = idx % (w_out);
        long in_row = maskIdx[out_row];
        long in_row_repeat = in_row/repeat;

        scalar_t offset = offsets[in_row* w_out + col];
        scalar_t repeat_xyz = grid_xyz[in_row_repeat * w_out + col];
        scalar_t scale = scaling_repeat[out_row * w_scale + col];
        // scalar_t scale = 0.;
        xyz[out_row * w_out + col] = repeat_xyz + offset*scale ;
    }

}


/*
EQUALS TO:
    repeat_xyz = repeat(grid_xyz, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    repeat_xyz = repeat_xyz[mask]
    offsets = offsets[mask]

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_xyz + offsets

*/
void RepeatMaskPostProcessOffsets(
    torch::Tensor grid_xyz,
    torch::Tensor offsets,
    torch::Tensor scaling_repeat,
    torch::Tensor maskIdx,
    torch::Tensor xyz,
    const int repeat){

    const int w_out = grid_xyz.size(1);
    const int height = maskIdx.size(0);
    const int w_scale = scaling_repeat.size(1);

    const uint64_t num_elements = (w_out) * height;
    const uint32_t gridDIM = (num_elements + blockDIM-1)/blockDIM;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grid_xyz.scalar_type(), "RepeatMaskPostProcessOffsets_kernel", ([&] {
            RepeatMaskPostProcessOffsets_kernel<scalar_t><<<gridDIM, blockDIM>>>
            (
            repeat,
            w_out, 
            w_scale,
            num_elements,
            grid_xyz.data_ptr<scalar_t>(),
            offsets.data_ptr<scalar_t>(),
            scaling_repeat.data_ptr<scalar_t>(),
            maskIdx.data_ptr<int64_t>(),
            xyz.data_ptr<scalar_t>());
     }));
}

__global__ void SelfContainedFeat_kernel(
    const int width, //输出的宽度
    const int height,
    const uint64_t num_elements,
    float *feat,
    float *bank_weight){

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float feats[blockDIM];
    if (idx < num_elements){
        int row = idx/width;
        int col = idx%width;
        float3 weights = reinterpret_cast<float3*> (bank_weight + row*3) [0];

        feats[idx%(blockDIM)] = feat[idx];
        __syncthreads();
        int srow = idx%(blockDIM)/width; //row in shared memory
        feat[row * width + col] = feats[srow*width + (col*4)%width] * weights.x
                                + feats[srow*width + (col*2)%width] * weights.y
                                + feats[srow*width + col]*weights.z;
    }
}
/*
EQUALS TO:
    feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
*/
void SelfContainedFeat(torch::Tensor feat, 
                torch::Tensor bank_weight){
        const int width = feat.size(1);
        const int height = feat.size(0);
        const uint64_t num_elements = height*width;
        const uint32_t gridDIM = (num_elements+ blockDIM -1)/blockDIM;

        SelfContainedFeat_kernel<<<gridDIM, blockDIM>>>
        (
        width, //输出的宽度
        height,
        num_elements,
        feat.data_ptr<float>(),
        bank_weight.data_ptr<float>());
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simpleIdx", &simpleIdx, "simpleIdx (CUDA)");
    m.def("simpleMask", &simpleMask, "simpleMask (CUDA)");
    m.def("ob_property", &ob_property, "ob_property (CUDA)");
    m.def("catRepeatMaskSplit", &catRepeatMaskSplit, "catRepeatMask (CUDA)");
    m.def("RepeatMask", &RepeatMask, "catRepeatMask (CUDA)");
    m.def("MaskPostProcessColor", &MaskPostProcessColor, "mask and post-process color (CUDA)");
    m.def("RepeatMaskPostProcessOffsets", &RepeatMaskPostProcessOffsets, "fuse mask and post-process offsets (CUDA)");

    m.def("SelfContainedFeat", &SelfContainedFeat, "SelfContainedFeat (CUDA)");

}
