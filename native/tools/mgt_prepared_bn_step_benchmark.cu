#include "mgt/benchmark_snapshot.hpp"
#include "mgt/config.hpp"
#include "mgt/distributed_batch.hpp"
#include "mgt_cuda/prepared_p888_train_step.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {
template<class T> bool Alloc(T** p, std::uint64_t n) {
    return n && cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(T)) == cudaSuccess;
}
std::uint64_t Params(const mgt_cuda::CudaMlpShape& s) {
    return static_cast<std::uint64_t>(s.state_len)*s.state_value_pad*s.hd1+s.hd1+
        static_cast<std::uint64_t>(s.hd1)*s.hd2+s.hd2+
        static_cast<std::uint64_t>(s.residual_blocks)*2ULL*
            (static_cast<std::uint64_t>(s.hd2)*s.hd2+s.hd2)+
        static_cast<std::uint64_t>(s.hd2)*s.output_dim+s.output_dim;
}
bool Copy(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}
}

int main(int argc,char** argv) {
    if(argc!=9) {
        std::cerr<<"usage: device rank world id_file expected_local_rows global_rows warmup steps\n";
        return 2;
    }
    const auto device=static_cast<std::uint32_t>(std::strtoul(argv[1],nullptr,10));
    const auto rank=static_cast<std::uint32_t>(std::strtoul(argv[2],nullptr,10));
    const auto world=static_cast<std::uint32_t>(std::strtoul(argv[3],nullptr,10));
    const auto expected_rows=static_cast<std::uint32_t>(std::strtoul(argv[5],nullptr,10));
    const auto global_rows=static_cast<std::uint32_t>(std::strtoul(argv[6],nullptr,10));
    const auto warmup=static_cast<std::uint32_t>(std::strtoul(argv[7],nullptr,10));
    const auto steps=static_cast<std::uint32_t>(std::strtoul(argv[8],nullptr,10));
    mgt::DistributedBatchSlice slice{};
    if(!world||rank>=world||!steps||
       mgt::PartitionGlobalBatch(global_rows,world,rank,&slice)!=mgt::Status::kOk||
       slice.active_rows!=expected_rows||device>=world) return 2;
    if(cudaSetDevice(device)!=cudaSuccess) return 3;

    const mgt_cuda::CudaMlpShape shape{
        mgt::kStateLen,mgt::kStateValuePad,
        mgt::RoundUp(mgt::kHd1,mgt::kHiddenAlignment),
        mgt::RoundUp(mgt::kHd2,mgt::kHiddenAlignment),
        mgt::kResidualBlocks,mgt::kOutputDim};
    mgt::BatchNormTrainingPlan plan{};
    if(mgt::BuildBatchNormTrainingPlan(
           mgt::kHd1,mgt::kHd2,shape.hd1,shape.hd2,shape.residual_blocks,
           slice.active_rows,&plan)!=mgt::Status::kOk) return 4;
    const std::uint64_t params=Params(shape);
    const mgt::BenchmarkSnapshotShape snapshot_shape{
        shape.state_len,shape.state_value_pad,shape.output_dim,params,
        plan.logical_feature_count};
    mgt::BenchmarkMutableState host{};
    std::vector<mgt::TrainStateStorage> all_states;
    std::vector<float> all_labels;
    if(mgt::FillBenchmarkSnapshot(888,0,global_rows,snapshot_shape,&host,
                                  &all_states,&all_labels)!=mgt::Status::kOk) return 5;
    std::string snapshot_sha;
    if(mgt::CanonicalBenchmarkSnapshotSha256(host,all_states,all_labels,
                                             &snapshot_sha)!=mgt::Status::kOk) return 5;

    const std::uint64_t rows=slice.active_rows;
    const std::uint64_t h1=rows*shape.hd1,h2=rows*shape.hd2;
    const std::uint64_t fw_count=mgt_cuda::MlpBatchNormForwardWorkspaceFloats(
        shape,plan,slice.active_rows);
    const std::uint64_t affine_count=2ULL*plan.logical_feature_count;
    float *w,*wg,*wm,*wv,*aff,*ag,*am,*av,*running,*out,*fw,*loss,*dy,*block,*fc1,*res,*input;
    float *labels[2]; mgt::TrainStateStorage* states[2];
    if(!Alloc(&w,params)||!Alloc(&wg,params)||!Alloc(&wm,params)||!Alloc(&wv,params)||
       !Alloc(&aff,affine_count)||!Alloc(&ag,affine_count)||!Alloc(&am,affine_count)||
       !Alloc(&av,affine_count)||!Alloc(&running,affine_count)||
       !Alloc(&out,rows*shape.output_dim)||!Alloc(&fw,fw_count)||!Alloc(&loss,1)||
       !Alloc(&dy,rows*shape.output_dim)||!Alloc(&block,h2)||!Alloc(&fc1,h2)||
       !Alloc(&res,h2)||!Alloc(&input,h1)||!Alloc(&labels[0],rows*shape.output_dim)||
       !Alloc(&labels[1],rows*shape.output_dim)||!Alloc(&states[0],rows)||!Alloc(&states[1],rows)) return 6;
    const auto* host_states=all_states.data()+slice.global_offset;
    const auto* host_labels=all_labels.data()+slice.global_offset*shape.output_dim;
    const std::size_t state_bytes=rows*sizeof(mgt::TrainStateStorage);
    const std::size_t label_bytes=rows*shape.output_dim*sizeof(float);
    if(!Copy(w,host.weights.data(),params*sizeof(float))||
       !Copy(wg,host.weight_grad.data(),params*sizeof(float))||
       !Copy(wm,host.weight_m.data(),params*sizeof(float))||
       !Copy(wv,host.weight_v.data(),params*sizeof(float))||
       !Copy(aff,host.affine.data(),affine_count*sizeof(float))||
       !Copy(ag,host.affine_grad.data(),affine_count*sizeof(float))||
       !Copy(am,host.affine_m.data(),affine_count*sizeof(float))||
       !Copy(av,host.affine_v.data(),affine_count*sizeof(float))||
       !Copy(running,host.running.data(),affine_count*sizeof(float))||
       !Copy(states[0],host_states,state_bytes)||!Copy(states[1],host_states,state_bytes)||
       !Copy(labels[0],host_labels,label_bytes)||!Copy(labels[1],host_labels,label_bytes)) return 7;

    const mgt_cuda::MlpBatchNormStepBuffers buffers{
        w,wg,wm,wv,aff,ag,am,av,running,out,fw,loss,dy,block,fc1,res,input};
    const std::uint32_t supported[]={slice.active_rows};
    mgt_cuda::PreparedP888StrictRuntimeCreateInfo info{};
    info.shape=shape; info.capacity_rows=slice.active_rows;
    info.supported_active_rows=supported; info.supported_active_row_count=1;
    info.device_id=device; info.rank=rank; info.world=world;
    info.strict_nccl_id_file=argv[4]; info.buffers=buffers; info.batch_norm_plan=plan;
    info.adam={params,1,1e-4f,.9f,.999f,1e-8f,0.f};
    info.state_slots={states[0],states[1]}; info.label_slots={labels[0],labels[1]};
    mgt_cuda::PreparedP888TrainRuntime* runtime=nullptr;
    if(mgt_cuda::CreatePreparedP888StrictRuntime(info,&runtime)!=mgt::Status::kOk) return 8;
    mgt_cuda::PreparedTrainStepTicket ticket{};
    for(std::uint32_t i=0;i<warmup;++i) {
        const mgt_cuda::PreparedTrainStepRequest request{
            i&1U,slice.active_rows,global_rows,slice.global_offset,i+1,false};
        if(mgt_cuda::LaunchPreparedP888TrainStep(runtime,request,&ticket)!=mgt::Status::kOk) return 9;
    }
    cudaEvent_t start=nullptr,stop=nullptr;
    if(cudaEventCreate(&start)!=cudaSuccess||cudaEventCreate(&stop)!=cudaSuccess||
       mgt_cuda::RecordPreparedP888TrainEvent(runtime,start)!=mgt::Status::kOk) return 10;
    for(std::uint32_t i=0;i<steps;++i) {
        const std::uint64_t step=static_cast<std::uint64_t>(warmup)+i+1;
        const mgt_cuda::PreparedTrainStepRequest request{
            static_cast<std::uint32_t>(step&1U),slice.active_rows,global_rows,
            slice.global_offset,step,false};
        if(mgt_cuda::LaunchPreparedP888TrainStep(runtime,request,&ticket)!=mgt::Status::kOk) return 11;
    }
    if(mgt_cuda::RecordPreparedP888TrainEvent(runtime,stop)!=mgt::Status::kOk||
       cudaEventSynchronize(stop)!=cudaSuccess) return 12;
    float region_ms=0;
    if(cudaEventElapsedTime(&region_ms,start,stop)!=cudaSuccess) return 13;
    const double step_ms=region_ms/steps;
    const double dense_flops=6.0*static_cast<double>(params)*global_rows;
    const double pflops=dense_flops/(step_ms*1.0e12);
    std::cout<<std::setprecision(10)
      <<"{\"rank\":"<<rank<<",\"world\":"<<world
      <<",\"active_rows\":"<<slice.active_rows
      <<",\"global_rows\":"<<global_rows
      <<",\"global_offset\":"<<slice.global_offset
      <<",\"steps\":"<<steps<<",\"region_ms\":"<<region_ms
      <<",\"avg_step_ms\":"<<step_ms
      <<",\"estimated_dense_pflops\":"<<pflops
      <<",\"peak_fraction\":"<<(pflops/2.496)
      <<",\"snapshot_sha256\":\""<<snapshot_sha<<"\"}\n";
    return mgt_cuda::DestroyPreparedP888TrainRuntime(runtime)==mgt::Status::kOk?0:14;
}
