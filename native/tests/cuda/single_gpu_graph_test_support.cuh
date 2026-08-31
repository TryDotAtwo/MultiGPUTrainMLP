#pragma once

// Test/tool-only private-state access. Do not export these diagnostics as a
// trainer API. The linked static archive supplies the remaining CUDA objects.
#include "../../cuda/single_gpu_trainer.cu"
#include "../../cuda/random_walk_kernel.cu"
#include "../../cuda/adamw.cu"
#include "mgt/puzzle_io.hpp"
#include "mgt_cuda/blas_stream.cuh"
#include <cuda.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mgt_graph_test {
void Check(bool ok, const char* where) {
    if (!ok) throw std::runtime_error(where);
}
void Cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(status));
}
template<class T> void Same(const T* a, const T* b, std::size_t n, const char* field) {
    std::vector<T> host_a(n), host_b(n);
    Cuda(cudaMemcpy(host_a.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost),field);
    Cuda(cudaMemcpy(host_b.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost),field);
    if (std::memcmp(host_a.data(),host_b.data(),n*sizeof(T)) != 0)
        throw std::runtime_error(std::string("byte mismatch: ")+field);
}
void Compare(mgt_cuda::SingleGpuTrainer* a, mgt_cuda::SingleGpuTrainer* b, bool exact) {
    const auto ptr=[](mgt_cuda::SingleGpuTrainer* t,std::uint64_t offset) {
        return static_cast<unsigned char*>(t->arena)+offset;
    };
    const auto& l=a->layout;
    Same(ptr(a,l.states),ptr(b,l.states),a->info.capacity_rows*sizeof(mgt::TrainStateStorage),"states");
    Same(ptr(a,l.labels),ptr(b,l.labels),a->info.capacity_rows*sizeof(float),"labels");
    Same(ptr(a,l.walk_meta),ptr(b,l.walk_meta),a->info.capacity_rows*sizeof(mgt::WalkMeta),"walk_meta");
    if (!exact) return; // Cross-CTA BN reduction order is not promised bitwise.
    for (auto offset : {l.weights,l.weight_grad,l.weight_m,l.weight_v})
        Same(ptr(a,offset),ptr(b,offset),l.parameter_count*sizeof(float),"weight/state");
    Same(ptr(a,l.weight_half),ptr(b,l.weight_half),l.parameter_count*sizeof(__half),"weight half");
    for (auto offset : {l.affine,l.affine_grad,l.affine_m,l.affine_v})
        Same(ptr(a,offset),ptr(b,offset),a->plan.trainable_count*sizeof(float),"affine/state");
    Same(ptr(a,l.running),ptr(b,l.running),a->plan.running_count*sizeof(float),"running");
    Same(ptr(a,l.outputs),ptr(b,l.outputs),a->info.capacity_rows*sizeof(float),"outputs");
    Same(ptr(a,l.loss),ptr(b,l.loss),sizeof(float),"loss");
}

inline mgt::PuzzleDefinition Puzzle() {
    mgt::PuzzleDefinition puzzle{};
    Check(mgt::LoadPuzzleDefinition(MGT_GRAPH_GROUP_JSON, MGT_GRAPH_TARGET_BIN,
        &puzzle)==mgt::Status::kOk, "puzzle load");
    Check(mgt::HasNonIdentityMove(puzzle), "identity-only puzzle");
    return puzzle;
}

struct TrainerOwner {
    mgt_cuda::SingleGpuTrainer* trainer=nullptr;
    void* workspace=nullptr;
    static constexpr std::size_t kWorkspaceBytes=4*1024*1024;
    TrainerOwner(unsigned rows, const mgt::PuzzleDefinition& puzzle, bool own_workspace) {
        using namespace mgt_cuda;
        SingleGpuTrainerCreateInfo info{};
        info.contract=mgt::OriginalP888SingleGpuContract(); info.capacity_rows=rows;
        info.adam={0,1,1e-4f,.9f,.999f,1e-8f,0}; info.puzzle=&puzzle;
        info.base_seed=0x0888000000000001ULL;
        try {
            Check(CreateSingleGpuTrainer(info,&trainer)==mgt::Status::kOk,"create");
            Check(PrepareSingleGpuTrainer(trainer)==mgt::Status::kOk,"prepare");
            if(own_workspace) {
                Cuda(cudaMalloc(&workspace,kWorkspaceBytes),"workspace allocation");
                Check(cublasSetWorkspace(trainer->blas,workspace,kWorkspaceBytes)==
                    CUBLAS_STATUS_SUCCESS,"workspace bind");
            }
        } catch(...) { Cleanup(); throw; }
    }
    TrainerOwner(const TrainerOwner&)=delete;
    TrainerOwner& operator=(const TrainerOwner&)=delete;
    void Cleanup() noexcept {
        if(trainer) mgt_cuda::DestroySingleGpuTrainer(&trainer);
        if(workspace) { cudaFree(workspace); workspace=nullptr; }
    }
    ~TrainerOwner() { Cleanup(); }
};

struct TrainGraph {
    mgt_cuda::SingleGpuTrainer* trainer;
    unsigned rows;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::array<cudaGraphNode_t,3> dynamic{};
    std::array<cudaKernelNodeParams,3> parameters{};
    std::map<int,unsigned> kinds;

    TrainGraph(mgt_cuda::SingleGpuTrainer* t,unsigned fixed_rows):trainer(t),rows(fixed_rows) {
        using namespace mgt_cuda;
        Check(t&&t->prepared&&fixed_rows&&fixed_rows<=t->info.capacity_rows,"graph shape");
        const auto old_sequence=t->sequence;
        const bool old_in_flight=t->in_flight;
        try {
            Cuda(cudaStreamBeginCapture(t->stream,cudaStreamCaptureModeGlobal),"capture begin");
            SingleGpuTrainStepTicket ticket{};
            const auto status=LaunchSingleGpuTrainStep(t,{rows,old_sequence+1,0,0},&ticket);
            const auto ended=cudaStreamEndCapture(t->stream,&graph);
            // Capture enqueues nothing. Restore the public launch bookkeeping.
            t->sequence=old_sequence; t->in_flight=old_in_flight;
            Check(status==mgt::Status::kOk,"capture body"); Cuda(ended,"capture end");
            std::size_t count=0;
            Cuda(cudaGraphGetNodes(graph,nullptr,&count),"node count");
            std::vector<cudaGraphNode_t> nodes(count);
            Cuda(cudaGraphGetNodes(graph,nodes.data(),&count),"nodes");
            const std::array<void*,3> functions{
                reinterpret_cast<void*>(RandomWalkKernel),
                reinterpret_cast<void*>(AdamWWithHalfMirrorKernel),
                reinterpret_cast<void*>(AdamWKernel)};
            std::array<CUfunction,3> driver_functions{};
            for(unsigned i=0;i<functions.size();++i) {
                cudaFunction_t function=nullptr;
                Cuda(cudaGetFuncBySymbol(&function,functions[i]),"kernel identity");
                driver_functions[i]=reinterpret_cast<CUfunction>(function);
            }
            for(auto node:nodes) {
                cudaGraphNodeType type;
                Cuda(cudaGraphNodeGetType(node,&type),"node type"); ++kinds[int(type)];
                if(type!=cudaGraphNodeTypeKernel) continue;
                // Driver queries also support cuBLAS's driver-loaded kernels;
                // runtime GetParams on those can return invalid-device-function.
                CUDA_KERNEL_NODE_PARAMS driver_params{};
                Check(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node),
                    &driver_params)==CUDA_SUCCESS,"driver kernel parameters");
                for(unsigned i=0;i<functions.size();++i) if(driver_params.func==driver_functions[i]) {
                    cudaKernelNodeParams params{};
                    Cuda(cudaGraphKernelNodeGetParams(node,&params),"owned kernel parameters");
                    Check(params.func==functions[i],"runtime/driver identity mismatch");
                    Check(!dynamic[i],"duplicate dynamic kernel");
                    Check(params.kernelParams&&!params.extra,"typed kernel arguments");
                    dynamic[i]=node; parameters[i]=params;
                }
            }
            Check(kinds[cudaGraphNodeTypeMemAlloc]==0&&kinds[cudaGraphNodeTypeMemFree]==0,
                "provided workspace was lost: allocation/free nodes remain");
            for(auto node:dynamic) Check(node!=nullptr,"missing dynamic kernel");
            Cuda(cudaGraphInstantiate(&executable,graph,0),"instantiate");
        } catch(...) {
            t->sequence=old_sequence; t->in_flight=old_in_flight;
            Cleanup(); throw;
        }
    }
    TrainGraph(const TrainGraph&)=delete;
    TrainGraph& operator=(const TrainGraph&)=delete;
    void Cleanup() noexcept {
        if(trainer) cudaStreamSynchronize(trainer->stream);
        if(executable) { cudaGraphExecDestroy(executable); executable=nullptr; }
        if(graph) { cudaGraphDestroy(graph); graph=nullptr; }
    }
    ~TrainGraph() { Cleanup(); }

    void Update(const mgt_cuda::SingleGpuTrainStepRequest& request,unsigned mask) {
        using namespace mgt_cuda;
        if(mask&1U) {
            auto params=parameters[0];
            std::array<void*,10> args{};
            std::copy_n(params.kernelParams,args.size(),args.begin());
            auto config=*static_cast<const RandomWalkKernelConfig*>(args[0]);
            auto epoch=request.epoch;
            auto step=request.optimizer_step;
            config.epoch_sample_offset=request.epoch_sample_offset;
            args[0]=&config; args[2]=&epoch; args[3]=&step;
            params.kernelParams=args.data();
            Cuda(cudaGraphExecKernelNodeSetParams(executable,dynamic[0],&params),"walk update");
        }
        for(unsigned i=1;i<3;++i) if(mask&(1U<<i)) {
            auto params=parameters[i];
            std::array<void*,6> args{};
            std::copy_n(params.kernelParams,i==1?6:5,args.begin());
            auto config=*static_cast<const AdamWKernelConfig*>(args[0]);
            config.step=request.optimizer_step;
            args[0]=&config; params.kernelParams=args.data();
            Cuda(cudaGraphExecKernelNodeSetParams(executable,dynamic[i],&params),"Adam update");
        }
    }

    mgt::Status Launch(const mgt_cuda::SingleGpuTrainStepRequest& request,
                       mgt_cuda::SingleGpuTrainStepTicket* ticket,unsigned update_mask=7) {
        if(!ticket||update_mask>7||request.active_rows!=rows||
            request.optimizer_step!=trainer->sequence+1||
            request.epoch_sample_offset>mgt::P888TrainingContract::kSamplesPerEpoch||
            rows>mgt::P888TrainingContract::kSamplesPerEpoch-request.epoch_sample_offset)
            return mgt::Status::kInvalidConfig;
        Update(request,update_mask);
        Cuda(cudaGraphLaunch(executable,trainer->stream),"graph launch");
        Cuda(cudaEventRecord(trainer->completion,trainer->stream),"completion event");
        trainer->sequence=request.optimizer_step; trainer->in_flight=true;
        *ticket={trainer->completion,trainer->sequence};
        return mgt::Status::kOk;
    }
};
} // namespace mgt_graph_test
