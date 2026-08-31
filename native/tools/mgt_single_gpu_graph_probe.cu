// Experimental scheduling probe, not a public trainer mode. Keep its runtime
// and model identical across modes; only workspace ownership and replay differ.
#include "../tests/cuda/single_gpu_graph_test_support.cuh"

#include <chrono>
#include <iostream>
#include <memory>

using namespace mgt_graph_test;
using namespace mgt_cuda;

std::uint32_t Number(const char* text) {
    Check(text&&*text&&*text!='-',"unsigned integer required");
    char* end=nullptr;
    const auto value=std::strtoull(text,&end,10);
    Check(end&&!*end&&value<=UINT32_MAX,"integer out of range");
    return static_cast<std::uint32_t>(value);
}

int main(int argc,char** argv) {
    try {
        Check(argc==7,"usage: batch warmup steps group_json target_bin eager-default|eager-workspace|graph");
        const auto batch=Number(argv[1]),warmup=Number(argv[2]),steps=Number(argv[3]);
        const std::string mode=argv[6];
        Check(batch&&steps&&(mode=="eager-default"||mode=="eager-workspace"||mode=="graph"),"invalid arguments");
        Check((std::uint64_t(warmup)+steps)*batch<=mgt::P888TrainingContract::kSamplesPerEpoch,
            "benchmark requests cross epoch boundary");
        mgt::PuzzleDefinition puzzle{};
        Check(mgt::LoadPuzzleDefinition(argv[4],argv[5],&puzzle)==mgt::Status::kOk,"puzzle load");
        Check(mgt::HasNonIdentityMove(puzzle),"identity-only move set");
        TrainerOwner owner(batch,puzzle,mode!="eager-default");
        std::unique_ptr<TrainGraph> graph;
        const auto setup_begin=std::chrono::steady_clock::now();
        if(mode=="graph") graph=std::make_unique<TrainGraph>(owner.trainer,batch);
        const auto setup_end=std::chrono::steady_clock::now();
        const auto allocations=SingleGpuTrainerAllocationCountForTest();
        SingleGpuTrainStepTicket ticket{};
        std::uint64_t sequence=0;
        auto launch=[&] {
            ++sequence;
            const SingleGpuTrainStepRequest request{batch,sequence,0,(sequence-1)*batch};
            const auto status=graph?graph->Launch(request,&ticket):
                LaunchSingleGpuTrainStep(owner.trainer,request,&ticket);
            Check(status==mgt::Status::kOk,"train step");
        };
        for(unsigned i=0;i<warmup;++i) launch();
        if(warmup) Cuda(cudaEventSynchronize(ticket.completion_event),"warmup completion");
        const auto begin=std::chrono::steady_clock::now();
        for(unsigned i=0;i<steps;++i) launch();
        Cuda(cudaEventSynchronize(ticket.completion_event),"timed completion");
        const auto end=std::chrono::steady_clock::now();
        Check(SingleGpuTrainerAllocationCountForTest()==allocations,"arena allocation in loop");
        SingleGpuTrainerMetrics metrics{};
        Check(ReadSingleGpuMetrics(owner.trainer,&metrics)==mgt::Status::kOk,"metrics");
        Check(metrics.optimizer_step==sequence&&std::isfinite(metrics.loss),"invalid metrics");
        cudaDeviceProp device{};
        Cuda(cudaGetDeviceProperties(&device,0),"device properties");
        const double step_ms=std::chrono::duration<double,std::milli>(end-begin).count()/steps;
        const double setup_ms=std::chrono::duration<double,std::milli>(setup_end-setup_begin).count();
        const std::size_t workspace_bytes=owner.workspace?TrainerOwner::kWorkspaceBytes:0;
        std::cout << "{\"gpu\":\"" << device.name << "\",\"arch\":" << device.major*10+device.minor
                  << ",\"mode\":\"" << mode << "\",\"batch\":" << batch << ",\"warmup\":" << warmup
                  << ",\"steps\":" << steps << ",\"step_ms\":" << step_ms << ",\"samples_s\":"
                  << batch*1000.0/step_ms << ",\"memory_bytes\":" << owner.trainer->layout.bytes+workspace_bytes
                  << ",\"arena_bytes\":" << owner.trainer->layout.bytes << ",\"workspace_bytes\":" << workspace_bytes
                  << ",\"capture_instantiate_ms\":" << setup_ms << ",\"loss\":" << metrics.loss
                  << ",\"status\":\"ok\"}\n";
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL graph benchmark: %s\n",e.what()); return 1; }
}
