#include "single_gpu_graph_test_support.cuh"

using namespace mgt_graph_test;
using namespace mgt_cuda;

void NegativeControls() {
    const auto puzzle=Puzzle();
    for(unsigned omitted=0;omitted<3;++omitted) {
        TrainerOwner a(4,puzzle,true),b(4,puzzle,true);
        TrainGraph graph(a.trainer,4);
        SingleGpuTrainStepTicket ticket{};
        for(unsigned step=1;step<=2;++step) {
            const SingleGpuTrainStepRequest request{4,step,step-1ULL,step==1?0ULL:12ULL};
            Check(graph.Launch(request,&ticket,7U&~(1U<<omitted))==mgt::Status::kOk,"negative replay");
            Check(LaunchSingleGpuTrainStep(b.trainer,request,&ticket)==mgt::Status::kOk,"negative reference");
            Cuda(cudaStreamSynchronize(a.trainer->stream),"negative graph sync");
            Cuda(cudaStreamSynchronize(b.trainer->stream),"negative reference sync");
            if(step==1) Compare(a.trainer,b.trainer,true);
        }
        bool detected=false;
        try { Compare(a.trainer,b.trainer,true); }
        catch(const std::runtime_error& e) {
            const std::array<const char*,3> expected{
                "byte mismatch: states","byte mismatch: weight/state","byte mismatch: affine/state"};
            Check(std::string(e.what())==expected[omitted],"unexpected negative-control failure");
            detected=true;
        }
        Check(detected,"stale node update was not detected");
        std::printf("PASS negative control omitted_node=%u\n",omitted);
    }
}

void Run(unsigned rows) {
    const auto puzzle=Puzzle();
    TrainerOwner a(rows,puzzle,true),b(rows,puzzle,true);
    TrainGraph graph(a.trainer,rows);
    const auto allocations=SingleGpuTrainerAllocationCountForTest();
    SingleGpuTrainStepTicket ticket{};
    for(unsigned pass=0;pass<4;++pass) {
        SingleGpuTrainStepRequest request{rows,pass+1ULL,pass/2ULL,
            pass==3?mgt::P888TrainingContract::kSamplesPerEpoch-rows:rows*(pass+1ULL)};
        Check(graph.Launch(request,&ticket)==mgt::Status::kOk,"graph replay");
        Cuda(cudaEventSynchronize(ticket.completion_event),"graph completion");
        Check(ticket.sequence==request.optimizer_step,"graph ticket sequence");
        Check(LaunchSingleGpuTrainStep(b.trainer,request,&ticket)==mgt::Status::kOk,"reference");
        Cuda(cudaEventSynchronize(ticket.completion_event),"reference completion");
        Compare(a.trainer,b.trainer,rows<=256);
        SingleGpuTrainerMetrics metrics{};
        Check(ReadSingleGpuMetrics(a.trainer,&metrics)==mgt::Status::kOk,"metrics");
        Check(metrics.optimizer_step==request.optimizer_step&&std::isfinite(metrics.loss),"metrics sequence/finite");
        std::printf("PASS replay rows=%u step=%llu epoch=%llu offset=%llu exact_model=%u loss=%.9g\n",
            rows,(unsigned long long)request.optimizer_step,(unsigned long long)request.epoch,
            (unsigned long long)request.epoch_sample_offset,rows<=256,metrics.loss);
    }
    Check(SingleGpuTrainerAllocationCountForTest()==allocations,"arena allocation in replay");
    const auto saved=ticket;
    for(auto request:{SingleGpuTrainStepRequest{rows,7,0,0},
                      SingleGpuTrainStepRequest{rows+1,5,0,0},
                      SingleGpuTrainStepRequest{rows,5,0,mgt::P888TrainingContract::kSamplesPerEpoch},
                      SingleGpuTrainStepRequest{rows,5,0,UINT64_MAX}}) {
        Check(graph.Launch(request,&ticket)==mgt::Status::kInvalidConfig,"invalid replay accepted");
        Check(a.trainer->sequence==4&&ticket.sequence==saved.sequence&&
            ticket.completion_event==saved.completion_event,"invalid request mutated state");
    }
    // Enqueue several requests without synchronizing. Node updates must affect
    // subsequent launches only, not previously enqueued graph executions.
    for(unsigned step=5;step<=8;++step) {
        const SingleGpuTrainStepRequest request{rows,step,step,rows*(step-5ULL)};
        Check(graph.Launch(request,&ticket)==mgt::Status::kOk,"queued replay");
        Check(LaunchSingleGpuTrainStep(b.trainer,request,&ticket)==mgt::Status::kOk,"queued reference");
    }
    Cuda(cudaStreamSynchronize(a.trainer->stream),"queued graph sync");
    Cuda(cudaStreamSynchronize(b.trainer->stream),"queued reference sync");
    Compare(a.trainer,b.trainer,rows<=256);
    std::printf("PASS queued replay rows=%u steps=5..8\n",rows);
}

int main(int argc,char** argv) {
    try {
        Check(argc<=2,"usage: [rows]");
        const unsigned rows=argc==2?std::strtoul(argv[1],nullptr,10):4;
        Check(rows>0&&rows<=4096,"test rows range");
        if(rows==4) NegativeControls();
        Run(rows);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL graph replay: %s\n",e.what()); return 1; }
}
