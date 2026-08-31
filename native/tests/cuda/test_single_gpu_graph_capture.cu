#include "single_gpu_graph_test_support.cuh"

using namespace mgt_graph_test;

void RunCapture(unsigned rows, bool warm, bool own_workspace, const mgt::PuzzleDefinition& puzzle) {
    using namespace mgt_cuda;
    SingleGpuTrainerCreateInfo info{};
    info.contract=mgt::OriginalP888SingleGpuContract(); info.capacity_rows=rows;
    info.adam={0,1,1e-4f,.9f,.999f,1e-8f,0}; info.puzzle=&puzzle;
    info.base_seed=0x0888000000000001ULL;
    SingleGpuTrainer* graph_trainer=nullptr;
    SingleGpuTrainer* eager_trainer=nullptr;
    cudaGraph_t retained=nullptr;
    cudaGraphExec_t executable=nullptr;
    cudaGraph_t pending=nullptr;
    void* workspace_a=nullptr;
    void* workspace_b=nullptr;
    auto cleanup=[&] {
        if(graph_trainer) cudaStreamSynchronize(graph_trainer->stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(pending) cudaGraphDestroy(pending);
        if(retained) cudaGraphDestroy(retained);
        if(graph_trainer) DestroySingleGpuTrainer(&graph_trainer);
        if(eager_trainer) DestroySingleGpuTrainer(&eager_trainer);
        if(workspace_a) cudaFree(workspace_a);
        if(workspace_b) cudaFree(workspace_b);
    };
    try {
        Check(CreateSingleGpuTrainer(info,&graph_trainer)==mgt::Status::kOk,"graph create");
        Check(CreateSingleGpuTrainer(info,&eager_trainer)==mgt::Status::kOk,"eager create");
        Check(PrepareSingleGpuTrainer(graph_trainer)==mgt::Status::kOk,"graph prepare");
        Check(PrepareSingleGpuTrainer(eager_trainer)==mgt::Status::kOk,"eager prepare");
        if(own_workspace) {
            Cuda(cudaMalloc(&workspace_a,4*1024*1024),"workspace A");
            Cuda(cudaMalloc(&workspace_b,4*1024*1024),"workspace B");
            Check(cublasSetWorkspace(graph_trainer->blas,workspace_a,4*1024*1024)==CUBLAS_STATUS_SUCCESS,"bind workspace A");
            Check(cublasSetWorkspace(eager_trainer->blas,workspace_b,4*1024*1024)==CUBLAS_STATUS_SUCCESS,"bind workspace B");
        }
        SingleGpuTrainStepTicket ticket{};
        std::uint64_t sequence=0;
        if(warm) {
            for(auto* trainer : {graph_trainer,eager_trainer}) {
                Check(LaunchSingleGpuTrainStep(trainer,{rows,1,0,0},&ticket)==mgt::Status::kOk,"warm step");
                Cuda(cudaEventSynchronize(ticket.completion_event),"warm completion");
            }
            sequence=1;
        }
        for(unsigned pass=0;pass<2;++pass) {
            ++sequence;
            // The second replay crosses an epoch and changes the sample offset;
            // comparing states/meta detects replaying the stale first request.
            SingleGpuTrainStepRequest request{rows,sequence,pass,pass?rows*3ULL:rows};
            Cuda(cudaStreamBeginCapture(graph_trainer->stream,cudaStreamCaptureModeGlobal),"begin capture");
            auto status=LaunchSingleGpuTrainStep(graph_trainer,request,&ticket);
            const auto ended=cudaStreamEndCapture(graph_trainer->stream,&pending);
            const auto captured=pending;
            Check(status==mgt::Status::kOk,"captured body"); Cuda(ended,"end capture");
            std::size_t count=0;
            Cuda(cudaGraphGetNodes(captured,nullptr,&count),"node count");
            std::vector<cudaGraphNode_t> nodes(count);
            Cuda(cudaGraphGetNodes(captured,nodes.data(),&count),"nodes");
            std::map<int,unsigned> kinds;
            for(auto node : nodes) {
                cudaGraphNodeType type;
                Cuda(cudaGraphNodeGetType(node,&type),"node type"); ++kinds[static_cast<int>(type)];
            }
            std::printf("rows=%u warm=%u pass=%u nodes=%zu types=",rows,warm,pass,count);
            for(auto [kind,n] : kinds) std::printf("%d:%u,",kind,n);
            std::puts("");
            if(own_workspace)
                Check(kinds[cudaGraphNodeTypeMemAlloc]==0&&kinds[cudaGraphNodeTypeMemFree]==0,
                    "provided workspace was lost: allocation/free nodes remain");
            if(!executable) {
                Cuda(cudaGraphInstantiate(&executable,captured,0),"instantiate");
                retained=captured;
                pending=nullptr;
            } else {
                cudaGraphExecUpdateResultInfo update{};
                const auto updated=cudaGraphExecUpdate(executable,captured,&update);
                std::printf("graph_update status=%d result=%d\n",(int)updated,(int)update.result);
                Cuda(updated,"whole graph update");
                Check(update.result==cudaGraphExecUpdateSuccess,"graph topology update rejected");
                Cuda(cudaGraphDestroy(captured),"updated source destroy");
                pending=nullptr;
            }
            Cuda(cudaGraphLaunch(executable,graph_trainer->stream),"graph launch");
            // Completion must be recorded outside capture for the public ticket.
            Cuda(cudaEventRecord(graph_trainer->completion,graph_trainer->stream),"graph event");
            Cuda(cudaEventSynchronize(graph_trainer->completion),"graph completion");
            Check(LaunchSingleGpuTrainStep(eager_trainer,request,&ticket)==mgt::Status::kOk,"eager reference");
            Cuda(cudaEventSynchronize(ticket.completion_event),"eager completion");
            Compare(graph_trainer,eager_trainer,rows<=256);
            SingleGpuTrainerMetrics metrics{};
            Check(ReadSingleGpuMetrics(graph_trainer,&metrics)==mgt::Status::kOk,"metrics");
            Check(metrics.optimizer_step==sequence&&std::isfinite(metrics.loss),"metrics sequence/finite");
            std::printf("PASS capture/update rows=%u warm=%u step=%llu epoch=%llu offset=%llu exact_model=%u loss=%.9g\n",
                rows,warm,(unsigned long long)sequence,(unsigned long long)request.epoch,
                (unsigned long long)request.epoch_sample_offset,rows<=256,metrics.loss);
        }
        cleanup();
    } catch(...) { cleanup(); throw; }
}

void StreamBinding() {
    cublasHandle_t blas=nullptr;
    cudaStream_t a=nullptr,b=nullptr,actual=nullptr;
    auto cleanup=[&] {
        if(blas)cublasDestroy(blas);
        if(a)cudaStreamDestroy(a);
        if(b)cudaStreamDestroy(b);
    };
    try {
        Check(mgt_cuda::detail::BindBlasStream(nullptr,nullptr)==CUBLAS_STATUS_NOT_INITIALIZED,
            "null handle accepted");
        Check(cublasCreate(&blas)==CUBLAS_STATUS_SUCCESS,"test blas create");
        Cuda(cudaStreamCreateWithFlags(&a,cudaStreamNonBlocking),"stream A");
        Cuda(cudaStreamCreateWithFlags(&b,cudaStreamNonBlocking),"stream B");
        for(auto stream:{a,a,b,b,static_cast<cudaStream_t>(nullptr)}) {
            Check(mgt_cuda::detail::BindBlasStream(blas,stream)==CUBLAS_STATUS_SUCCESS,"stream binding");
            Check(cublasGetStream(blas,&actual)==CUBLAS_STATUS_SUCCESS&&actual==stream,"bound stream mismatch");
        }
        cleanup();
    } catch(...) { cleanup(); throw; }
}
int main(int argc,char** argv) {
    try {
        Check(argc<=2,"usage: [rows]");
        const unsigned rows=argc==2?std::strtoul(argv[1],nullptr,10):4;
        Check(rows>0&&rows<=4096,"test rows range");
        StreamBinding();
        const auto puzzle=Puzzle();
        RunCapture(rows,false,true,puzzle);
        RunCapture(rows,true,true,puzzle);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL graph capture: %s\n",e.what()); return 1; }
}
