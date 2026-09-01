#include "single_gpu_graph_test_support.cuh"

using namespace mgt_graph_test;
using namespace mgt_cuda;

struct NativeOwner {
    SingleGpuTrainer* trainer=nullptr;
    explicit NativeOwner(const SingleGpuTrainerCreateInfo& info) {
        Check(CreateSingleGpuTrainer(info,&trainer)==mgt::Status::kOk,"native create");
    }
    ~NativeOwner() { if(trainer) DestroySingleGpuTrainer(&trainer); }
    NativeOwner(const NativeOwner&)=delete;
    NativeOwner& operator=(const NativeOwner&)=delete;
};

void CheckStructuralInputMap(SingleGpuTrainer* trainer) {
    const auto bins = static_cast<std::uint64_t>(trainer->shape.state_len) *
                      trainer->shape.state_value_pad;
    Check(trainer->input_active_bins.size() == trainer->shape.state_len * 24U,
          "production structural input-map cardinality");
    Check(trainer->input_active_bins.size() < bins &&
          trainer->layout.input_active_bins % 256U == 0,
          "compact input-map layout/gating");
    std::vector<std::uint16_t> device(trainer->input_active_bins.size());
    Cuda(cudaMemcpy(device.data(),
                    At<std::uint16_t>(trainer->arena,
                                      trainer->layout.input_active_bins),
                    device.size() * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost), "input active-map download");
    Check(device == trainer->input_active_bins,
          "device structural input-map differs from host owner");
}

void CheckSparseAdamGraphConfig(SingleGpuTrainer* trainer) {
    const auto& weight = *static_cast<const AdamWKernelConfig*>(
        trainer->graph.parameters[1].kernelParams[0]);
    const auto& affine = *static_cast<const AdamWKernelConfig*>(
        trainer->graph.parameters[2].kernelParams[0]);
    const std::uint64_t input_count =
        static_cast<std::uint64_t>(trainer->shape.state_len) *
        trainer->shape.state_value_pad * trainer->shape.hd1;
    const std::uint64_t active_count =
        trainer->input_active_bins.size() * trainer->shape.hd1;
    const std::uint64_t expected_live = active_count +
        trainer->layout.parameter_count - input_count;
    std::uint64_t physical_count = 0;
    Check(weight.param_count == expected_live &&
          weight.sparse_active_bins == At<std::uint16_t>(
              trainer->arena, trainer->layout.input_active_bins) &&
          weight.sparse_full_prefix_count == input_count &&
          weight.sparse_row_width == trainer->shape.hd1 &&
          weight.sparse_active_bin_count == trainer->input_active_bins.size() &&
          QueryAdamWPhysicalParameterCount(weight, &physical_count) ==
              mgt::Status::kOk &&
          physical_count == trainer->layout.parameter_count,
          "captured weight Adam did not retain sparse physical map");
    Check(!affine.sparse_active_bins &&
          affine.param_count == trainer->plan.trainable_count &&
          QueryAdamWPhysicalParameterCount(affine, &physical_count) ==
              mgt::Status::kOk &&
          physical_count == trainer->plan.trainable_count,
          "captured affine Adam must remain dense");
}

void CheckImpossibleInputGradientsStayZero(SingleGpuTrainer* trainer) {
    const std::uint64_t bins =
        static_cast<std::uint64_t>(trainer->shape.state_len) *
        trainer->shape.state_value_pad;
    std::vector<bool> active(bins, false);
    for (std::uint16_t bin : trainer->input_active_bins) active[bin] = true;
    std::vector<std::uint32_t> bits(bins * trainer->shape.hd1);
    Cuda(cudaMemcpy(bits.data(),
                    At<float>(trainer->arena, trainer->layout.weight_grad),
                    bits.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
         "input gradient download");
    for (std::uint64_t bin = 0; bin < bins; ++bin) {
        if (active[bin]) continue;
        for (unsigned h = 0; h < trainer->shape.hd1; ++h)
            Check(bits[bin * trainer->shape.hd1 + h] == 0,
                  "impossible input gradient lost persistent-zero invariant");
    }
}

void Run(unsigned rows, unsigned tail,
         SingleGpuInputGradientPrecision precision) {
    const auto puzzle=Puzzle();
    SingleGpuTrainerCreateInfo eager{};
    eager.contract=mgt::OriginalP888SingleGpuContract(); eager.capacity_rows=rows;
    eager.adam={0,1,1e-4f,.9f,.999f,1e-8f,0}; eager.puzzle=&puzzle;
    eager.base_seed=0x0888000000000001ULL;
    eager.input_gradient_precision=precision;
    auto graph=eager;
    graph.execution_mode=SingleGpuExecutionMode::kFixedBatchGraph;
    std::uint64_t eager_bytes=0,graph_bytes=0;
    Check(QuerySingleGpuTrainerBytes(eager,&eager_bytes)==mgt::Status::kOk,"eager query");
    Check(QuerySingleGpuTrainerBytes(graph,&graph_bytes)==mgt::Status::kOk,"graph query");
    Check(graph_bytes==eager_bytes+4194304,"workspace arena budget");
    auto bad=graph;
    bad.execution_mode=static_cast<SingleGpuExecutionMode>(99);
    Check(QuerySingleGpuTrainerBytes(bad,&graph_bytes)==mgt::Status::kInvalidConfig,"unknown mode accepted");
    bad=graph;
    bad.input_gradient_precision=static_cast<SingleGpuInputGradientPrecision>(99);
    Check(QuerySingleGpuTrainerBytes(bad,&graph_bytes)==mgt::Status::kInvalidConfig,
          "unknown input-gradient precision accepted");

    NativeOwner a(graph),b(eager);
    SingleGpuTrainStepTicket ticket{};
    Check(LaunchSingleGpuTrainStep(a.trainer,{rows,1,0,0},&ticket)==mgt::Status::kInvalidConfig,"unprepared graph accepted");
    Check(PrepareSingleGpuTrainer(a.trainer)==mgt::Status::kOk,"graph prepare");
    Check(PrepareSingleGpuTrainer(b.trainer)==mgt::Status::kOk,"eager prepare");
    CheckStructuralInputMap(a.trainer);
    CheckStructuralInputMap(b.trainer);
    CheckSparseAdamGraphConfig(a.trainer);
    Check(a.trainer->sequence==0&&!a.trainer->in_flight,"capture advanced training state");
    Check(a.trainer->graph.source&&a.trainer->graph.executable&&
        a.trainer->graph.rows==rows&&!b.trainer->graph.source&&!b.trainer->graph.executable,
        "explicit native graph was not instantiated");
    Check(a.trainer->layout.bytes==eager_bytes+4194304&&
        a.trainer->layout.blas_workspace%256==0,"owned workspace layout");
    Compare(a.trainer,b.trainer,true);
    Check(PrepareSingleGpuTrainer(a.trainer)==mgt::Status::kInvalidConfig,"second prepare accepted");
    const auto allocations=SingleGpuTrainerAllocationCountForTest();
    const std::array<SingleGpuTrainStepRequest,6> requests{{
        {rows,1,0,0},{tail,2,0,rows},{rows,3,0,rows+tail},
        {tail,4,0,mgt::P888TrainingContract::kSamplesPerEpoch-tail},
        {rows,5,1,0},{rows,6,1,rows}}};
    for(const auto& request:requests) {
        Check(LaunchSingleGpuTrainStep(a.trainer,request,&ticket)==mgt::Status::kOk,"native graph/tail enqueue");
        Check(ticket.sequence==request.optimizer_step&&ticket.completion_event,"graph ticket");
        Cuda(cudaEventSynchronize(ticket.completion_event),"graph completion");
        Check(LaunchSingleGpuTrainStep(b.trainer,request,&ticket)==mgt::Status::kOk,"eager enqueue");
        Cuda(cudaEventSynchronize(ticket.completion_event),"eager completion");
        Compare(a.trainer,b.trainer,rows<=256);
        if(request.optimizer_step==1) {
            CheckImpossibleInputGradientsStayZero(a.trainer);
            CheckImpossibleInputGradientsStayZero(b.trainer);
        }
        SingleGpuTrainerMetrics metrics{};
        Check(ReadSingleGpuMetrics(a.trainer,&metrics)==mgt::Status::kOk&&
            metrics.optimizer_step==request.optimizer_step&&std::isfinite(metrics.loss),"graph metrics");
        std::printf("PASS native graph rows=%u active=%u step=%llu epoch=%llu exact_model=%u precision=%s\n",
            rows,request.active_rows,(unsigned long long)request.optimizer_step,
            (unsigned long long)request.epoch,rows<=256,
            precision==SingleGpuInputGradientPrecision::kFp16Mirror?"fp16":"fp32");
    }
    const auto saved=ticket;
    for(const auto& request:{SingleGpuTrainStepRequest{rows,0,0,0},
            SingleGpuTrainStepRequest{rows,8,0,0},SingleGpuTrainStepRequest{0,7,0,0},
            SingleGpuTrainStepRequest{rows,7,0,UINT64_MAX},
            SingleGpuTrainStepRequest{rows,7,0,mgt::P888TrainingContract::kSamplesPerEpoch}}) {
        Check(LaunchSingleGpuTrainStep(a.trainer,request,&ticket)==mgt::Status::kInvalidConfig,"invalid request accepted");
        Check(a.trainer->sequence==6&&ticket.sequence==saved.sequence&&ticket.completion_event==saved.completion_event,
            "invalid request mutated bookkeeping");
    }
    for(unsigned step=7;step<=10;++step) {
        const SingleGpuTrainStepRequest request{step==8?tail:rows,step,2,(step-7ULL)*rows};
        Check(LaunchSingleGpuTrainStep(a.trainer,request,&ticket)==mgt::Status::kOk,"queued graph");
        Check(LaunchSingleGpuTrainStep(b.trainer,request,&ticket)==mgt::Status::kOk,"queued eager");
    }
    Cuda(cudaStreamSynchronize(a.trainer->stream),"queued graph completion");
    Cuda(cudaStreamSynchronize(b.trainer->stream),"queued eager completion");
    Compare(a.trainer,b.trainer,rows<=256);
    CheckImpossibleInputGradientsStayZero(a.trainer);
    CheckImpossibleInputGradientsStayZero(b.trainer);
    Check(SingleGpuTrainerAllocationCountForTest()==allocations,"arena allocation during steps");
    std::printf("PASS native queued full/tail rows=%u precision=%s\n",rows,
        precision==SingleGpuInputGradientPrecision::kFp16Mirror?"fp16":"fp32");
    // Force an internal graph invariant failure before any CUDA call. The
    // public request is valid; the trainer must fail-stop rather than use eager.
    const auto before_failure=ticket;
    a.trainer->graph.rows=0;
    Check(LaunchSingleGpuTrainStep(a.trainer,{rows,11,2,0},&ticket)==mgt::Status::kInvalidConfig&&
        a.trainer->failed&&a.trainer->sequence==10,"graph error did not fail-stop");
    Check(ticket.sequence==before_failure.sequence&&ticket.completion_event==before_failure.completion_event,
        "failed graph mutated ticket");
    SingleGpuTrainerMetrics metrics{};
    Check(PrepareSingleGpuTrainer(a.trainer)==mgt::Status::kCudaFailure&&
        LaunchSingleGpuTrainStep(a.trainer,{rows,11,2,0},&ticket)==mgt::Status::kCudaFailure&&
        ReadSingleGpuMetrics(a.trainer,&metrics)==mgt::Status::kCudaFailure,"failed trainer was reused");
    Compare(a.trainer,b.trainer,rows<=256);
    std::printf("PASS native fail-stop rows=%u precision=%s\n",rows,
        precision==SingleGpuInputGradientPrecision::kFp16Mirror?"fp16":"fp32");
}

int main(int argc,char** argv) {
    try {
        Check(argc<=3,"usage: [capacity_rows [tail_rows]]");
        const unsigned rows=argc>=2?std::strtoul(argv[1],nullptr,10):4;
        const unsigned tail=argc==3?std::strtoul(argv[2],nullptr,10):rows-1;
        Check(rows>=2&&rows<=4096&&tail>0&&tail<rows,"test capacity/tail range");
        Run(rows,tail,SingleGpuInputGradientPrecision::kFp32);
        Run(rows,tail,SingleGpuInputGradientPrecision::kFp16Mirror);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL native graph: %s\n",e.what()); return 1; }
}
