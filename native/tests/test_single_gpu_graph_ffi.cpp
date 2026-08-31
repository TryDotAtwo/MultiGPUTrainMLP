#include "mgt/single_gpu_trainer_ffi.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

static_assert(sizeof(MgtSingleGpuConfigV1)==80);
static_assert(sizeof(MgtSingleGpuExecutionOptionsV1)==16);

void Check(bool value,const char* message) {
    if(!value) throw std::runtime_error(message);
}
struct Owner {
    MgtSingleGpuHandle* handle=nullptr;
    ~Owner() { if(handle)mgt_single_gpu_v1_destroy(&handle); }
};

int main() {
    try {
        MgtSingleGpuConfigV1 config{};
        config.struct_size=sizeof(config); config.abi_version=MGT_SINGLE_GPU_ABI_V1;
        config.capacity_rows=4; config.learning_rate=.001f; config.beta1=.9f;
        config.beta2=.999f; config.epsilon=1e-8f;
        config.group_json_utf8=MGT_TEST_SOURCE_DIR "/production_inputs/p888.json";
        config.target_bin_utf8=MGT_TEST_SOURCE_DIR "/tests/fixtures/p888-target.bin";
        config.base_seed=0x8881; config.k_min=1; config.k_max=29;
        MgtSingleGpuExecutionOptionsV1 options{sizeof(options),MGT_SINGLE_GPU_ABI_V1,
            MGT_SINGLE_GPU_FIXED_BATCH_GRAPH,0};
        for(unsigned fault=0;fault<4;++fault) {
            Owner owner;
            auto bad=options;
            if(fault==0)--bad.struct_size;
            if(fault==1)++bad.abi_version;
            if(fault==2)bad.execution_mode=99;
            if(fault==3)bad.reserved_u32=1;
            const auto status=mgt_single_gpu_v1_create_with_options(&config,&bad,&owner.handle);
            Check(status==MGT_STATUS_INVALID_CONFIG&&!owner.handle,"invalid execution options accepted");
        }
        Owner owner;
        Check(mgt_single_gpu_v1_create_with_options(&config,nullptr,&owner.handle)==MGT_STATUS_INVALID_CONFIG&&
            !owner.handle,"null options accepted");
        Check(mgt_single_gpu_v1_create_with_options(&config,&options,nullptr)==MGT_STATUS_INVALID_CONFIG,
            "null output accepted");
        auto reserved=config;
        reserved.reserved_u32[0]=1;
        Check(mgt_single_gpu_v1_create_with_options(&reserved,&options,&owner.handle)==MGT_STATUS_INVALID_CONFIG&&
            !owner.handle,"legacy reserved field repurposed");
        Check(mgt_single_gpu_v1_create_with_options(&config,&options,&owner.handle)==MGT_STATUS_OK&&owner.handle,
            "graph create");
        MgtSingleGpuMetricsV1 metrics{};
        metrics.struct_size=sizeof(metrics);
        Check(mgt_single_gpu_v1_read_metrics(owner.handle,&metrics)==MGT_STATUS_INVALID_CONFIG,"unprepared read accepted");
        Check(mgt_single_gpu_v1_prepare(owner.handle)==MGT_STATUS_OK,"graph prepare");
        Check(mgt_single_gpu_v1_read_metrics(owner.handle,&metrics)==MGT_STATUS_OK&&
            metrics.optimizer_step==0&&metrics.completed_sequence==0,"capture advanced sequence");
        for(unsigned step=1;step<=4;++step) {
            MgtSingleGpuStepV1 request{sizeof(request),step==2?3U:4U,step,step/3U,(step%3U)*4ULL};
            Check(mgt_single_gpu_v1_train_step(owner.handle,&request,nullptr)==MGT_STATUS_OK,"queued graph/tail step");
        }
        Check(mgt_single_gpu_v1_read_metrics(owner.handle,&metrics)==MGT_STATUS_OK&&
            metrics.optimizer_step==4&&metrics.completed_sequence==4&&std::isfinite(metrics.loss),"queued metrics");
        Check(mgt_single_gpu_v1_read_metrics(owner.handle,&metrics)==MGT_STATUS_OK&&metrics.optimizer_step==4,
            "repeated read advanced sequence");
        auto bad_metrics=metrics;
        --bad_metrics.struct_size;
        Check(mgt_single_gpu_v1_read_metrics(owner.handle,&bad_metrics)==MGT_STATUS_INVALID_CONFIG&&
            mgt_single_gpu_v1_read_metrics(owner.handle,nullptr)==MGT_STATUS_INVALID_CONFIG,"invalid metrics accepted");
        // Destruction must drain an outstanding replay before freeing its arena.
        MgtSingleGpuStepV1 last{sizeof(last),4,5,2,0};
        Check(mgt_single_gpu_v1_train_step(owner.handle,&last,nullptr)==MGT_STATUS_OK,"final queued step");
        Check(mgt_single_gpu_v1_destroy(&owner.handle)==MGT_STATUS_OK&&!owner.handle,"graph destroy");
        auto eager_options=options;
        eager_options.execution_mode=MGT_SINGLE_GPU_EAGER;
        Check(mgt_single_gpu_v1_create_with_options(&config,&eager_options,&owner.handle)==MGT_STATUS_OK,
            "explicit eager options");
        Check(mgt_single_gpu_v1_prepare(owner.handle)==MGT_STATUS_OK,"explicit eager prepare");
        last.optimizer_step=1;
        Check(mgt_single_gpu_v1_train_step(owner.handle,&last,&metrics)==MGT_STATUS_OK&&
            metrics.optimizer_step==1&&std::isfinite(metrics.loss),"explicit eager step");
        std::puts("PASS C ABI graph options, async full/tail, metrics and destruction");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL graph C ABI: %s\n",e.what()); return 1; }
}
