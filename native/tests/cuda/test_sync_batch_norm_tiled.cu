#include "mgt_cuda/allreduce_nccl.cuh"
#include "mgt_cuda/sync_batch_norm_tiled.cuh"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <cstdint>
#include <vector>

namespace {
template<class T> T* Managed(std::size_t n) { T* p=nullptr; return cudaMallocManaged(&p,n*sizeof(T))==cudaSuccess?p:nullptr; }
bool Near(float a,float b,float tol){return std::isfinite(a)&&std::fabs(a-b)<=tol*(1.0f+std::fabs(b));}
}
int main(){
    const char* local_env=std::getenv("SLURM_LOCALID");const char* world_env=std::getenv("SLURM_NTASKS");const char* rank_env=std::getenv("SLURM_PROCID");const char* id_env=std::getenv("MGT_TILED_BN_ID");
    const unsigned device=local_env?std::strtoul(local_env,nullptr,10):0,world=world_env?std::strtoul(world_env,nullptr,10):1,rank=rank_env?std::strtoul(rank_env,nullptr,10):0;
    if(cudaSetDevice(device)!=cudaSuccess||!world||rank>=world||(world>1&&!id_env))return 10;
    constexpr unsigned rows=300,cap=320,logical=37,physical=40;
    const std::size_t matrix=static_cast<std::size_t>(cap)*physical;
    float *x=Managed<float>(matrix),*gamma=Managed<float>(logical),*beta=Managed<float>(logical),
          *rm=Managed<float>(logical),*rv=Managed<float>(logical),*mean=Managed<float>(logical),
          *inv=Managed<float>(logical),*partials=Managed<float>(4*logical),
          *reduced=Managed<float>(2*logical),*up=Managed<float>(matrix),
          *dg=Managed<float>(logical),*db=Managed<float>(logical),*resgrad=Managed<float>(matrix);
    auto *xhat=Managed<__nv_bfloat16>(matrix),*activation=Managed<__nv_bfloat16>(matrix),
         *dz=Managed<__nv_bfloat16>(matrix); auto *mask=Managed<std::uint32_t>(cap*2);
    if(!x||!gamma||!beta||!rm||!rv||!mean||!inv||!partials||!reduced||!up||!dg||!db||!resgrad||!xhat||!activation||!dz||!mask)return 1;
    for(unsigned r=0;r<cap;++r)for(unsigned c=0;c<physical;++c){auto q=(std::size_t)r*physical+c;x[q]=r<rows&&c<logical?(float(int(r%17)-8)*0.1f+c*0.01f):99.0f;up[q]=0.25f+c*0.001f;}
    for(unsigned c=0;c<logical;++c){gamma[c]=0.8f+c*0.002f;beta[c]=-0.05f+c*0.001f;rm[c]=0;rv[c]=1;}
    mgt_cuda::NcclRankContext*ctx=nullptr;auto cs=world==1?mgt_cuda::CreateNcclSingleRankContext(device,&ctx):mgt_cuda::CreateNcclRankContext(device,world,rank,std::filesystem::path(id_env),&ctx);if(cs!=mgt::Status::kOk)return 2;
    mgt_cuda::TiledSyncBatchNormConfig cfg{256,32,mgt::A100XhatStorage::kBf16};
    mgt_cuda::TiledSyncBatchNormWorkspace ws{partials,4ULL*logical,reduced,2ULL*logical};
    if(mgt_cuda::TiledSyncBatchNormPartialFloats(cap,logical,cfg)!=4ULL*logical)return 3;
    if(mgt_cuda::LaunchTiledSyncBatchNormForward(cfg,x,nullptr,rows,cap,rows*world,logical,physical,gamma,beta,rm,rv,0.1f,1e-5f,mean,inv,xhat,activation,mask,ws,ctx,0)!=mgt::Status::kOk||cudaDeviceSynchronize()!=cudaSuccess)return 4;
    for(unsigned c=0;c<logical;++c){double s=0,q=0;for(unsigned r=0;r<rows;++r){float v=x[(std::size_t)r*physical+c];s+=v;q+=double(v)*v;}float m=float(s/rows),iv=1/std::sqrt(float(q/rows)-m*m+1e-5f);if(!Near(mean[c],m,2e-5f)||!Near(inv[c],iv,3e-5f))return 5;}
    if(mgt_cuda::LaunchTiledSyncBatchNormBackward(cfg,up,mask,xhat,rows,cap,rows*world,logical,physical,gamma,inv,dg,db,dz,resgrad,ws,ctx,0)!=mgt::Status::kOk||cudaDeviceSynchronize()!=cudaSuccess)return 6;
    for(unsigned c=0;c<logical;++c){double eg=0,eb=0;for(unsigned r=0;r<rows;++r){auto i=(std::size_t)r*physical+c;bool on=((mask[(std::size_t)r*2+c/32]>>(c&31))&1U)!=0;float dy=on?up[i]:0;eb+=dy;eg+=dy*__bfloat162float(xhat[i]);}if(!Near(dg[c],float(eg*world),3e-5f)||!Near(db[c],float(eb*world),3e-5f))return 7;}
    for(unsigned r=rows;r<cap;++r)for(unsigned c=0;c<physical;++c)if(__bfloat162float(dz[(std::size_t)r*physical+c])!=0.0f)return 8;
    auto result=mgt_cuda::DestroyNcclRankContext(ctx);cudaFree(x);cudaFree(gamma);cudaFree(beta);cudaFree(rm);cudaFree(rv);cudaFree(mean);cudaFree(inv);cudaFree(partials);cudaFree(reduced);cudaFree(up);cudaFree(dg);cudaFree(db);cudaFree(resgrad);cudaFree(xhat);cudaFree(activation);cudaFree(dz);cudaFree(mask);return result==mgt::Status::kOk?0:9;
}
