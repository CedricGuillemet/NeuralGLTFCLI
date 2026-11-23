#include "../passes.h"
#include "codec.h"

namespace gtc {

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err) {
    if (opts.codec == "nvidia-ntc") {
#ifdef GTC_WITH_NVIDIA_NTC
        return createNvidiaNtcCodec(opts, err);
#else
        err = "the nvidia-ntc codec is not built in; reconfigure with -DGTC_WITH_NVIDIA_NTC=ON";
        return nullptr;
#endif
    }
    if (opts.codec == "babylon-ntc-v1" || opts.codec == "babylon-ntc-v2" || opts.codec == "babylon-ntc-v3") {
#ifdef GTC_WITH_CUDA_CODEC
        return createBabylonNtcCodec(opts, err);
#else
        err = opts.codec + " requires a build with GTC_WITH_CUDA_CODEC=ON (this build only has the nvidia-ntc codec)";
        return nullptr;
#endif
    }
    err = "unknown neural codec '" + opts.codec + "' (available: nvidia-ntc, babylon-ntc-v1, babylon-ntc-v2, babylon-ntc-v3)";
    return nullptr;
}

} // namespace gtc
