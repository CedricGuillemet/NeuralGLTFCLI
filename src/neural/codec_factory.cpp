#include "../passes.h"
#include "codec.h"

namespace gtc {

std::unique_ptr<INeuralCodec> createNeuralCodec(const NeuralOptions& opts, std::string& err) {
    if (opts.codec == "babylon-ntc-v1" || opts.codec == "babylon-ntc-v2" || opts.codec == "babylon-ntc-v3") {
#ifdef GTC_WITH_CUDA_CODEC
        return createBabylonNtcCodec(opts, err);
#else
        err = opts.codec + " requires a build with GTC_WITH_CUDA_CODEC=ON";
        return nullptr;
#endif
    }
    if (opts.codec == "babylon-ntc-v4") {
#ifdef GTC_WITH_CUDA_CODEC
        return createBabylonNtcV4Codec(opts, err);
#else
        err = "babylon-ntc-v4 (the default neural codec) requires a build with GTC_WITH_CUDA_CODEC=ON";
        return nullptr;
#endif
    }
    err = "unknown neural codec '" + opts.codec + "' (available: babylon-ntc-v4, babylon-ntc-v1, babylon-ntc-v2, babylon-ntc-v3)";
    return nullptr;
}

} // namespace gtc
