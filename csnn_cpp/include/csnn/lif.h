#ifndef HLS4NM_INCLUDE_LIF_ACTIVATION_H_
#define HLS4NM_INCLUDE_LIF_ACTIVATION_H_

#include <array>
#include "hls4nm/utils.h"
#include "hls4nm/params.h"
#include "hls_stream.h"

namespace hls4nm {

template <unsigned N>
class LIF_Activation {
 public:
  LIF_Activation(void) = default;
  LIF_Activation(leak_t leak, thres_t thres);

  void run(hls::stream<std::array<spike_t, N>>& dinStream,
           hls::stream<std::array<spike_t, N>>& doutStream);

  void reset(void);

 private:
  state_t _states[N];
  const leak_t _leak;
  const thres_t _thres;
};

template <unsigned N>
LIF_Activation<N>::LIF_Activation(leak_t leak, thres_t thres)
    : _leak(leak), _thres(thres) {
  reset();
}

template <unsigned N>
void LIF_Activation<N>::reset(void) {
  for (auto n = 0; n < N; n++) {
    _states[n] = 0;
  }
}

template <unsigned N>
void LIF_Activation<N>::run(hls::stream<std::array<spike_t, N>>& dinStream,
                            hls::stream<std::array<spike_t, N>>& doutStream) {
  constexpr unsigned STATE_MIN = -128;
  auto in_spk = dinStream.read();
  std::array<spike_t, N> out_spk;

  for (auto n = 0; n < N; n++) {
    int next_state = _states[n] * _leak;
    next_state = quantize(next_state);
    next_state += in_spk[n]; // Assuming in_spk contains the weighted input
    if (next_state >= _thres) {
      out_spk[n] = 1;
      next_state -= _thres;
    } else {
      out_spk[n] = 0;
    }
    _states[n] = next_state < STATE_MIN ? STATE_MIN : next_state;
  }

  doutStream << out_spk;
}

}  // namespace hls4nm

#endif  //  HLS4NM_INCLUDE_LIF_ACTIVATION_H_
