// C ABI over AMD's XRT runtime for cera's XDNA NPU backend.
//
// Built by cera/build.rs into a small DLL (cera_xdna_shim.dll) that links
// against xrt_coreutil.dll, the XRT runtime the AMD NPU driver installs in
// System32. The Rust side embeds the DLL and loads it at runtime, so a cera
// binary still starts on machines without the driver: loading the shim simply
// fails there and the backend reports itself unavailable.
//
// XRT's NPU path (xrt::hw_context, xrt::kernel, xrt::run) is exported only as
// MSVC C++ symbols, which is why this shim exists. Every C++ exception is
// caught at this boundary and turned into an error string; nothing throws
// across it. Handles are opaque and owned by the caller, who frees them in
// reverse order of creation (runs, then buffers, then kernels, then the
// device).
//
// Kernels follow the IRON (mlir-aie) convention: one kernel named MLIR_AIE per
// xclbin, invoked with (opcode 3, instruction buffer, instruction word count,
// data buffers...). The instruction words are uploaded once per kernel.

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define CERA_XDNA_API extern "C" __declspec(dllexport)

// Bumped whenever a signature below changes; the Rust side refuses a shim
// whose version it does not know.
#define CERA_XDNA_SHIM_ABI 2

// Shared hardware context and kernel handle for kernels loaded from the same
// xclbin UUID and kernel name. `ctx` is declared before `kernel` so C++ member
// destruction tears down `kernel` before `ctx`. `cera_xdna_device` holds only
// a `weak_ptr` so dropping the last `cera_xdna_kernel` (for example after
// `probe()`) releases the NPU column reservation immediately.
struct cached_hw_ctx {
  xrt::uuid uuid;
  std::string kernel_name;
  xrt::hw_context ctx;
  xrt::kernel kernel;
};

struct cera_xdna_device {
  xrt::device dev;
  std::weak_ptr<cached_hw_ctx> cached_hw;
};

struct cera_xdna_kernel {
  xrt::device dev;
  std::shared_ptr<cached_hw_ctx> hw;
  xrt::bo instr;
  uint32_t n_instr = 0;
};

struct cera_xdna_bo {
  xrt::bo bo;
};

struct cera_xdna_run {
  xrt::run run;
};

namespace {

void set_err(char* err, size_t len, const char* what) {
  if (!err || len == 0) return;
  std::snprintf(err, len, "%s", what ? what : "unknown error");
}

template <typename F>
int guarded(char* err, size_t len, F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    set_err(err, len, e.what());
  } catch (...) {
    set_err(err, len, "non-standard C++ exception from XRT");
  }
  return -1;
}

}  // namespace

CERA_XDNA_API uint32_t cera_xdna_shim_abi(void) { return CERA_XDNA_SHIM_ABI; }

// Process-wide power request toggled on 0->1 / 1->0 transitions by Rust's
// VoteCounter. Uses PowerCreateRequest rather than ES_CONTINUOUS so the vote
// is process-scoped (safe across threads when XdnaDevice moves).
CERA_XDNA_API void cera_xdna_set_wakelock(int enable) {
#ifdef _WIN32
  static std::mutex g_power_mu;
  static HANDLE g_power_req = INVALID_HANDLE_VALUE;
  static bool g_power_active = false;
  std::lock_guard<std::mutex> lock(g_power_mu);
  if (enable) {
    if (g_power_req == INVALID_HANDLE_VALUE) {
      REASON_CONTEXT rc{};
      rc.Version = POWER_REQUEST_CONTEXT_VERSION;
      rc.Flags = POWER_REQUEST_CONTEXT_SIMPLE_STRING;
      wchar_t reason[] = L"Cera XDNA NPU inference";
      rc.Reason.SimpleReasonString = reason;
      g_power_req = PowerCreateRequest(&rc);
    }
    if (g_power_req != INVALID_HANDLE_VALUE && !g_power_active) {
      PowerSetRequest(g_power_req, PowerRequestSystemRequired);
      PowerSetRequest(g_power_req, PowerRequestDisplayRequired);
      g_power_active = true;
    }
  } else if (g_power_req != INVALID_HANDLE_VALUE && g_power_active) {
    PowerClearRequest(g_power_req, PowerRequestDisplayRequired);
    PowerClearRequest(g_power_req, PowerRequestSystemRequired);
    g_power_active = false;
  }
#else
  (void)enable;
#endif
}

CERA_XDNA_API int cera_xdna_device_open(uint32_t index, cera_xdna_device** out, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!out) throw std::runtime_error("null out pointer passed to device_open");
    *out = new cera_xdna_device{xrt::device(index)};
    return 0;
  });
}

CERA_XDNA_API void cera_xdna_device_free(cera_xdna_device* d) {
  try {
    delete d;
  } catch (...) {
  }
}

// Writes the device's BDF-style identifier (for logs) into buf.
CERA_XDNA_API int cera_xdna_device_bdf(cera_xdna_device* d, char* buf, size_t len, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!d) throw std::runtime_error("null device handle passed to device_bdf");
    auto bdf = d->dev.get_info<xrt::info::device::bdf>();
    set_err(buf, len, bdf.c_str());
    return 0;
  });
}

// Registers an in-memory xclbin, creates a hardware context for it, opens its
// MLIR_AIE kernel and uploads the instruction words.
CERA_XDNA_API int cera_xdna_kernel_load(cera_xdna_device* d, const void* xclbin, size_t xclbin_len,
                                        const uint32_t* instr, size_t n_instr, cera_xdna_kernel** out, char* err,
                                        size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!d || !out || !xclbin || !instr) throw std::runtime_error("null pointer passed to kernel_load");
    if (n_instr == 0 || n_instr > UINT32_MAX / sizeof(uint32_t))
      throw std::runtime_error("instruction stream is empty or too long");
    if (xclbin_len < sizeof(axlf)) throw std::runtime_error("xclbin buffer is smaller than axlf header");
    axlf hdr{};
    std::memcpy(&hdr, xclbin, sizeof(hdr));
    if (std::memcmp(hdr.m_magic, "xclbin2\0", 8) != 0) throw std::runtime_error("invalid xclbin magic");
    const size_t base_hdr_bytes = sizeof(axlf) - sizeof(axlf_section_header);
    if (hdr.m_header.m_length < base_hdr_bytes || hdr.m_header.m_length > xclbin_len)
      throw std::runtime_error("xclbin header length exceeds input buffer");
    if (hdr.m_header.m_numSections > XCLBIN_MAX_NUM_SECTION)
      throw std::runtime_error("xclbin section count exceeds maximum");
    const uint64_t sec_table_end = static_cast<uint64_t>(base_hdr_bytes) +
                                   static_cast<uint64_t>(hdr.m_header.m_numSections) * sizeof(axlf_section_header);
    if (sec_table_end > hdr.m_header.m_length)
      throw std::runtime_error("xclbin section table exceeds container length");
    const auto* raw = static_cast<const char*>(xclbin);
    for (uint32_t i = 0; i < hdr.m_header.m_numSections; ++i) {
      axlf_section_header sec{};
      std::memcpy(&sec, raw + base_hdr_bytes + static_cast<size_t>(i) * sizeof(axlf_section_header), sizeof(sec));
      if (sec.m_sectionOffset > hdr.m_header.m_length ||
          sec.m_sectionSize > hdr.m_header.m_length - sec.m_sectionOffset) {
        throw std::runtime_error("xclbin section exceeds container length");
      }
      if (sec.m_sectionSize > 0 && sec.m_sectionOffset < sec_table_end) {
        throw std::runtime_error("xclbin section overlaps container header");
      }
    }
    std::vector<char> bytes(raw, raw + xclbin_len);
    xrt::xclbin x{bytes};
    std::string name;
    for (auto& k : x.get_kernels())
      if (k.get_name().rfind("MLIR_AIE", 0) == 0) name = k.get_name();
    if (name.empty()) throw std::runtime_error("xclbin has no MLIR_AIE kernel");
    auto k = new cera_xdna_kernel;
    try {
      k->dev = d->dev;
      const xrt::uuid x_uuid = x.get_uuid();
      auto hw = d->cached_hw.lock();
      if (!hw || !static_cast<bool>(x_uuid) || hw->uuid != x_uuid || hw->kernel_name != name) {
        k->dev.register_xclbin(x);
        // Vote for high-throughput, low-latency NPU DPM clocks via XRT QoS
        // (analogous to Hexagon's FASTRPC_CONTROL_LATENCY), falling back to a
        // lighter QoS hint or default shared context if resource limits reject it.
        xrt::hw_context ctx;
        bool created = false;
        for (uint32_t gops : {40000u, 8000u, 1000u}) {
          try {
            xrt::hw_context::cfg_param_type qos{
                {"gops", gops},
                {"fps", 100},
                {"dma_bandwidth", 55},
                {"latency", 10},
            };
            ctx = xrt::hw_context(k->dev, x_uuid, qos);
            created = true;
            break;
          } catch (...) {
          }
        }
        if (!created) {
          ctx = xrt::hw_context(k->dev, x_uuid);
        }
        xrt::kernel kern(ctx, name);
        hw = std::make_shared<cached_hw_ctx>(cached_hw_ctx{x_uuid, name, std::move(ctx), std::move(kern)});
      }
      k->hw = hw;
      k->n_instr = static_cast<uint32_t>(n_instr);
      const size_t instr_bytes = n_instr * sizeof(uint32_t);
      k->instr = xrt::bo(k->dev, instr_bytes, XCL_BO_FLAGS_CACHEABLE, k->hw->kernel.group_id(1));
      void* instr_map = k->instr.map<void*>();
      if (!instr_map) throw std::runtime_error("instruction buffer mapping failed");
      std::memcpy(instr_map, instr, instr_bytes);
      k->instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      if (static_cast<bool>(x_uuid)) {
        d->cached_hw = hw;
      }
    } catch (...) {
      delete k;
      throw;
    }
    *out = k;
    return 0;
  });
}

CERA_XDNA_API void cera_xdna_kernel_free(cera_xdna_kernel* k) {
  try {
    delete k;
  } catch (...) {
  }
}

// Host-only buffer for data argument `arg` (3 or above), in the memory group
// the kernel expects for that argument.
CERA_XDNA_API int cera_xdna_bo_alloc(cera_xdna_kernel* k, size_t bytes, uint32_t arg, cera_xdna_bo** out, char* err,
                                     size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!k || !k->hw || !out) throw std::runtime_error("null pointer passed to bo_alloc");
    if (bytes == 0) throw std::runtime_error("buffer size must be non-zero");
    if (arg < 3 || arg > static_cast<uint32_t>(INT_MAX))
      throw std::runtime_error("data argument index out of range");
    *out = new cera_xdna_bo{
        xrt::bo(k->dev, bytes, XRT_BO_FLAGS_HOST_ONLY, k->hw->kernel.group_id(static_cast<int>(arg)))};
    return 0;
  });
}

CERA_XDNA_API void* cera_xdna_bo_map(cera_xdna_bo* b) {
  if (!b) return nullptr;
  try {
    return b->bo.map<void*>();
  } catch (...) {
    return nullptr;
  }
}

CERA_XDNA_API int cera_xdna_bo_sync(cera_xdna_bo* b, int to_device, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!b) throw std::runtime_error("null buffer handle passed to bo_sync");
    b->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE);
    return 0;
  });
}

CERA_XDNA_API int cera_xdna_bo_sync_range(cera_xdna_bo* b, int to_device, size_t size, size_t offset,
                                          char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!b) throw std::runtime_error("null buffer handle passed to bo_sync_range");
    const size_t bo_size = b->bo.size();
    if (offset > bo_size || size > bo_size - offset)
      throw std::runtime_error("buffer sync range out of bounds");
    if (size == 0) return 0;
    b->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
    return 0;
  });
}

CERA_XDNA_API void cera_xdna_bo_free(cera_xdna_bo* b) {
  try {
    delete b;
  } catch (...) {
  }
}

// A run with the instruction arguments set and bos[i] bound to argument 3 + i.
CERA_XDNA_API int cera_xdna_run_create(cera_xdna_kernel* k, cera_xdna_bo* const* bos, size_t n_bos,
                                       cera_xdna_run** out, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!k || !k->hw || !out || (n_bos > 0 && !bos))
      throw std::runtime_error("null pointer passed to run_create");
    if (n_bos > static_cast<size_t>(INT_MAX - 3)) throw std::runtime_error("too many buffer arguments");
    auto r = new cera_xdna_run{xrt::run(k->hw->kernel)};
    try {
      r->run.set_arg(0, uint64_t(3));
      r->run.set_arg(1, k->instr);
      r->run.set_arg(2, k->n_instr);
      for (size_t i = 0; i < n_bos; ++i) {
        if (!bos[i]) throw std::runtime_error("null buffer handle in run_create");
        r->run.set_arg(static_cast<int>(3 + i), bos[i]->bo);
      }
    } catch (...) {
      delete r;
      throw;
    }
    *out = r;
    return 0;
  });
}

CERA_XDNA_API int cera_xdna_run_start(cera_xdna_run* r, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!r) throw std::runtime_error("null run handle passed to run_start");
#ifdef _WIN32
    SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
#endif
    r->run.start();
    return 0;
  });
}

// Waits for the run; timeout_ms of 0 waits without a limit. Returns the
// ert_cmd_state value (4 is ERT_CMD_STATE_COMPLETED), or -1 with err set.
CERA_XDNA_API int cera_xdna_run_wait(cera_xdna_run* r, uint32_t timeout_ms, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!r) throw std::runtime_error("null run handle passed to run_wait");
    auto state = timeout_ms ? r->run.wait(std::chrono::milliseconds(timeout_ms)) : r->run.wait();
    return static_cast<int>(state);
  });
}

CERA_XDNA_API void cera_xdna_run_free(cera_xdna_run* r) {
  try {
    delete r;
  } catch (...) {
  }
}

struct cera_xdna_runlist {
  xrt::runlist list;
};

// An empty runlist on the kernel's hardware context. Runs added to it execute
// in submission order on the device; the fused layer design submits a whole
// token's runs this way instead of waiting on each one.
CERA_XDNA_API int cera_xdna_runlist_create(cera_xdna_kernel* k, cera_xdna_runlist** out, char* err,
                                           size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!k || !k->hw || !out) throw std::runtime_error("null pointer passed to runlist_create");
    *out = new cera_xdna_runlist{xrt::runlist(k->hw->ctx)};
    return 0;
  });
}

CERA_XDNA_API int cera_xdna_runlist_add(cera_xdna_runlist* l, cera_xdna_run* r, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!l || !r) throw std::runtime_error("null handle passed to runlist_add");
    l->list.add(r->run);
    return 0;
  });
}

CERA_XDNA_API int cera_xdna_runlist_execute(cera_xdna_runlist* l, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!l) throw std::runtime_error("null runlist handle passed to runlist_execute");
#ifdef _WIN32
    // Prevent Windows Modern Standby / display-off SoC fabric downclocking
    // during active NPU decode turns (see docs/XDNA_PHASE0_RESULTS.md sec. 0).
    SetThreadExecutionState(ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);
#endif
    l->list.execute();
    return 0;
  });
}

// Waits for the whole list; timeout_ms of 0 waits without a limit. Returns 0
// when all runs complete, or -1 with err set on command failure or timeout.
CERA_XDNA_API int cera_xdna_runlist_wait(cera_xdna_runlist* l, uint32_t timeout_ms, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (!l) throw std::runtime_error("null runlist handle passed to runlist_wait");
    if (timeout_ms) {
      if (l->list.wait(std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout)
        throw std::runtime_error("runlist wait timed out");
    } else {
      l->list.wait();
    }
    return 0;
  });
}

CERA_XDNA_API void cera_xdna_runlist_free(cera_xdna_runlist* l) {
  try {
    delete l;
  } catch (...) {
  }
}
