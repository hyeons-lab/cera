// XDNA Phase 0b measurement harness (see README.md and docs/XDNA_PHASE0_RESULTS.md).
//
// Drives the NPU through the XRT runtime that ships with the AMD NPU driver.
// The nop-job commands mirror xrt-smi's NPU latency test (validate xclbin, DPU
// flow, all-zero instruction buffer, opcode 1) but record every run, so they
// report a distribution. The IRON commands load an xclbin plus instruction
// binary built from iron/ and check the NPU's output against a CPU reference.

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static const char kUsage[] =
    "usage:\n"
    "  spike latency      <validate.xclbin> <iters>              nop job, start+wait per run, 50 warmup runs\n"
    "  spike latency-cold <validate.xclbin> <iters>              same, no warmup (as xrt-smi does it)\n"
    "  spike chain        <validate.xclbin> <iters> <L>...       xrt::runlist of L nop runs\n"
    "  spike stream       <validate.xclbin> <iters> <L>...       L nop runs started back to back, then waited on\n"
    "  spike load         <any.xclbin>                           register, create a hw_context, open the DPU kernel\n"
    "  spike gemv         <xclbin> <insts> <M> <K> [iters]       IRON int16 GEMV, checked\n"
    "  spike memcpy       <xclbin> <insts> <n_int32> [iters]     IRON memcpy, checked\n"
    "  spike readbw       <xclbin> <insts> <n_int32> <workers> [tile] [iters]  read-dominant design, checked\n"
    "  spike gemvchain    <xclbin> <insts> <M> <K> <iters> <L>...  dependent int16 GEMV chains\n"
    "  spike q4gemv       <xclbin> <insts> <M> <K> [iters]       Q4_0-style GEMV (iron/q4gemv), checked\n"
    "  spike q4chain      <artifact-dir> [tokens]               dependent Q4 GEMV chain at Llama-3.2-1B shapes\n"
    "  spike q4layer      <xclbin> <insts> [iters]              fused Q4 layer (iron/q4layer), checked\n"
    "  spike q4prefill    <xclbin> <insts> <N> <K> [iters]      MemTile-tiled Q4 prefill GEMM (B=32), checked\n"
    "  spike q4prefillchain <artifact-dir> [iters]              16-layer Llama-3.2-1B prefill runlist (B=32)\n"
    "  spike q4prefill_layer <xclbin> <insts> [pos0] [causal] [iters]  fused B=32 layer prefill, checked\n"
    "Shapes passed to the IRON commands must match the ones the xclbin was built for.\n";

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

static int parse_pos(const char* s, const char* what) {
  if (!s || !*s) {
    throw std::invalid_argument(std::string(what) + " must be a positive integer, got empty string");
  }
  errno = 0;
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (end == s || *end || errno == ERANGE || v <= 0 || v > INT_MAX - 64) {
    throw std::invalid_argument(std::string(what) + " must be a positive integer in 1.." +
                                std::to_string(INT_MAX - 64) + ", got '" + s + "'");
  }
  return int(v);
}

static size_t parse_size(const char* s, const char* what) {
  if (!s) {
    throw std::invalid_argument(std::string(what) + " must be a positive integer, got null");
  }
  const char* p = s;
  while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
  if (!*p || *p == '-') {
    throw std::invalid_argument(std::string(what) + " must be a positive integer, got '" + s + "'");
  }
  errno = 0;
  char* end = nullptr;
  unsigned long long v = std::strtoull(p, &end, 10);
  if (end == p || *end || errno == ERANGE || v == 0 || v > SIZE_MAX) {
    throw std::invalid_argument(std::string(what) + " must be a positive integer within size_t, got '" + s + "'");
  }
  return size_t(v);
}

static double pct(const std::vector<double>& sorted, double p) {
  return sorted[size_t(p * (sorted.size() - 1) + 0.5)];
}

// Prints min/p50/p90/p99/max/mean of v / divisor and returns the sorted values.
static std::vector<double> report(const char* label, std::vector<double> v, double divisor) {
  if (v.empty()) throw std::logic_error("no timing samples");
  for (auto& x : v) x /= divisor;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  std::printf("%-30s n=%zu  min=%.1f  p50=%.1f  p90=%.1f  p99=%.1f  max=%.1f  mean=%.1f us\n", label, v.size(),
              v.front(), pct(v, 0.50), pct(v, 0.90), pct(v, 0.99), v.back(), sum / v.size());
  return v;
}

// Times `iters` start+wait cycles after `warmup` untimed ones.
static std::vector<double> time_runs(xrt::run& run, int warmup, int iters) {
  std::vector<double> t;
  const int64_t total = int64_t(warmup) + int64_t(iters);
  for (int64_t i = 0; i < total; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait2();
    if (i >= warmup) t.push_back(us_since(t0));
  }
  return t;
}

// Times `iters` cycles of starting every run in `runs` and then waiting on all
// of them, after `warmup` untimed cycles. Returns each cycle's wall time.
static std::vector<double> time_batch(std::vector<xrt::run>& runs, int warmup, int iters) {
  std::vector<double> t;
  const int64_t total = int64_t(warmup) + int64_t(iters);
  for (int64_t i = 0; i < total; ++i) {
    auto t0 = clk::now();
    size_t started = 0;
    try {
      for (; started < runs.size(); ++started) runs[started].start();
    } catch (...) {
      for (size_t j = 0; j < started; ++j) {
        try {
          runs[j].wait();
        } catch (...) {
        }
      }
      throw;
    }
    size_t waited = 0;
    try {
      for (; waited < runs.size(); ++waited) runs[waited].wait2();
    } catch (...) {
      for (size_t j = waited; j < runs.size(); ++j) {
        try {
          runs[j].wait();
        } catch (...) {
        }
      }
      throw;
    }
    if (i >= warmup) t.push_back(us_since(t0));
  }
  return t;
}

// Same for one runlist: execute, wait, repeat.
static std::vector<double> time_runlist(xrt::runlist& rl, int warmup, int iters) {
  std::vector<double> t;
  const int64_t total = int64_t(warmup) + int64_t(iters);
  for (int64_t i = 0; i < total; ++i) {
    auto t0 = clk::now();
    rl.execute();
    rl.wait();
    if (i >= warmup) t.push_back(us_since(t0));
  }
  return t;
}

// ---------------------------------------------------------------------------
// Nop job on the driver's validate xclbin (DPU-sequence flow).

struct NopCtx {
  xrt::device dev;
  xrt::xclbin xclbin;
  xrt::hw_context hwctx;
  xrt::kernel kernel;
  xrt::xclbin::ip cu;
};

static std::string dpu_kernel_name(const xrt::xclbin& xclbin) {
  for (auto& k : xclbin.get_kernels()) {
    auto n = k.get_name();
    if (n.rfind("DPU", 0) == 0 || n.rfind("dpu", 0) == 0) return n;
  }
  return {};
}

static NopCtx open_nop(const char* path) {
  NopCtx c;
  c.dev = xrt::device(0);
  c.xclbin = xrt::xclbin(std::string(path));
  c.dev.register_xclbin(c.xclbin);
  c.hwctx = xrt::hw_context(c.dev, c.xclbin.get_uuid());
  auto name = dpu_kernel_name(c.xclbin);
  if (name.empty()) throw std::runtime_error("no DPU kernel in xclbin");
  c.kernel = xrt::kernel(c.hwctx, name);
  for (const auto& ip : c.xclbin.get_ips()) {
    if (ip.get_type() == xrt::xclbin::ip::ip_type::ps) {
      c.cu = ip;
      break;
    }
  }
  return c;
}

// One nop run with its own buffers, set up the way TestNPULatency does it.
static xrt::run make_nop_run(NopCtx& c, std::vector<xrt::bo>& keep) {
  xrt::run run(c.kernel);
  for (const auto& arg : c.cu.get_args()) {
    auto idx = static_cast<int>(arg.get_index());
    auto ty = arg.get_host_type();
    if (ty == "uint64_t") {
      run.set_arg(idx, static_cast<uint64_t>(1));  // DPU sequence opcode
    } else if (ty == "uint32_t") {
      run.set_arg(idx, static_cast<uint32_t>(1));
    } else if (ty.find('*') != std::string::npos) {
      xrt::bo bo;
      if (arg.get_name() == "instruct") {
        bo = xrt::bo(c.hwctx, arg.get_size(), xrt::bo::flags::cacheable, c.kernel.group_id(idx));
        std::memset(bo.map<void*>(), 0, arg.get_size());
      } else {
        bo = xrt::bo(c.dev, arg.get_size(), xrt::bo::flags::host_only, c.kernel.group_id(idx));
      }
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      keep.push_back(bo);
      run.set_arg(idx, bo);
    }
  }
  return run;
}

static int cmd_latency(const char* path, int iters, bool warm) {
  auto c = open_nop(path);
  std::vector<xrt::bo> keep;
  auto run = make_nop_run(c, keep);
  report(warm ? "single start+wait" : "single, no warmup", time_runs(run, warm ? 50 : 0, iters), 1.0);
  return 0;
}

// Runlist: L runs submitted as one list, executed in order on the device.
static int cmd_chain(const char* path, int iters, const std::vector<int>& lens) {
  auto c = open_nop(path);
  for (int len : lens) {
    std::vector<xrt::bo> keep;
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make_nop_run(c, keep));
    xrt::runlist rl(c.hwctx);
    for (auto& r : runs) rl.add(r);
    auto t = time_runlist(rl, 10, iters);
    char label[64];
    std::snprintf(label, sizeof label, "runlist L=%d (per list)", len);
    report(label, t, 1.0);
    std::snprintf(label, sizeof label, "runlist L=%d (per run)", len);
    report(label, t, len);
  }
  return 0;
}

// No runlist: start L independent runs back to back, then wait for all.
static int cmd_stream(const char* path, int iters, const std::vector<int>& lens) {
  auto c = open_nop(path);
  for (int len : lens) {
    std::vector<xrt::bo> keep;
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make_nop_run(c, keep));
    auto t = time_batch(runs, 10, iters);
    char label[64];
    std::snprintf(label, sizeof label, "stream L=%d (per run)", len);
    report(label, t, len);
  }
  return 0;
}

static int cmd_load(const char* path) {
  auto dev = xrt::device(0);
  auto t0 = clk::now();
  xrt::xclbin x{std::string(path)};
  std::printf("parsed: uuid=%s kernels:", x.get_uuid().to_string().c_str());
  for (auto& k : x.get_kernels()) std::printf(" %s", k.get_name().c_str());
  std::printf("\n");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  std::printf("parse + register + hw_context in %.1f us\n", us_since(t0));
  auto name = dpu_kernel_name(x);
  if (!name.empty()) {
    xrt::kernel k(ctx, name);
    std::printf("kernel %s opened\n", name.c_str());
  }
  return 0;
}

// ---------------------------------------------------------------------------
// IRON designs: kernel MLIR_AIE, arguments (opcode 3, instruction bo,
// instruction word count, then the design's buffers), the way the mlir-aie
// host tests submit them.

struct AieKernel {
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel kernel;
  xrt::bo instr;
  uint32_t n_instr = 0;
  std::string name;
};

static std::vector<uint32_t> load_insts(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) throw std::runtime_error(std::string("cannot open instruction file ") + path);
  std::fseek(f, 0, SEEK_END);
  long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (bytes <= 0 || bytes % 4) {
    std::fclose(f);
    throw std::runtime_error(std::string("instruction file is empty or not a whole number of 32-bit words: ") + path);
  }
  std::vector<uint32_t> words(size_t(bytes) / 4);
  size_t got = std::fread(words.data(), 4, words.size(), f);
  std::fclose(f);
  if (got != words.size()) throw std::runtime_error(std::string("short read on ") + path);
  return words;
}

static AieKernel open_mlir_aie(const char* xclbin_path, const char* insts_path) {
  auto words = load_insts(insts_path);
  AieKernel k;
  k.dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  for (auto& kk : x.get_kernels())
    if (kk.get_name().rfind("MLIR_AIE", 0) == 0) k.name = kk.get_name();
  if (k.name.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  k.dev.register_xclbin(x);
  k.ctx = xrt::hw_context(k.dev, x.get_uuid());
  k.kernel = xrt::kernel(k.ctx, k.name);
  k.n_instr = uint32_t(words.size());
  k.instr = xrt::bo(k.dev, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, k.kernel.group_id(1));
  std::memcpy(k.instr.map<void*>(), words.data(), words.size() * 4);
  k.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  return k;
}

static xrt::bo host_bo(AieKernel& k, size_t bytes, int arg) {
  return xrt::bo(k.dev, bytes, XRT_BO_FLAGS_HOST_ONLY, k.kernel.group_id(arg));
}

// A run with the three instruction arguments set; the caller sets 3 onward.
static xrt::run aie_run(AieKernel& k) {
  xrt::run r(k.kernel);
  r.set_arg(0, uint64_t(3));
  r.set_arg(1, k.instr);
  r.set_arg(2, k.n_instr);
  return r;
}

static ert_cmd_state wait_completed(const xrt::run& r, const char* where) {
  auto state = r.wait();
  if (state != ERT_CMD_STATE_COMPLETED) {
    throw std::runtime_error(std::string(where) + " ended in non-completed ERT state " + std::to_string(int(state)));
  }
  return state;
}


// Shared by gemv and gemvchain: deterministic int16 inputs in [-1000, 1000],
// and the reference dot product, accumulated wide and wrapped to int32 the way
// the NPU's int32 accumulator wraps.
static void fill_gemv_inputs(int16_t* a, int16_t* b, int M, int K) {
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return int16_t(int((s >> 16) % 2001) - 1000); };
  for (size_t i = 0; i < size_t(M) * K; ++i) a[i] = rnd();
  for (int i = 0; i < K; ++i) b[i] = rnd();
}

static int32_t gemv_ref_row(const int16_t* a, const int16_t* b, int row, int K) {
  int64_t acc = 0;
  for (int kk = 0; kk < K; ++kk) acc += int64_t(a[size_t(row) * K + kk]) * b[kk];
  return int32_t(uint32_t(uint64_t(acc)));
}

// int16 GEMV (C[M] = A[M,K] . B[K], int32 out).
static int cmd_gemv(const char* xclbin_path, const char* insts_path, int M, int K, int iters) {
  if (size_t(K) > SIZE_MAX / 2 || size_t(M) > SIZE_MAX / 4 || size_t(M) > SIZE_MAX / (size_t(K) * 2)) {
    throw std::invalid_argument("gemv M*K byte size overflows size_t");
  }
  auto k = open_mlir_aie(xclbin_path, insts_path);
  std::printf("kernel %s, %u instruction words\n", k.name.c_str(), k.n_instr);
  auto bo_a = host_bo(k, size_t(M) * K * 2, 3);
  auto bo_b = host_bo(k, size_t(K) * 2, 4);
  auto bo_c = host_bo(k, size_t(M) * 4, 5);
  auto* a = bo_a.map<int16_t*>();
  auto* b = bo_b.map<int16_t*>();
  auto* c = bo_c.map<int32_t*>();
  fill_gemv_inputs(a, b, M, K);
  std::memset(c, 0, size_t(M) * 4);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = aie_run(k);
  run.set_arg(3, bo_a);
  run.set_arg(4, bo_b);
  run.set_arg(5, bo_c);
  run.start();
  auto state = wait_completed(run, "gemv verify");
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::printf("first run state=%d\n", int(state));

  int bad = 0;
  for (int i = 0; i < M; ++i) {
    int32_t ref = gemv_ref_row(a, b, i, K);
    if (ref != c[i] && bad++ < 5) std::printf("  mismatch C[%d]: npu=%d ref=%d\n", i, c[i], ref);
  }
  std::printf("verify: %d of %d outputs wrong -> %s\n", bad, M, bad ? "FAIL" : "PASS");

  auto t = report("gemv start+wait", time_runs(run, 10, iters), 1.0);
  double p50 = pct(t, 0.5);
  std::printf("weights %.1f KiB, effective weight bandwidth at p50: %.2f GB/s\n", double(M) * K * 2 / 1024.0,
              M * double(K) * 2 / (p50 * 1e3));
  return bad ? 1 : 0;
}

// memcpy over n int32 words (args: in, out). Reports DDR read plus write.
static int cmd_memcpy(const char* xclbin_path, const char* insts_path, size_t n, int iters) {
  if (n > SIZE_MAX / 4) throw std::invalid_argument("memcpy n_int32 * 4 overflows size_t");
  auto k = open_mlir_aie(xclbin_path, insts_path);
  auto bo_in = host_bo(k, n * 4, 3);
  auto bo_out = host_bo(k, n * 4, 4);
  auto* in = bo_in.map<int32_t*>();
  auto* out = bo_out.map<int32_t*>();
  for (size_t i = 0; i < n; ++i) in[i] = int32_t(i * 2654435761u);
  std::memset(out, 0, n * 4);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = aie_run(k);
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);
  run.start();
  auto state = wait_completed(run, "memcpy verify");
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += out[i] != in[i];
  std::printf("kernel %s, state=%d, verify: %zu of %zu words wrong -> %s\n", k.name.c_str(), int(state), bad, n,
              bad ? "FAIL" : "PASS");

  auto t = report("memcpy start+wait", time_runs(run, 3, iters), 1.0);
  double p50 = pct(t, 0.5), best = t.front();
  std::printf("%.1f MiB each way; DDR read+write at p50: %.2f GB/s (best %.2f GB/s); read side alone: %.2f GB/s\n",
              n * 4 / 1048576.0, 2.0 * n * 4 / (p50 * 1e3), 2.0 * n * 4 / (best * 1e3), n * 4 / (p50 * 1e3));
  return bad ? 1 : 0;
}

// iron/readbw.py: n int32 words in, `workers` x 16 words out. Worker w sums the
// first word of each tile of its contiguous slice into its output word 0.
static int cmd_readbw(const char* xclbin_path, const char* insts_path, size_t n, int workers, size_t tile, int iters) {
  const size_t out_words = 16;
  if (n > SIZE_MAX / 4 || size_t(workers) > SIZE_MAX / tile || size_t(workers) > SIZE_MAX / (out_words * 4)) {
    throw std::invalid_argument("readbw byte or tile count overflows size_t");
  }
  if (n % (size_t(workers) * tile)) throw std::invalid_argument("n_int32 must be a multiple of workers x tile");
  auto k = open_mlir_aie(xclbin_path, insts_path);
  size_t n_out = size_t(workers) * out_words;
  auto bo_in = host_bo(k, n * 4, 3);
  auto bo_out = host_bo(k, n_out * 4, 4);
  auto* in = bo_in.map<int32_t*>();
  auto* out = bo_out.map<int32_t*>();
  for (size_t i = 0; i < n; ++i) in[i] = int32_t((i * 2654435761u) >> 8);
  std::memset(out, 0, n_out * 4);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = aie_run(k);
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);
  run.start();
  auto state = wait_completed(run, "readbw verify");
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  size_t per_worker = n / workers;
  int bad = 0;
  for (int wk = 0; wk < workers; ++wk) {
    uint32_t ref = 0;  // int32 wraparound, matching the core's add
    for (size_t i = 0; i < per_worker; i += tile) ref += uint32_t(in[wk * per_worker + i]);
    if (int32_t(ref) != out[wk * out_words] && bad++ < 4)
      std::printf("  worker %d: npu=%d ref=%d\n", wk, out[wk * out_words], int32_t(ref));
  }
  std::printf("kernel %s, state=%d, verify: %d of %d workers wrong -> %s\n", k.name.c_str(), int(state), bad, workers,
              bad ? "FAIL" : "PASS");

  auto t = report("readbw start+wait", time_runs(run, 3, iters), 1.0);
  std::printf("%.1f MiB read: %.2f GB/s at p50, %.2f GB/s at p90, best %.2f GB/s\n", n * 4 / 1048576.0,
              n * 4 / (pct(t, 0.5) * 1e3), n * 4 / (pct(t, 0.9) * 1e3), n * 4 / (t.front() * 1e3));
  return bad ? 1 : 0;
}

// Chains of L dependent int16 GEMVs: run i+1 reads run i's output buffer as its
// input vector B (the first K int16 of the int32 C buffer; needs 4M >= 2K).
// Order comes only from in-order execution on one hardware context, which is
// what a backend would rely on. Each length is timed as one runlist and as L
// back-to-back start() calls. Only the first link is checked.
static int cmd_gemvchain(const char* xclbin_path, const char* insts_path, int M, int K, int iters,
                         const std::vector<int>& lens) {
  if (size_t(K) > SIZE_MAX / 2 || size_t(M) > SIZE_MAX / 4 || size_t(M) > SIZE_MAX / (size_t(K) * 2)) {
    throw std::invalid_argument("gemvchain M*K byte size overflows size_t");
  }
  if (size_t(M) * 4 < size_t(K) * 2) throw std::invalid_argument("need 4*M >= 2*K to chain C into B");
  auto k = open_mlir_aie(xclbin_path, insts_path);
  auto bo_a = host_bo(k, size_t(M) * K * 2, 3);
  auto bo_b0 = host_bo(k, size_t(K) * 2, 4);
  auto* a = bo_a.map<int16_t*>();
  auto* b0 = bo_b0.map<int16_t*>();
  fill_gemv_inputs(a, b0, M, K);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int max_len = *std::max_element(lens.begin(), lens.end());
  std::vector<xrt::bo> bo_c;
  for (int i = 0; i < max_len; ++i) {
    bo_c.push_back(host_bo(k, size_t(M) * 4, 5));
    std::memset(bo_c.back().map<void*>(), 0, size_t(M) * 4);
    bo_c.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  auto make = [&](int i) {
    auto r = aie_run(k);
    r.set_arg(3, bo_a);
    if (i == 0) r.set_arg(4, bo_b0);
    else r.set_arg(4, bo_c[i - 1]);  // input vector = previous link's output
    r.set_arg(5, bo_c[i]);
    return r;
  };

  int bad = 0;
  {
    auto r = make(0);
    r.start();
    r.wait2();
    bo_c[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* c = bo_c[0].map<int32_t*>();
    for (int i = 0; i < M; ++i) bad += gemv_ref_row(a, b0, i, K) != c[i];
    std::printf("link 0 verify: %d of %d wrong -> %s\n", bad, M, bad ? "FAIL" : "PASS");
  }

  for (int len : lens) {
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make(i));
    xrt::runlist rl(k.ctx);
    for (auto& r : runs) rl.add(r);
    auto t_rl = time_runlist(rl, 3, iters);
    // Plain submission needs fresh run objects; a run added to a runlist belongs to it.
    std::vector<xrt::run> runs2;
    for (int i = 0; i < len; ++i) runs2.push_back(make(i));
    auto t_st = time_batch(runs2, 3, iters);
    char label[64];
    std::snprintf(label, sizeof label, "dep runlist L=%d (per run)", len);
    report(label, t_rl, len);
    std::snprintf(label, sizeof label, "dep stream L=%d (per run)", len);
    report(label, t_st, len);
  }
  return bad ? 1 : 0;
}

// Q4_0-style GEMV (iron/q4gemv): weights int4 (q - 8) with a bf16 scale per
// 32-weight block, x int8 with an f32 scale per 32-element block. The host
// quantizes random data the way llama.cpp's Q4_0 / Q8_0 do, packs one stream
// per worker in consumption order, and checks the NPU against a CPU reference
// that uses the same quantized values (so a mismatch is the kernel's, not
// quantization error).
static uint16_t f32_to_bf16(float f);  // defined below, with the Q4 section

// Shared Q4 shape rules, mirroring q4gemv.py: weight tiles stream in chunks
// of at most 2048 columns (the kernel accumulates split-K chunks), and the
// output-element grouping shrinks for small row counts.
static int q4_chunk_k(int K) { return K < 2048 ? K : 2048; }

static int q4_out_tiles(int M) {
  const int tiles_per_worker = (M / 16) / 16;  // 16 workers, 16 rows per tile
  const int cands[] = {16, 8, 4, 2, 1};
  for (int cand : cands)
    if (tiles_per_worker % cand == 0 && tiles_per_worker / cand >= 2) return cand;
  return 0;
}

// Random Q4_0-style weights for one shape: wq holds q - 8 in [-8, 7], wd the
// bf16 scale of each 32-weight block, generated like cmd_q4gemv does.
static void q4_gen_weights(int M, int K, uint32_t seed, std::vector<int8_t>& wq, std::vector<uint16_t>& wd,
                           float weight_scale = 1.0f) {
  const int QB = 32, nb = K / QB;
  wq.assign(size_t(M) * K, 0);
  wd.assign(size_t(M) * nb, 0);
  uint32_t s = seed;
  auto frand = [&]() {
    s = s * 1664525u + 1013904223u;
    return (float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * weight_scale;
  };
  std::vector<float> wrow(K);
  for (int r = 0; r < M; ++r) {
    for (int kk = 0; kk < K; ++kk) wrow[kk] = frand();
    for (int b = 0; b < nb; ++b) {
      float amax = 0, mx = 0;
      for (int j = 0; j < QB; ++j) {
        float v = wrow[b * QB + j];
        if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; }
      }
      float d = mx / -8.0f, id = d ? 1.0f / d : 0.0f;
      wd[size_t(r) * nb + b] = f32_to_bf16(d);
      for (int j = 0; j < QB; ++j) {
        int q = int(wrow[b * QB + j] * id + 8.5f);
        q = q < 0 ? 0 : (q > 15 ? 15 : q);
        wq[size_t(r) * K + b * QB + j] = int8_t(q - 8);
      }
    }
  }
}

// Packs wq/wd into the stream layout q4gemv.py consumes: worker w owns rows
// [w*rpw, (w+1)*rpw); each 16-row tile is sent as K-chunk slices holding, per
// 32-block, two 16x16 int4 B matrices then the 16 bf16 row scales.
static void q4_pack_weights(const std::vector<int8_t>& wq, const std::vector<uint16_t>& wd, int M, int K,
                            uint8_t* w, bool bt, bool hi_first) {
  const int QB = 32, ROWS = 16, nb = K / QB;
  const int chunk_k = q4_chunk_k(K), chunks = K / chunk_k, nb_chunk = chunk_k / QB;
  const size_t tb = size_t(nb_chunk) * (2 * 128 + ROWS * 2);
  for (int tile = 0; tile < M / ROWS; ++tile) {
    int r0 = tile * ROWS;
    for (int ch = 0; ch < chunks; ++ch) {
      uint8_t* tp = w + size_t(tile * chunks + ch) * tb;
      for (int lb = 0; lb < nb_chunk; ++lb) {
        int b = ch * nb_chunk + lb;
        uint8_t* bp = tp + size_t(lb) * 256;
        for (int h = 0; h < 2; ++h)
          for (int kk = 0; kk < 16; ++kk)
            for (int n = 0; n < 16; ++n) {
              int e = bt ? n * 16 + kk : kk * 16 + n;
              uint8_t nib = uint8_t(wq[size_t(r0 + n) * K + b * QB + h * 16 + kk]) & 0xF;
              bool high = ((e & 1) != 0) != hi_first;
              bp[h * 128 + e / 2] |= high ? uint8_t(nib << 4) : nib;
            }
        auto* sp = reinterpret_cast<uint16_t*>(tp + size_t(nb_chunk) * 256 + size_t(lb) * ROWS * 2);
        for (int n = 0; n < 16; ++n) sp[n] = wd[size_t(r0 + n) * nb + b];
      }
    }
  }
}

// Q8_0-style quantization of an f32 vector (cycled to length K): one f32
// scale per 32 elements, q = round-half-away(x / d) clamped to [-127, 127].
static void q4_quantize_f32(const float* in, int in_len, int K, int8_t* xq, float* xd) {
  if (in_len <= 0) throw std::invalid_argument("q4_quantize_f32: in_len must be positive");
  const int QB = 32;
  for (int b = 0; b < K / QB; ++b) {
    float amax = 0;
    for (int j = 0; j < QB; ++j) {
      float v = in[(b * QB + j) % in_len];
      if (!std::isfinite(v)) throw std::runtime_error("non-finite activation in q4_quantize_f32");
      amax = std::max(amax, std::fabs(v));
    }
    float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    xd[b] = d;
    for (int j = 0; j < QB; ++j) {
      float q = std::round(in[(b * QB + j) % in_len] * id);
      q = std::max(-127.0f, std::min(127.0f, q));
      xq[b * QB + j] = int8_t(int(q));
    }
  }
}

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return uint16_t((u >> 16) | 0x0040u);
  u += 0x7FFF + ((u >> 16) & 1);  // round to nearest even
  return uint16_t(u >> 16);
}
static float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static int parse_nonneg(const char* s, const char* what) {
  if (!s || !*s) {
    throw std::invalid_argument(std::string(what) + " must be a non-negative integer, got empty string");
  }
  errno = 0;
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (end == s || *end || errno == ERANGE || v < 0 || v > INT_MAX - 64) {
    throw std::invalid_argument(std::string(what) + " must be a non-negative integer in 0.." +
                                std::to_string(INT_MAX - 64) + ", got '" + s + "'");
  }
  return int(v);
}

static int cmd_q4gemv(const char* xclbin_path, const char* insts_path, int M, int K, int iters) {
  // q4gemv.py: 16 workers x 16 rows per tile; the output-element grouping
  // and the split-K chunking follow q4_out_tiles / q4_chunk_k above.
  const int QB = 32, ROWS = 16, WORKERS = 16;
  if (M % (WORKERS * ROWS) || q4_out_tiles(M) == 0)
    throw std::invalid_argument("M must be a multiple of 256 with a workable output tiling (see q4gemv.py)");
  if (K % 64) throw std::invalid_argument("K must be a multiple of 64");
  const int chunk_k = q4_chunk_k(K);
  if (K % chunk_k) throw std::invalid_argument("K must be a multiple of split-K chunk width");
  const int nb = K / QB, nb_chunk = chunk_k / QB, CHUNKS = K / chunk_k;
  const size_t tb = size_t(nb_chunk) * (2 * 128 + ROWS * 2);
  if (size_t(K) + size_t(nb) * 4 > tb) throw std::invalid_argument("K too wide for one x slot");
  if (size_t(M) > SIZE_MAX / size_t(K) || size_t(M / ROWS) > SIZE_MAX / (size_t(CHUNKS) * tb)) {
    throw std::invalid_argument("q4gemv M*K weight size overflows size_t");
  }

  // Random weights and activations, quantized like Q4_0 / Q8_0.
  uint32_t s = 2024;
  auto frand = [&]() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
  std::vector<int8_t> wq(size_t(M) * K);     // q - 8, in [-8, 7]
  std::vector<uint16_t> wd(size_t(M) * nb);  // bf16 scales
  std::vector<float> wrow(K);
  for (int r = 0; r < M; ++r) {
    for (int kk = 0; kk < K; ++kk) wrow[kk] = frand();
    for (int b = 0; b < nb; ++b) {
      float amax = 0, mx = 0;
      for (int j = 0; j < QB; ++j) {
        float v = wrow[b * QB + j];
        if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; }
      }
      float d = mx / -8.0f, id = d ? 1.0f / d : 0.0f;
      wd[size_t(r) * nb + b] = f32_to_bf16(d);
      for (int j = 0; j < QB; ++j) {
        int q = int(wrow[b * QB + j] * id + 8.5f);
        q = q < 0 ? 0 : (q > 15 ? 15 : q);
        wq[size_t(r) * K + b * QB + j] = int8_t(q - 8);
      }
    }
  }
  std::vector<int8_t> xq(K);
  std::vector<float> xd(nb);
  for (int b = 0; b < nb; ++b) {
    float xs[32], amax = 0;
    for (int j = 0; j < QB; ++j) { xs[j] = frand(); amax = std::max(amax, std::fabs(xs[j])); }
    float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    xd[b] = d;
    for (int j = 0; j < QB; ++j) {
      float q = std::round(xs[j] * id);
      q = std::max(-127.0f, std::min(127.0f, q));
      xq[b * QB + j] = int8_t(int(q));
    }
  }

  // Layout diagnostics (env): SPIKE_Q4_UNIT=1 sets every scale to 1 so the
  // output is the raw integer dot product; SPIKE_Q4_HI=1 puts even elements in
  // the high nibble; SPIKE_Q4_BT=1 packs B as [n][k] instead of [k][n].
  const bool unit = std::getenv("SPIKE_Q4_UNIT") != nullptr;
  const bool hi_first = std::getenv("SPIKE_Q4_HI") != nullptr;
  const bool bt = std::getenv("SPIKE_Q4_BT") != nullptr;
  if (unit) {
    for (auto& d : wd) d = f32_to_bf16(1.0f);
    for (auto& d : xd) d = 1.0f;
  }

  auto k = open_mlir_aie(xclbin_path, insts_path);
  const size_t w_bytes = size_t(M / ROWS) * CHUNKS * tb;
  auto bo_w = host_bo(k, w_bytes, 3);
  auto bo_x = host_bo(k, tb, 4);
  auto bo_y = host_bo(k, size_t(M) * 4, 5);

  // Pack: worker w owns rows [w*rpw, (w+1)*rpw); its tiles are contiguous. A
  // tile holds, for every block b, two 16x16 int4 B matrices (B[k][n] =
  // W[r0+n][b*32 + h*16 + k], element e = k*16+n, low nibble first), then the
  // 16 bf16 row scales of every block.
  auto* w = bo_w.map<uint8_t*>();
  std::memset(w, 0, w_bytes);
  q4_pack_weights(wq, wd, M, K, w, bt, hi_first);
  auto* xs = bo_x.map<uint8_t*>();
  std::memset(xs, 0, tb);
  std::memcpy(xs, xq.data(), K);
  std::memcpy(xs + K, xd.data(), size_t(nb) * 4);
  auto* y = bo_y.map<float*>();
  std::memset(y, 0, size_t(M) * 4);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto make = [&]() {
    auto r = aie_run(k);
    r.set_arg(3, bo_w);
    r.set_arg(4, bo_x);
    r.set_arg(5, bo_y);
    return r;
  };
  auto run = make();
  run.start();
  auto state = wait_completed(run, "q4gemv verify");
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference from the same quantized values; error relative to the row's
  // magnitude (sum of |terms|), since f32 accumulation order differs.
  int bad = 0;
  double worst = 0;
  bool any_nan = false;
  // Reference-range diagnostics (env): SPIKE_Q4_REF_FROM / SPIKE_Q4_REF_BLOCKS
  // limit the blocks summed; SPIKE_Q4_REF_MODE=1 uses x chunk 0 for every
  // chunk, =2 uses weight chunk 0 for every chunk (both assume 64-block
  // chunks). They exist to tell a stale or misindexed kernel from a bad one.
  const int nb_ref = std::getenv("SPIKE_Q4_REF_BLOCKS")
                         ? int(parse_pos(std::getenv("SPIKE_Q4_REF_BLOCKS"), "SPIKE_Q4_REF_BLOCKS"))
                         : nb;
  const int b_from = std::getenv("SPIKE_Q4_REF_FROM")
                         ? int(parse_nonneg(std::getenv("SPIKE_Q4_REF_FROM"), "SPIKE_Q4_REF_FROM"))
                         : 0;
  const int ref_mode = std::getenv("SPIKE_Q4_REF_MODE")
                           ? int(parse_nonneg(std::getenv("SPIKE_Q4_REF_MODE"), "SPIKE_Q4_REF_MODE"))
                           : 0;
  if (b_from > nb_ref || nb_ref > nb)
    throw std::invalid_argument("SPIKE_Q4_REF_FROM..SPIKE_Q4_REF_BLOCKS out of bounds for K");
  if (ref_mode > 2 || (ref_mode != 0 && nb < 64))
    throw std::invalid_argument("SPIKE_Q4_REF_MODE must be 0, 1, or 2 (modes 1 and 2 require K >= 2048)");
  for (int r = 0; r < M; ++r) {
    double ref = 0, mag = 0;
    for (int b = b_from; b < nb_ref; ++b) {
      int32_t dot = 0;
      const int bx = ref_mode == 1 ? b % 64 : b;
      const int bw = ref_mode == 2 ? b % 64 : b;
      for (int j = 0; j < QB; ++j) dot += int32_t(wq[size_t(r) * K + bw * QB + j]) * xq[bx * QB + j];
      double term = double(dot) * bf16_to_f32(wd[size_t(r) * nb + bw]) * xd[bx];
      ref += term;
      mag += std::fabs(term);
    }
    double err = std::fabs(double(y[r]) - ref) / (mag > 0 ? mag : 1);
    // Written with !(<=) so a NaN output counts as a failure.
    if (!(err <= 1e-4) && bad++ < 5) std::printf("  row %d: npu=%.6f ref=%.6f rel=%.2e\n", r, y[r], ref, err);
    if (std::isnan(err)) any_nan = true;
    else worst = std::max(worst, err);
  }
  std::printf("kernel %s, state=%d, verify: %d of %d rows off by >1e-4 of row magnitude (worst %.2e%s) -> %s\n",
              k.name.c_str(), int(state), bad, M, worst, any_nan ? ", and NaN rows" : "", bad ? "FAIL" : "PASS");

  // SPIKE_Q4_STREAM=L keeps L runs in flight (all started, then all waited),
  // so the NPU never idles between runs. Each sample is then one batch's wall
  // time divided by L: a per-run mean within the batch, and n counts batches.
  const char* stream_env = std::getenv("SPIKE_Q4_STREAM");
  int depth = stream_env ? int(parse_pos(stream_env, "SPIKE_Q4_STREAM")) : 1;
  std::vector<xrt::run> runs;
  for (int i = 0; i < depth; ++i) runs.push_back(make());
  auto t = time_batch(runs, 3, iters);
  for (auto& x : t) x /= runs.size();
  if (std::getenv("SPIKE_Q4_TRACE")) {  // samples in order, to see where a tail comes from
    for (size_t i = 0; i < t.size(); ++i) std::printf("%s%.0f", i % 20 ? " " : "\n  ", t[i]);
    std::printf("\n");
  }
  char label[64];
  if (depth > 1) std::snprintf(label, sizeof label, "q4gemv batch of %d (mean/run)", depth);
  else std::snprintf(label, sizeof label, "q4gemv start+wait");
  auto sorted = report(label, t, 1.0);
  std::printf("weights %.2f MiB (4.5 bits/weight incl. bf16 scales): %.2f GB/s at p50, %.2f at p90, best %.2f\n",
              w_bytes / 1048576.0, w_bytes / (pct(sorted, 0.5) * 1e3), w_bytes / (pct(sorted, 0.9) * 1e3),
              w_bytes / (sorted.front() * 1e3));
  return bad ? 1 : 0;
}

// ---------------------------------------------------------------------------

// Dependent Q4 GEMV chain at Llama-3.2-1B decode shapes: one full token of
// GEMVs (16 layers x q/k/v/o/gate/up/down, then the tied head at its padded
// row count 131072), each link's f32 output host-quantized into the next
// link's input, submitted start+wait the way a dependent decode step forces.
// This is the submission pattern the 3e spike could not measure (it kept
// independent runs in flight). Norms, RoPE and attention are host work in
// the planned backend too but are not timed here; per-shape correctness is
// covered by q4gemv. Artifact names follow iron/build.sh in <dir>.
static int cmd_q4chain(const char* dir, int tokens) {
  struct Shape { int M, K; };
  const Shape shapes[] = {{2048, 2048}, {512, 2048}, {8192, 2048}, {2048, 8192}, {131072, 2048}};
  struct Link {
    AieKernel kern;
    int M = 0, K = 0;
    size_t w_bytes = 0, tb = 0;
    xrt::bo bo_w, bo_x, bo_y;
    std::vector<float> y;
  };
  std::vector<Link> links;
  for (const auto& sh : shapes) {
    char name[64], xcl[1024], ins[1024];
    std::snprintf(name, sizeof name, "q4gemv_%dx%d", sh.M, sh.K);
    std::snprintf(xcl, sizeof xcl, "%s/%s.xclbin", dir, name);
    std::snprintf(ins, sizeof ins, "%s/insts_%s.bin", dir, name);
    Link L;
    L.M = sh.M; L.K = sh.K;
    std::printf("loading %s ...\n", name);
    L.kern = open_mlir_aie(xcl, ins);
    const int ROWS = 16, nb_chunk = q4_chunk_k(sh.K) / 32, chunks = sh.K / q4_chunk_k(sh.K);
    L.tb = size_t(nb_chunk) * (2 * 128 + ROWS * 2);
    L.w_bytes = size_t(sh.M / ROWS) * chunks * L.tb;
    std::vector<int8_t> wq;
    std::vector<uint16_t> wd;
    // Scale uniform [-1, 1] weights by sqrt(3 / K) so each GEMV preserves activation
    // variance across the 113-link chain instead of overflowing f32 to Inf/NaN.
    q4_gen_weights(sh.M, sh.K, 2024u + unsigned(sh.M), wq, wd, std::sqrt(3.0f / float(sh.K)));
    L.bo_w = host_bo(L.kern, L.w_bytes, 3);
    L.bo_x = host_bo(L.kern, L.tb, 4);
    L.bo_y = host_bo(L.kern, size_t(sh.M) * 4, 5);
    auto* w = L.bo_w.map<uint8_t*>();
    std::memset(w, 0, L.w_bytes);
    q4_pack_weights(wq, wd, sh.M, sh.K, w, false, false);
    L.bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    L.y.assign(sh.M, 0.0f);
    links.push_back(std::move(L));
  }
  auto by_shape = [&](int M, int K) -> Link* {
    for (auto& L : links)
      if (L.M == M && L.K == K) return &L;
    throw std::runtime_error("shape not loaded");
  };
  // One token: 16 layers of q, k, v, o, gate, up, down, then the head.
  std::vector<Link*> seq;
  for (int layer = 0; layer < 16; ++layer) {
    seq.push_back(by_shape(2048, 2048));
    seq.push_back(by_shape(512, 2048));
    seq.push_back(by_shape(512, 2048));
    seq.push_back(by_shape(2048, 2048));
    seq.push_back(by_shape(8192, 2048));
    seq.push_back(by_shape(8192, 2048));
    seq.push_back(by_shape(2048, 8192));
  }
  seq.push_back(by_shape(131072, 2048));
  if (std::getenv("SPIKE_Q4CHAIN_SAME")) seq.assign(113, by_shape(2048, 2048));  // diagnostic: no context switches
  double bytes_per_token = 0;
  for (auto* L : seq) bytes_per_token += double(L->w_bytes);

  std::vector<float> init_cur(2048);
  uint32_t s = 7;
  for (auto& v : init_cur) {
    s = s * 1664525u + 1013904223u;
    v = float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
  std::vector<float> cur;
  std::vector<int8_t> xq;
  std::vector<float> xd;
  auto run_token = [&]() {
    cur = init_cur;
    for (auto* L : seq) {
      xq.assign(L->K, 0);
      xd.assign(L->K / 32, 0.0f);
      q4_quantize_f32(cur.data(), int(cur.size()), L->K, xq.data(), xd.data());
      auto* xs = L->bo_x.map<uint8_t*>();
      std::memset(xs, 0, L->tb);
      std::memcpy(xs, xq.data(), L->K);
      std::memcpy(xs + L->K, xd.data(), size_t(L->K / 32) * 4);
      L->bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto run = aie_run(L->kern);
      run.set_arg(3, L->bo_w);
      run.set_arg(4, L->bo_x);
      run.set_arg(5, L->bo_y);
      run.start();
      wait_completed(run, "q4chain link");
      L->bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const float* y = L->bo_y.map<float*>();
      cur.assign(y, y + L->M);
    }
    for (float v : cur) {
      if (!std::isfinite(v)) throw std::runtime_error("non-finite output at end of q4chain token");
    }
  };
  for (int i = 0; i < 3; ++i) run_token();  // warmup tokens, untimed
  std::vector<double> per_token_ms;
  for (int i = 0; i < tokens; ++i) {
    auto t0 = clk::now();
    run_token();
    per_token_ms.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
  }
  std::vector<double> sorted = per_token_ms;
  std::sort(sorted.begin(), sorted.end());
  double p50 = sorted[sorted.size() / 2], mean = 0;
  for (double v : per_token_ms) mean += v;
  mean /= per_token_ms.size();
  std::printf("q4chain: %zu links/token, %.2f MiB streamed/token, n=%d tokens\n", seq.size(),
              bytes_per_token / 1048576.0, tokens);
  std::printf("  per token: p50 %.2f ms, mean %.2f ms -> %.1f tok/s at p50, %.2f us/link mean\n", p50, mean,
              1000.0 / p50, mean * 1000.0 / seq.size());
  std::printf("  effective streaming: %.2f GB/s at p50 (host quantize included, norms/attention not timed)\n",
              bytes_per_token / (p50 / 1000.0) / 1e9);
  return 0;
}

// Fused Q4 layer (iron/q4layer): all seven GEMV stages of one Llama-3.2-1B
// layer in a single run under one xclbin, activations quantized on the core
// between stages. Checked against a host simulation of the same stage chain
// (attention is proxied by its input, as in the design), then timed.
static int cmd_q4layer(const char* xclbin_path, const char* insts_path, int iters) {
  struct Stage { int in_slot, M, K, out_slot; };
  const int ACT_SLOT = 8192, N_SLOTS = 8, ACT_TAIL = 1024, TILE_BYTES = 18432;
  // SPIKE_Q4LAYER_SPEC="in:M:K:out;..." overrides the stage list (build the
  // xclbin with the matching Q4LAYER_SPEC) for isolating stage behavior.
  std::vector<Stage> stages = {{0, 2048, 2048, 1}, {0, 512, 2048, 2}, {0, 512, 2048, 3}, {1, 2048, 2048, 4},
                               {4, 8192, 2048, 5}, {4, 8192, 2048, 6}, {5, 2048, 8192, 7}};
  if (const char* spec = std::getenv("SPIKE_Q4LAYER_SPEC")) {
    stages.clear();
    std::string str = spec;
    size_t pos = 0;
    while (pos <= str.size()) {
      size_t semi = str.find(';', pos);
      std::string part = str.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos);
      if (!part.empty()) {
        int in_s = 0, M = 0, K = 0, out_s = 0;
        char extra = 0;
        if (std::sscanf(part.c_str(), "%d:%d:%d:%d%c", &in_s, &M, &K, &out_s, &extra) != 4)
          throw std::invalid_argument("bad SPIKE_Q4LAYER_SPEC stage: " + part);
        stages.push_back({in_s, M, K, out_s});
      }
      if (semi == std::string::npos) break;
      pos = semi + 1;
    }
  }
  if (stages.empty()) throw std::invalid_argument("SPIKE_Q4LAYER_SPEC must define at least one stage");
  for (const auto& st : stages) {
    if (st.in_slot < 0 || st.in_slot >= N_SLOTS || st.out_slot < 0 || st.out_slot >= N_SLOTS)
      throw std::invalid_argument("q4layer stage slot out of bounds 0..7");
    if (st.in_slot == st.out_slot)
      throw std::invalid_argument("q4layer stage in_slot and out_slot must differ");
    if (st.M < 512 || st.M > ACT_SLOT || st.M % 512 != 0)
      throw std::invalid_argument("q4layer stage M must be a multiple of 512 in 512..=8192");
    if (st.K < 2048 || st.K > ACT_SLOT || st.K % 2048 != 0)
      throw std::invalid_argument("q4layer stage K must be a multiple of 2048 in 2048..=8192");
  }
  auto kern = open_mlir_aie(xclbin_path, insts_path);

  size_t w_bytes = 0;
  for (const auto& st : stages) w_bytes += size_t(st.M / 16) * (st.K / 2048) * TILE_BYTES;
  const size_t act_floats = size_t(N_SLOTS * ACT_SLOT + ACT_TAIL);
  auto bo_w = host_bo(kern, w_bytes, 3);
  auto bo_y = host_bo(kern, act_floats * 4, 4);
  auto* w = bo_w.map<uint8_t*>();
  std::memset(w, 0, w_bytes);

  // Host simulation state, kept per stage for verification.
  struct Sim { std::vector<int8_t> wq; std::vector<uint16_t> wd; };
  std::vector<Sim> sims;
  size_t base = 0;
  for (size_t si = 0; si < stages.size(); ++si) {
    const auto& st = stages[si];
    Sim s;
    q4_gen_weights(st.M, st.K, 2024u + unsigned(si * 997 + size_t(st.M) * 31 + size_t(st.K)), s.wq, s.wd);
    q4_pack_weights(s.wq, s.wd, st.M, st.K, w + base, false, false);
    base += size_t(st.M / 16) * (st.K / 2048) * TILE_BYTES;
    sims.push_back(std::move(s));
  }
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto* y = bo_y.map<float*>();
  std::memset(y, 0, act_floats * 4);
  uint32_t s = 7;
  for (int i = 0; i < N_SLOTS * ACT_SLOT; ++i) {
    s = s * 1664525u + 1013904223u;
    y[i] = float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
  // Snapshot initial host slots before the NPU writes into bo_y so stages that
  // overwrite an earlier input slot cannot contaminate the host reference.
  std::vector<float> slots(y, y + size_t(N_SLOTS * ACT_SLOT));
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto the_run = aie_run(kern);
  the_run.set_arg(3, bo_w);
  the_run.set_arg(4, bo_y);
  auto run_once = [&]() {
    the_run.start();
    return wait_completed(the_run, "q4layer");
  };
  auto state = run_once();
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Host simulation: quantize each stage's input slot the way the core does
  // (f32 math, round half away from zero), then the Q4 GEMV in double.
  int bad_total = 0;
  size_t verified_rows = 0;
  for (size_t si = 0; si < stages.size(); ++si) {
    const auto& st = stages[si];
    const float* in = &slots[size_t(st.in_slot) * ACT_SLOT];
    std::vector<int8_t> xq(st.K);
    std::vector<float> xd(st.K / 32);
    for (int b = 0; b < st.K / 32; ++b) {
      float amax = 0;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(in[b * 32 + j]));
      float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
      xd[b] = d;
      for (int j = 0; j < 32; ++j) {
        float v = in[b * 32 + j] * id;
        int q = int(v >= 0.0f ? v + 0.5f : v - 0.5f);
        xq[b * 32 + j] = int8_t(q > 127 ? 127 : (q < -127 ? -127 : q));
      }
    }
    float* out = &slots[size_t(st.out_slot) * ACT_SLOT];
    int bad = 0;
    int checked = 0;
    double worst = 0;
    for (int r = 0; r < st.M; ++r) {
      double ref = 0, mag = 0;
      for (int b = 0; b < st.K / 32; ++b) {
        int32_t dot = 0;
        for (int j = 0; j < 32; ++j) dot += int32_t(sims[si].wq[size_t(r) * st.K + b * 32 + j]) * xq[b * 32 + j];
        double term = double(dot) * bf16_to_f32(sims[si].wd[size_t(r) * (st.K / 32) + b]) * xd[b];
        ref += term;
        mag += std::fabs(term);
      }
      out[r] = float(ref);
      bool overwritten_later = false;
      for (size_t sj = si + 1; sj < stages.size(); ++sj) {
        if (stages[sj].out_slot == st.out_slot && r < stages[sj].M) {
          overwritten_later = true;
          break;
        }
      }
      if (overwritten_later) continue;
      ++checked;
      ++verified_rows;
      double err = std::fabs(double(y[size_t(st.out_slot) * ACT_SLOT + r]) - ref) / (mag > 0 ? mag : 1);
      if (!(err <= 1e-4)) {
        if (bad < 3)
          std::printf("    row %d: npu=%.6f ref=%.6f\n", r, y[size_t(st.out_slot) * ACT_SLOT + r], ref);
        ++bad;
      } else {
        worst = std::max(worst, err);
      }
    }
    if (checked == 0) {
      std::printf("  stage %zu (%dx%d -> slot %d): overwritten by later stage (propagated in host sim)\n",
                  si, st.M, st.K, st.out_slot);
    } else {
      std::printf("  stage %zu (%dx%d -> slot %d): %d of %d rows off (worst ok %.2e) %s\n", si, st.M, st.K,
                  st.out_slot, bad, checked, worst, bad ? "FAIL" : "PASS");
    }
    bad_total += bad;
  }
  if (verified_rows == 0) throw std::logic_error("no q4layer output rows verified");
  std::printf("kernel %s, state=%d, fused layer verify: %s\n", kern.name.c_str(), int(state),
              bad_total ? "FAIL" : "PASS");

  auto t = [&]() {
    std::vector<double> v;
    const int64_t total = 3 + int64_t(iters);
    for (int64_t i = 0; i < total; ++i) {
      auto t0 = clk::now();
      run_once();
      if (i >= 3) v.push_back(us_since(t0));
    }
    return v;
  }();
  auto sorted = report("q4layer start+wait", t, 1.0);
  double p50_us = pct(sorted, 0.5);
  std::printf("weights %.2f MiB streamed per run: %.2f GB/s at p50\n", w_bytes / 1048576.0,
              w_bytes / (p50_us * 1e3));
  std::printf("projection: 16 layers + head (2816 us, one switch) = %.2f ms/token -> %.1f tok/s "
              "(host norms/RoPE/attention not included)\n",
              (16 * p50_us + 2816.0) / 1000.0, 1e6 / (16 * p50_us + 2816.0));
  return bad_total ? 1 : 0;
}

// MemTile-tiled Q4_0 prefill GEMM (iron/q4prefill): computes Y[32, N] = X[32, K] @ W[N, K]^T
// across all 32 AIE2P cores (4 token-rows x 8 weight-cols).
static int cmd_q4prefill(const char* xclbin_path, const char* insts_path, int N, int K, int iters) {
  const int N_ROWS = 4, N_COLS = 8, B_CORE = 8, B = N_ROWS * B_CORE, ROWS = 16, QB = 32, CHUNK_K = 2048;
  const int NB_CHUNK = CHUNK_K / QB, MAX_ACC_TILES = 8;
  if (N <= 0 || N % (N_COLS * ROWS) != 0)
    throw std::invalid_argument("q4prefill N must be a positive multiple of 128");
  if (K <= 0 || K % CHUNK_K != 0)
    throw std::invalid_argument("q4prefill K must be a positive multiple of 2048");

  const int n_col = N / N_COLS;
  const int tiles_per_col = n_col / ROWS;
  const int k_chunks = K / CHUNK_K;
  const int pass_tiles = (k_chunks > 1 && tiles_per_col > MAX_ACC_TILES) ? MAX_ACC_TILES : tiles_per_col;
  if (tiles_per_col % pass_tiles != 0)
    throw std::invalid_argument("q4prefill tiles_per_col not divisible by pass_tiles");
  const int n_passes = tiles_per_col / pass_tiles;

  const size_t TILE_BYTES = size_t(NB_CHUNK) * (2 * 128 + ROWS * 2);  // 18,432
  const size_t XQ_BYTES = size_t(B_CORE) * CHUNK_K;                   // 16,384
  const size_t XD_BYTES = size_t(B_CORE) * NB_CHUNK * 2;              // 1,024
  const size_t X_HDR_BYTES = 64;
  const size_t X_SLOT_BYTES = XQ_BYTES + XD_BYTES + X_HDR_BYTES;      // 17,472

  const size_t w_bytes = size_t(N / ROWS) * k_chunks * TILE_BYTES;
  const size_t x_bytes = size_t(k_chunks) * N_ROWS * X_SLOT_BYTES;
  const size_t y_bytes = size_t(B) * N * 4;
  const int nb = K / QB;

  std::vector<int8_t> wq;
  std::vector<uint16_t> wd;
  q4_gen_weights(N, K, 2024u + unsigned(N), wq, wd);

  std::vector<int8_t> xq(size_t(B) * K);
  std::vector<uint16_t> xd(size_t(B) * nb);
  uint32_t s = 1337u + unsigned(K);
  auto frand = [&]() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
  for (int tok = 0; tok < B; ++tok) {
    for (int b = 0; b < nb; ++b) {
      float xs[32], amax = 0;
      for (int j = 0; j < QB; ++j) {
        xs[j] = frand();
        amax = std::max(amax, std::fabs(xs[j]));
      }
      float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
      uint16_t d_bf16 = f32_to_bf16(d);
      xd[size_t(tok) * nb + b] = d_bf16;
      // Quantize with the exact f32 value of the bf16 scale or raw d (reference uses xd[tok*nb+b]).
      for (int j = 0; j < QB; ++j) {
        float q = std::round(xs[j] * id);
        q = std::max(-127.0f, std::min(127.0f, q));
        xq[size_t(tok) * K + b * QB + j] = int8_t(int(q));
      }
    }
  }

  auto k = open_mlir_aie(xclbin_path, insts_path);
  auto bo_w = host_bo(k, w_bytes, 3);
  auto bo_x = host_bo(k, x_bytes, 4);
  auto bo_y = host_bo(k, y_bytes, 5);

  auto* w = bo_w.map<uint8_t*>();
  std::memset(w, 0, w_bytes);
  q4_pack_weights(wq, wd, N, K, w, false, false);

  auto* x_buf = bo_x.map<uint8_t*>();
  std::memset(x_buf, 0, x_bytes);
  for (int ch = 0; ch < k_chunks; ++ch) {
    for (int r = 0; r < N_ROWS; ++r) {
      uint8_t* slot = x_buf + (size_t(ch) * N_ROWS + r) * X_SLOT_BYTES;
      auto* xq_dst = reinterpret_cast<int8_t*>(slot);
      auto* xd_dst = reinterpret_cast<uint16_t*>(slot + XQ_BYTES);
      auto* hdr = reinterpret_cast<int32_t*>(slot + XQ_BYTES + XD_BYTES);
      hdr[0] = n_passes;
      hdr[1] = pass_tiles;
      hdr[2] = k_chunks;
      for (int lb = 0; lb < NB_CHUNK; ++lb) {
        int b = ch * NB_CHUNK + lb;
        for (int grp = 0; grp < 2; ++grp) {
          for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < 4; ++t) {
              int tok = r * B_CORE + grp * 4 + t;
              for (int kk = 0; kk < 16; ++kk) {
                xq_dst[lb * 256 + grp * 128 + h * 64 + t * 16 + kk] =
                    xq[size_t(tok) * K + b * QB + h * 16 + kk];
              }
            }
          }
        }
        for (int t = 0; t < B_CORE; ++t) {
          xd_dst[lb * B_CORE + t] = xd[size_t(r * B_CORE + t) * nb + b];
        }
      }
    }
  }

  auto* y = bo_y.map<float*>();
  std::memset(y, 0, y_bytes);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto make = [&]() {
    auto r = aie_run(k);
    r.set_arg(3, bo_w);
    r.set_arg(4, bo_x);
    r.set_arg(5, bo_y);
    return r;
  };
  auto run = make();
  run.start();
  auto state = wait_completed(run, "q4prefill verify");
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int bad = 0;
  double worst = 0;
  bool any_nan = false;
  double sq_err_f64 = 0.0, sq_ref_f64 = 0.0;
  for (int tok = 0; tok < B; ++tok) {
    for (int row = 0; row < N; ++row) {
      double ref_bf16 = 0.0, mag_bf16 = 0.0, ref_f64 = 0.0;
      for (int ch = 0; ch < k_chunks; ++ch) {
        float chunk_acc = 0.0f;
        for (int lb = 0; lb < NB_CHUNK; ++lb) {
          int b = ch * NB_CHUNK + lb;
          int32_t dot = 0;
          for (int j = 0; j < QB; ++j) {
            dot += int32_t(wq[size_t(row) * K + b * QB + j]) * xq[size_t(tok) * K + b * QB + j];
          }
          float dw = bf16_to_f32(wd[size_t(row) * nb + b]);
          float dx = bf16_to_f32(xd[size_t(tok) * nb + b]);
          float r_b = bf16_to_f32(f32_to_bf16(float(dot)));
          float sc_b = bf16_to_f32(f32_to_bf16(dw * dx));
          chunk_acc += r_b * sc_b;
          mag_bf16 += std::fabs(double(r_b) * double(sc_b));
          ref_f64 += double(dot) * double(dw) * double(dx);
        }
        ref_bf16 += double(chunk_acc);
      }
      float got = y[size_t(tok) * N + row];
      double err = std::fabs(double(got) - ref_bf16) / (mag_bf16 > 0 ? mag_bf16 : 1.0);
      if (!(err <= 1e-4) && bad++ < 5)
        std::printf("  (tok=%d, row=%d): npu=%.6f ref_bf16=%.6f ref_f64=%.6f rel=%.2e\n", tok, row, got, ref_bf16,
                    ref_f64, err);
      if (std::isnan(err)) any_nan = true;
      else worst = std::max(worst, err);
      double d64 = double(got) - ref_f64;
      sq_err_f64 += d64 * d64;
      sq_ref_f64 += ref_f64 * ref_f64;
    }
  }
  double rms_rel_f64 = sq_ref_f64 > 0.0 ? std::sqrt(sq_err_f64 / sq_ref_f64) : 0.0;
  std::printf("kernel %s, state=%d, B=%d N=%d K=%d (n_passes=%d, pass_tiles=%d, k_chunks=%d)\n",
              k.name.c_str(), int(state), B, N, K, n_passes, pass_tiles, k_chunks);
  std::printf("verify vs bf16-epilogue sim: %d of %d outputs off by >1e-4 (worst %.2e%s) -> %s; rel RMS vs f64=%.4f%%\n",
              bad, B * N, worst, any_nan ? ", NaN" : "", bad ? "FAIL" : "PASS", rms_rel_f64 * 100.0);

  auto t_sw = time_runs(run, 3, iters);
  auto s_sw = report("q4prefill start+wait", t_sw, 1.0);

  const char* rl_env = std::getenv("SPIKE_Q4_RUNLIST");
  int rl_len = rl_env ? int(parse_pos(rl_env, "SPIKE_Q4_RUNLIST")) : 16;
  std::vector<xrt::run> rl_runs;
  xrt::runlist rl(k.ctx);
  for (int i = 0; i < rl_len; ++i) {
    rl_runs.push_back(make());
    rl.add(rl_runs.back());
  }
  auto t_rl = time_runlist(rl, 3, iters);
  char label[64];
  std::snprintf(label, sizeof label, "q4prefill runlist L=%d (/run)", rl_len);
  auto s_rl = report(label, t_rl, rl_len);

  double p50_rl = pct(s_rl, 0.5);
  double p50_sw = pct(s_sw, 0.5);
  double ops = 2.0 * B * N * K;
  std::printf("weights %.2f MiB: start+wait %.2f GB/s (%.2f TOPS), runlist L=%d %.2f GB/s (%.2f TOPS) at p50\n",
              w_bytes / 1048576.0, w_bytes / (p50_sw * 1e3), ops / (p50_sw * 1e6),
              rl_len, w_bytes / (p50_rl * 1e3), ops / (p50_rl * 1e6));
  return bad ? 1 : 0;
}

// 16-layer Llama-3.2-1B prefill GEMM runlist (112 runs = 16 layers x 7 stages, B=32 tokens)
// sharing ONE q4prefill xclbin/hw_context with per-shape instruction binaries.
static int cmd_q4prefillchain(const char* dir, int iters) {
  const int B = 32, B_CORE = 8, N_ROWS = 4, N_COLS = 8, ROWS = 16, QB = 32, CHUNK_K = 2048;
  const int NB_CHUNK = CHUNK_K / QB, MAX_ACC_TILES = 8;
  const size_t TILE_BYTES = size_t(NB_CHUNK) * (2 * 128 + ROWS * 2);
  const size_t X_SLOT_BYTES = size_t(B_CORE) * CHUNK_K + size_t(B_CORE) * NB_CHUNK * 2 + 64;

  char xcl[1024], inst2048[1024];
  std::snprintf(xcl, sizeof xcl, "%s/q4prefill_2048x2048.xclbin", dir);
  std::snprintf(inst2048, sizeof inst2048, "%s/insts_q4prefill_2048x2048.bin", dir);
  auto kern = open_mlir_aie(xcl, inst2048);

  struct ShapeInfo {
    int N, K;
    int n_passes, pass_tiles, k_chunks;
    size_t w_bytes, x_bytes, y_bytes;
    xrt::bo instr_bo;
    uint32_t n_instr;
  };
  const int specs[4][2] = {{2048, 2048}, {512, 2048}, {8192, 2048}, {2048, 8192}};
  std::vector<ShapeInfo> shapes(4);
  for (int i = 0; i < 4; ++i) {
    ShapeInfo& sh = shapes[i];
    sh.N = specs[i][0];
    sh.K = specs[i][1];
    int tiles_per_col = (sh.N / N_COLS) / ROWS;
    sh.k_chunks = sh.K / CHUNK_K;
    sh.pass_tiles = (sh.k_chunks > 1 && tiles_per_col > MAX_ACC_TILES) ? MAX_ACC_TILES : tiles_per_col;
    sh.n_passes = tiles_per_col / sh.pass_tiles;
    sh.w_bytes = size_t(sh.N / ROWS) * sh.k_chunks * TILE_BYTES;
    sh.x_bytes = size_t(sh.k_chunks) * N_ROWS * X_SLOT_BYTES;
    sh.y_bytes = size_t(B) * sh.N * 4;
    char ipath[1024];
    std::snprintf(ipath, sizeof ipath, "%s/insts_q4prefill_%dx%d.bin", dir, sh.N, sh.K);
    auto words = load_insts(ipath);
    sh.n_instr = uint32_t(words.size());
    sh.instr_bo = xrt::bo(kern.dev, words.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.kernel.group_id(1));
    std::memcpy(sh.instr_bo.map<void*>(), words.data(), words.size() * 4);
    sh.instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  const int stage_idx[7] = {0, 1, 1, 0, 2, 2, 3};

  // Allocate per-layer weights (522 MiB total, matching real model DDR footprint) and per-stage x/y BOs.
  std::vector<xrt::bo> bo_ws, bo_xs, bo_ys;
  std::vector<xrt::run> runs;
  xrt::runlist rl(kern.ctx);
  double total_w_bytes = 0.0, total_ops = 0.0;

  for (int s = 0; s < 4; ++s) {
    const auto& sh = shapes[s];
    auto bx = host_bo(kern, sh.x_bytes, 4);
    auto* x_buf = bx.map<uint8_t*>();
    std::memset(x_buf, 1, sh.x_bytes);
    for (int ch = 0; ch < sh.k_chunks; ++ch) {
      for (int r = 0; r < N_ROWS; ++r) {
        uint8_t* slot = x_buf + (size_t(ch) * N_ROWS + r) * X_SLOT_BYTES;
        auto* hdr = reinterpret_cast<int32_t*>(slot + size_t(B_CORE) * CHUNK_K + size_t(B_CORE) * NB_CHUNK * 2);
        hdr[0] = sh.n_passes;
        hdr[1] = sh.pass_tiles;
        hdr[2] = sh.k_chunks;
      }
    }
    bx.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_xs.push_back(std::move(bx));
    auto by = host_bo(kern, sh.y_bytes, 5);
    std::memset(by.map<void*>(), 0, sh.y_bytes);
    by.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ys.push_back(std::move(by));
  }

  for (int layer = 0; layer < 16; ++layer) {
    for (int st = 0; st < 7; ++st) {
      int si = stage_idx[st];
      const auto& sh = shapes[si];
      auto bw = host_bo(kern, sh.w_bytes, 3);
      std::memset(bw.map<void*>(), 0x11, sh.w_bytes);
      bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);

      xrt::run r(kern.kernel);
      r.set_arg(0, uint64_t(3));
      r.set_arg(1, sh.instr_bo);
      r.set_arg(2, sh.n_instr);
      r.set_arg(3, bw);
      r.set_arg(4, bo_xs[si]);
      r.set_arg(5, bo_ys[si]);
      rl.add(r);

      bo_ws.push_back(std::move(bw));
      runs.push_back(std::move(r));
      total_w_bytes += double(sh.w_bytes);
      total_ops += 2.0 * B * sh.N * sh.K;
    }
  }

  auto t = time_runlist(rl, 3, iters);
  auto sorted = report("q4prefillchain 16L (B=32)", t, 1.0);
  double p50_us = pct(sorted, 0.50);
  double p90_us = pct(sorted, 0.90);
  std::printf("Llama-3.2-1B 16-layer prefill (112 runs, B=%d, %.2f MiB weights, shared xclbin):\n",
              B, total_w_bytes / 1048576.0);
  std::printf("  p50=%.2f ms (%.1f tok/s, %.2f GB/s, %.2f TOPS), p90=%.2f ms (%.1f tok/s)\n",
              p50_us / 1000.0, B * 1e6 / p50_us, total_w_bytes / (p50_us * 1e3), total_ops / (p50_us * 1e6),
              p90_us / 1000.0, B * 1e6 / p90_us);
  return 0;
}

static int cmd_q4prefill_layer(const char* xclbin_path, const char* insts_path,
                               int pos0, int is_causal, int iters) {
  const int B = 32, H = 2048, Q_DIM = 2048, KV_DIM = 512, FFN = 8192;
  const int N_COLS = 8, ROWS = 16, QB = 32, CHUNK_K = 2048, NB_CHUNK = CHUNK_K / QB;
  const int HEAD_DIM = 64, MAX_KV_CHUNKS = 64, W_ELEMS = 4608, X_ELEMS = 5120, N_ROWS = 4;
  const int SLOT_HIDDEN_IN = 0;
  const int SLOT_XQ = SLOT_HIDDEN_IN + B * H;
  const int SLOT_Q = SLOT_XQ + 4 * N_ROWS * X_ELEMS;
  const int SLOT_KV_F32 = SLOT_Q + N_COLS * 2 * W_ELEMS;
  const int SLOT_KV_CUR = SLOT_KV_F32 + N_COLS * W_ELEMS;
  const int SLOT_ATTN_OUT = SLOT_KV_CUR + N_COLS * W_ELEMS;
  const int SLOT_O_OUT = SLOT_ATTN_OUT + B * H;
  const int SLOT_HIDDEN_MID = SLOT_O_OUT + B * H;
  const int SLOT_GATE = SLOT_HIDDEN_MID + B * H;
  const int SLOT_UP = SLOT_GATE + B * FFN;
  const int SLOT_DOWN_OUT = SLOT_UP + B * FFN;
  const int SLOT_META = SLOT_DOWN_OUT + B * H;
  const size_t ACT_TOTAL = size_t(SLOT_META) + 12288;
  const size_t KV_TOTAL = size_t(N_COLS) * MAX_KV_CHUNKS * W_ELEMS;
  const size_t TILE_BYTES = size_t(NB_CHUNK) * (2 * 128 + ROWS * 2);  // 18432

  if (pos0 < 0 || pos0 + B > MAX_KV_CHUNKS * 64) {
    throw std::invalid_argument("q4prefill_layer pos0 out of range 0.." + std::to_string(MAX_KV_CHUNKS * 64 - B));
  }
  const int kv_chunks = (pos0 + 63) / 64;

  struct Proj { const char* name; int N, K; size_t w_off; std::vector<int8_t> wq; std::vector<uint16_t> wd; };
  std::vector<Proj> projs = {
      {"Q", Q_DIM, H, 0, {}, {}},
      {"K", KV_DIM, H, 0, {}, {}},
      {"V", KV_DIM, H, 0, {}, {}},
      {"O", H, Q_DIM, 0, {}, {}},
      {"Gate", FFN, H, 0, {}, {}},
      {"Up", FFN, H, 0, {}, {}},
      {"Down", H, FFN, 0, {}, {}},
  };
  size_t w_bytes = 0;
  for (size_t i = 0; i < projs.size(); ++i) {
    projs[i].w_off = w_bytes;
    w_bytes += size_t(projs[i].N / ROWS) * (projs[i].K / CHUNK_K) * TILE_BYTES;
  }

  auto kern = open_mlir_aie(xclbin_path, insts_path);
  auto bo_w = host_bo(kern, w_bytes, 3);
  auto bo_y = host_bo(kern, ACT_TOTAL * 4, 4);
  auto bo_y_next = host_bo(kern, ACT_TOTAL * 4, 5);
  auto bo_kv = host_bo(kern, KV_TOTAL * 4, 6);

  auto* w_ptr = bo_w.map<uint8_t*>();
  std::memset(w_ptr, 0, w_bytes);
  const char* ws_env = std::getenv("SPIKE_W_SCALE");
  float w_scale = ws_env ? std::strtof(ws_env, nullptr) : 0.02f;
  for (size_t i = 0; i < projs.size(); ++i) {
    q4_gen_weights(projs[i].N, projs[i].K, 3000u + unsigned(i * 131), projs[i].wq, projs[i].wd, w_scale);
    q4_pack_weights(projs[i].wq, projs[i].wd, projs[i].N, projs[i].K, w_ptr + projs[i].w_off, false, false);
  }
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  uint32_t rng = 42u;
  auto frand = [&](float amp) {
    rng = rng * 1664525u + 1013904223u;
    return (float((rng >> 8) & 0xFFFF) / 32768.0f - 1.0f) * amp;
  };

  auto* y = bo_y.map<float*>();
  auto* y_next = bo_y_next.map<float*>();
  std::memset(y, 0, ACT_TOTAL * 4);
  std::memset(y_next, 0, ACT_TOTAL * 4);

  std::vector<float> x_in(size_t(B) * H);
  for (size_t i = 0; i < x_in.size(); ++i) {
    x_in[i] = frand(0.5f);
    y[SLOT_HIDDEN_IN + i] = x_in[i];
  }
  std::vector<float> norm_w0(H), norm_w1(H);
  for (int d = 0; d < H; ++d) {
    norm_w0[d] = 1.0f + frand(0.25f);
    norm_w1[d] = 1.0f + frand(0.25f);
    y[SLOT_META + d] = norm_w0[d];
    y[SLOT_META + 2048 + d] = norm_w1[d];
  }
  const float eps = 1e-5f;
  int32_t eps_bits = 0;
  std::memcpy(&eps_bits, &eps, 4);
  auto* params = reinterpret_cast<int32_t*>(y + SLOT_META + 4096);
  params[0] = pos0;
  params[1] = B;
  params[2] = kv_chunks;
  params[3] = pos0;       // n_past_tokens
  params[4] = is_causal;  // 1=causal, 0=bidirectional
  params[5] = 1;          // rope_mode = NeoX
  params[7] = eps_bits;

  std::vector<float> rope_cs(size_t(B) * 64);
  for (int t = 0; t < B; ++t) {
    double pos = double(pos0 + t);
    for (int i = 0; i < 32; ++i) {
      double inv_freq = std::pow(500000.0, -double(2 * i) / 64.0);
      double ang = pos * inv_freq;
      rope_cs[size_t(t) * 64 + i] = float(std::cos(ang));
      rope_cs[size_t(t) * 64 + 32 + i] = float(std::sin(ang));
      y[SLOT_META + 4608 + size_t(t) * 64 + i] = rope_cs[size_t(t) * 64 + i];
      y[SLOT_META + 4608 + size_t(t) * 64 + 32 + i] = rope_cs[size_t(t) * 64 + 32 + i];
    }
  }

  auto* kv_f32 = bo_kv.map<float*>();
  std::memset(kv_f32, 0, KV_TOTAL * 4);
  std::vector<float> past_k(size_t(N_COLS) * pos0 * HEAD_DIM, 0.0f);
  std::vector<float> past_v(size_t(N_COLS) * pos0 * HEAD_DIM, 0.0f);
  for (int c = 0; c < N_COLS; ++c) {
    for (int tp = 0; tp < pos0; ++tp) {
      int ch = tp / 64, kt = tp % 64;
      auto* k_dst = reinterpret_cast<uint16_t*>(kv_f32 + (size_t(c) * MAX_KV_CHUNKS + ch) * W_ELEMS);
      auto* v_dst = reinterpret_cast<uint16_t*>(kv_f32 + (size_t(c) * MAX_KV_CHUNKS + ch) * W_ELEMS + 2304);
      for (int d = 0; d < HEAD_DIM; ++d) {
        uint16_t kb = f32_to_bf16(frand(0.3f));
        uint16_t vb = f32_to_bf16(frand(0.3f));
        k_dst[kt * HEAD_DIM + d] = kb;
        v_dst[kt * HEAD_DIM + d] = vb;
        past_k[(size_t(c) * pos0 + tp) * HEAD_DIM + d] = bf16_to_f32(kb);
        past_v[(size_t(c) * pos0 + tp) * HEAD_DIM + d] = bf16_to_f32(vb);
      }
    }
  }

  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y_next.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_kv.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto make_run = [&]() {
    auto r = aie_run(kern);
    r.set_arg(3, bo_w);
    r.set_arg(4, bo_y);
    r.set_arg(5, bo_y_next);
    r.set_arg(6, bo_kv);
    return r;
  };
  auto run = make_run();
  run.start();
  auto state = wait_completed(run, "q4prefill_layer verify");
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_y_next.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // CPU f64 reference for the full transformer layer
  auto rmsnorm_ref = [&](const std::vector<double>& in, const std::vector<float>& nw, std::vector<double>& out) {
    out.resize(size_t(B) * H);
    for (int t = 0; t < B; ++t) {
      double ss = 0.0;
      for (int d = 0; d < H; ++d) ss += in[size_t(t) * H + d] * in[size_t(t) * H + d];
      double inv_r = 1.0 / std::sqrt(ss / double(H) + double(eps));
      for (int d = 0; d < H; ++d) out[size_t(t) * H + d] = in[size_t(t) * H + d] * inv_r * double(nw[d]);
    }
  };
  auto matmul_q4_ref = [&](const Proj& p, const std::vector<double>& in, std::vector<double>& out) {
    out.assign(size_t(B) * p.N, 0.0);
    int nb = p.K / QB;
    for (int t = 0; t < B; ++t) {
      for (int n = 0; n < p.N; ++n) {
        double acc = 0.0;
        for (int b = 0; b < nb; ++b) {
          double sub = 0.0;
          for (int j = 0; j < QB; ++j) {
            sub += double(p.wq[size_t(n) * p.K + b * QB + j]) * in[size_t(t) * p.K + b * QB + j];
          }
          acc += sub * double(bf16_to_f32(p.wd[size_t(n) * nb + b]));
        }
        out[size_t(t) * p.N + n] = acc;
      }
    }
  };

  std::vector<double> x_f64(x_in.size()), xn0, q_v, k_v, v_v;
  for (size_t i = 0; i < x_in.size(); ++i) x_f64[i] = double(x_in[i]);
  rmsnorm_ref(x_f64, norm_w0, xn0);
  matmul_q4_ref(projs[0], xn0, q_v);
  matmul_q4_ref(projs[1], xn0, k_v);
  matmul_q4_ref(projs[2], xn0, v_v);

  // NeoX RoPE on Q (32 heads) and K (8 heads)
  for (int t = 0; t < B; ++t) {
    const float* cs = &rope_cs[size_t(t) * 64];
    for (int h = 0; h < 32; ++h) {
      double* qp = &q_v[size_t(t) * Q_DIM + h * HEAD_DIM];
      for (int i = 0; i < 32; ++i) {
        double x0 = qp[i], x1 = qp[i + 32];
        qp[i] = x0 * cs[i] - x1 * cs[32 + i];
        qp[i + 32] = x0 * cs[32 + i] + x1 * cs[i];
      }
    }
    for (int h = 0; h < 8; ++h) {
      double* kp = &k_v[size_t(t) * KV_DIM + h * HEAD_DIM];
      for (int i = 0; i < 32; ++i) {
        double x0 = kp[i], x1 = kp[i + 32];
        kp[i] = x0 * cs[i] - x1 * cs[32 + i];
        kp[i + 32] = x0 * cs[32 + i] + x1 * cs[i];
      }
    }
  }

  std::vector<double> attn_out(size_t(B) * H, 0.0);
  const int total_k = pos0 + B;
  for (int hq = 0; hq < 32; ++hq) {
    int hk = hq / 4;
    for (int t = 0; t < B; ++t) {
      int lim = is_causal ? (pos0 + t + 1) : total_k;
      std::vector<double> sc(lim);
      double mx = -1e30;
      for (int kp_i = 0; kp_i < lim; ++kp_i) {
        double dot = 0.0;
        const double* qp = &q_v[size_t(t) * Q_DIM + hq * HEAD_DIM];
        if (kp_i < pos0) {
          const float* pf = &past_k[(size_t(hk) * pos0 + kp_i) * HEAD_DIM];
          for (int d = 0; d < HEAD_DIM; ++d) dot += qp[d] * double(pf[d]);
        } else {
          const double* pd = &k_v[size_t(kp_i - pos0) * KV_DIM + hk * HEAD_DIM];
          for (int d = 0; d < HEAD_DIM; ++d) dot += qp[d] * pd[d];
        }
        sc[kp_i] = dot * 0.125;
        if (sc[kp_i] > mx) mx = sc[kp_i];
      }
      double sum_p = 0.0;
      for (int kp_i = 0; kp_i < lim; ++kp_i) {
        sc[kp_i] = std::exp(sc[kp_i] - mx);
        sum_p += sc[kp_i];
      }
      double* op = &attn_out[size_t(t) * H + hq * HEAD_DIM];
      for (int kp_i = 0; kp_i < lim; ++kp_i) {
        double w = sc[kp_i] / sum_p;
        if (kp_i < pos0) {
          const float* vf = &past_v[(size_t(hk) * pos0 + kp_i) * HEAD_DIM];
          for (int d = 0; d < HEAD_DIM; ++d) op[d] += w * double(vf[d]);
        } else {
          const double* vd = &v_v[size_t(kp_i - pos0) * KV_DIM + hk * HEAD_DIM];
          for (int d = 0; d < HEAD_DIM; ++d) op[d] += w * vd[d];
        }
      }
    }
  }

  std::vector<double> o_out, mid(size_t(B) * H), xn1, gate_v, up_v, ffn_in(size_t(B) * FFN), down_out, ref(size_t(B) * H);
  matmul_q4_ref(projs[3], attn_out, o_out);
  for (size_t i = 0; i < mid.size(); ++i) mid[i] = x_f64[i] + o_out[i];
  rmsnorm_ref(mid, norm_w1, xn1);
  matmul_q4_ref(projs[4], xn1, gate_v);
  matmul_q4_ref(projs[5], xn1, up_v);
  for (size_t i = 0; i < ffn_in.size(); ++i) {
    double g = gate_v[i];
    ffn_in[i] = (g / (1.0 + std::exp(-g))) * up_v[i];
  }
  matmul_q4_ref(projs[6], ffn_in, down_out);
  for (size_t i = 0; i < ref.size(); ++i) ref[i] = mid[i] + down_out[i];

  if (std::getenv("SPIKE_DEBUG")) {
    auto check_stage = [&](const char* lbl, const float* got_p, const std::vector<double>& ref_v) {
      double e = 0.0, r = 0.0, mx = 0.0;
      for (size_t i = 0; i < ref_v.size(); ++i) {
        double d = double(got_p[i]) - ref_v[i];
        e += d * d; r += ref_v[i] * ref_v[i];
        mx = std::max(mx, std::fabs(d));
      }
      std::printf("  stage %-10s rel_RMS=%.4f%% max_abs=%.4f\n", lbl, 100.0 * std::sqrt(e / (r + 1e-30)), mx);
    };
    check_stage("attn_out", y + SLOT_ATTN_OUT, attn_out);
    check_stage("o_out", y + SLOT_O_OUT, o_out);
    check_stage("hidden_mid", y + SLOT_HIDDEN_MID, mid);
    check_stage("gate", y + SLOT_GATE, gate_v);
    check_stage("up", y + SLOT_UP, up_v);
    check_stage("down_out", y + SLOT_DOWN_OUT, down_out);
    double ke = 0, kr = 0, ve = 0, vr = 0;
    for (int c = 0; c < N_COLS; ++c) {
      const uint16_t* kp = reinterpret_cast<const uint16_t*>(y + SLOT_KV_CUR + size_t(c) * W_ELEMS);
      const uint16_t* vp = reinterpret_cast<const uint16_t*>(y + SLOT_KV_CUR + size_t(c) * W_ELEMS + 2304);
      for (int r = 0; r < N_ROWS; ++r) {
        for (int lt = 0; lt < 8; ++lt) {
          int t = r * 8 + lt;
          for (int d = 0; d < HEAD_DIM; ++d) {
            double gk = bf16_to_f32(kp[size_t(r) * 512 + lt * HEAD_DIM + d]);
            double rk = k_v[size_t(t) * KV_DIM + c * HEAD_DIM + d];
            ke += (gk - rk) * (gk - rk); kr += rk * rk;
            double gv = bf16_to_f32(vp[size_t(r) * 512 + lt * HEAD_DIM + d]);
            double rv = v_v[size_t(t) * KV_DIM + c * HEAD_DIM + d];
            ve += (gv - rv) * (gv - rv); vr += rv * rv;
          }
        }
      }
    }
    std::printf("  stage k_cur      rel_RMS=%.4f%%   v_cur rel_RMS=%.4f%%\n",
                100.0 * std::sqrt(ke / (kr + 1e-30)), 100.0 * std::sqrt(ve / (vr + 1e-30)));
    std::vector<double> npu_ffn(size_t(B) * FFN), down_from_npu;
    for (size_t i = 0; i < npu_ffn.size(); ++i) {
      double g = double(y[SLOT_GATE + i]);
      npu_ffn[i] = (g / (1.0 + std::exp(-g))) * double(y[SLOT_UP + i]);
    }
    matmul_q4_ref(projs[6], npu_ffn, down_from_npu);
    check_stage("down_only", y + SLOT_DOWN_OUT, down_from_npu);
  }

  double sq_err = 0.0, sq_ref = 0.0, max_diff = 0.0;
  int nans = 0, bad_sample = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    float got = y_next[SLOT_HIDDEN_IN + i];
    if (!std::isfinite(got)) { ++nans; continue; }
    double diff = double(got) - ref[i];
    sq_err += diff * diff;
    sq_ref += ref[i] * ref[i];
    max_diff = std::max(max_diff, std::fabs(diff));
    if (std::fabs(diff) > 0.15 && bad_sample++ < 5) {
      std::printf("  sample[%zu] got=%.5f ref=%.5f diff=%.5f\n", i, got, ref[i], diff);
    }
  }
  double rel_rms = sq_ref > 0.0 ? std::sqrt(sq_err / sq_ref) : 0.0;
  std::printf("kernel %s, state=%d, pos0=%d (kv_chunks=%d) is_causal=%d\n",
              kern.name.c_str(), int(state), pos0, kv_chunks, is_causal);
  std::printf("layer check vs f64: rel_RMS=%.4f%%, max_abs=%.4f, nans=%d -> %s\n",
              rel_rms * 100.0, max_diff, nans, (!nans && rel_rms < 0.01) ? "PASS" : "FAIL");

  if (iters > 0) {
    auto t_sw = time_runs(run, 2, iters);
    auto s_sw = report("q4prefill_layer start+wait", t_sw, 1.0);
    double p50_us = pct(s_sw, 0.50);
    std::printf("16L est at single-run p50=%.2f ms/layer -> %.2f ms (%.1f tok/s)\n",
                p50_us / 1000.0, 16.0 * p50_us / 1000.0, (B * 1e6) / (16.0 * p50_us));
  }
  if (const char* rl_env = std::getenv("SPIKE_LAYER_RUNLIST")) {
    int n_layers = int(parse_pos(rl_env, "SPIKE_LAYER_RUNLIST"));
    std::vector<xrt::bo> rl_ws, rl_ys, rl_kvs;
    std::vector<xrt::run> rl_runs;
    xrt::runlist rl(kern.ctx);
    for (int l = 0; l < n_layers; ++l) {
      auto bw = host_bo(kern, w_bytes, 3);
      std::memcpy(bw.map<void*>(), w_ptr, w_bytes);
      bw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto by = host_bo(kern, ACT_TOTAL * 4, 4);
      std::memcpy(by.map<void*>(), y, ACT_TOTAL * 4);
      by.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto bkv = host_bo(kern, KV_TOTAL * 4, 6);
      std::memcpy(bkv.map<void*>(), kv_f32, KV_TOTAL * 4);
      bkv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      auto r = aie_run(kern);
      r.set_arg(3, bw);
      r.set_arg(4, (l == 0) ? bo_y : rl_ys.back());
      r.set_arg(5, by);
      r.set_arg(6, bkv);
      rl.add(r);
      rl_ws.push_back(std::move(bw));
      rl_ys.push_back(std::move(by));
      rl_kvs.push_back(std::move(bkv));
      rl_runs.push_back(std::move(r));
    }
    auto t_rl = time_runlist(rl, 2, iters > 0 ? iters : 10);
    auto s_rl = report("q4prefill_layer runlist", t_rl, 1.0);
    double p50_us = pct(s_rl, 0.50);
    std::printf("%d-layer runlist: p50=%.2f ms (%.1f tok/s)\n",
                n_layers, p50_us / 1000.0, (B * 1e6) / p50_us);
  }
  return (!nans && rel_rms < 0.01) ? 0 : 1;
}

static std::vector<int> parse_lens(int argc, char** argv, int first) {
  std::vector<int> lens;
  for (int i = first; i < argc; ++i) lens.push_back(int(parse_pos(argv[i], "chain length")));
  if (lens.empty()) throw std::invalid_argument("at least one chain length is required");
  return lens;
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  auto need = [&](int n) { return argc >= n; };
  try {
    if ((cmd == "latency" || cmd == "latency-cold") && need(4))
      return cmd_latency(argv[2], int(parse_pos(argv[3], "iters")), cmd == "latency");
    if (cmd == "chain" && need(5)) return cmd_chain(argv[2], int(parse_pos(argv[3], "iters")), parse_lens(argc, argv, 4));
    if (cmd == "stream" && need(5)) return cmd_stream(argv[2], int(parse_pos(argv[3], "iters")), parse_lens(argc, argv, 4));
    if (cmd == "load" && need(3)) return cmd_load(argv[2]);
    if (cmd == "gemv" && need(6))
      return cmd_gemv(argv[2], argv[3], int(parse_pos(argv[4], "M")), int(parse_pos(argv[5], "K")),
                      argc > 6 ? int(parse_pos(argv[6], "iters")) : 1000);
    if (cmd == "memcpy" && need(5))
      return cmd_memcpy(argv[2], argv[3], parse_size(argv[4], "n_int32"), argc > 5 ? int(parse_pos(argv[5], "iters")) : 50);
    if (cmd == "readbw" && need(6))
      return cmd_readbw(argv[2], argv[3], parse_size(argv[4], "n_int32"), int(parse_pos(argv[5], "workers")),
                        argc > 6 ? parse_size(argv[6], "tile") : 1024, argc > 7 ? int(parse_pos(argv[7], "iters")) : 50);
    if (cmd == "gemvchain" && need(8))
      return cmd_gemvchain(argv[2], argv[3], int(parse_pos(argv[4], "M")), int(parse_pos(argv[5], "K")),
                           int(parse_pos(argv[6], "iters")), parse_lens(argc, argv, 7));
    if (cmd == "q4chain" && need(3))
      return cmd_q4chain(argv[2], argc > 3 ? int(parse_pos(argv[3], "tokens")) : 20);
    if (cmd == "q4layer" && need(4))
      return cmd_q4layer(argv[2], argv[3], argc > 4 ? int(parse_pos(argv[4], "iters")) : 30);
    if (cmd == "q4prefill" && need(6))
      return cmd_q4prefill(argv[2], argv[3], int(parse_pos(argv[4], "N")), int(parse_pos(argv[5], "K")),
                           argc > 6 ? int(parse_pos(argv[6], "iters")) : 30);
    if (cmd == "q4prefillchain" && need(3))
      return cmd_q4prefillchain(argv[2], argc > 3 ? int(parse_pos(argv[3], "iters")) : 20);
    if (cmd == "q4prefill_layer" && need(4))
      return cmd_q4prefill_layer(argv[2], argv[3],
                                 argc > 4 ? parse_nonneg(argv[4], "pos0") : 0,
                                 argc > 5 ? parse_nonneg(argv[5], "is_causal") : 1,
                                 argc > 6 ? parse_nonneg(argv[6], "iters") : 10);
    if (cmd == "q4gemv" && need(6))
      return cmd_q4gemv(argv[2], argv[3], int(parse_pos(argv[4], "M")), int(parse_pos(argv[5], "K")),
                        argc > 6 ? int(parse_pos(argv[6], "iters")) : 200);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fputs(kUsage, stderr);
  return 2;
}
