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
#include <chrono>
#include <cmath>
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
    "Shapes passed to the IRON commands must match the ones the xclbin was built for.\n";

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

static long parse_pos(const char* s, const char* what) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!*s || *end || v <= 0) throw std::invalid_argument(std::string(what) + " must be a positive integer, got '" + s + "'");
  return v;
}

static size_t parse_size(const char* s, const char* what) {
  if (*s == '-') throw std::invalid_argument(std::string(what) + " must be a positive integer, got '" + s + "'");
  char* end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (!*s || *end || v == 0) throw std::invalid_argument(std::string(what) + " must be a positive integer, got '" + s + "'");
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
  for (int i = 0; i < warmup + iters; ++i) {
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
  for (int i = 0; i < warmup + iters; ++i) {
    auto t0 = clk::now();
    for (auto& r : runs) r.start();
    for (auto& r : runs) r.wait2();
    if (i >= warmup) t.push_back(us_since(t0));
  }
  return t;
}

// Same for one runlist: execute, wait, repeat.
static std::vector<double> time_runlist(xrt::runlist& rl, int warmup, int iters) {
  std::vector<double> t;
  for (int i = 0; i < warmup + iters; ++i) {
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
  auto state = run.wait();
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
  auto state = run.wait();
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
  auto state = run.wait();
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
static void q4_gen_weights(int M, int K, uint32_t seed, std::vector<int8_t>& wq, std::vector<uint16_t>& wd) {
  const int QB = 32, nb = K / QB;
  wq.assign(size_t(M) * K, 0);
  wd.assign(size_t(M) * nb, 0);
  uint32_t s = seed;
  auto frand = [&]() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
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
  const int QB = 32;
  for (int b = 0; b < K / QB; ++b) {
    float amax = 0;
    for (int j = 0; j < QB; ++j) amax = std::max(amax, std::fabs(in[(b * QB + j) % in_len]));
    float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
    xd[b] = d;
    for (int j = 0; j < QB; ++j) xq[b * QB + j] = int8_t(std::lround(in[(b * QB + j) % in_len] * id));
  }
}

static uint16_t f32_to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);  // round to nearest even
  return uint16_t(u >> 16);
}
static float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static int cmd_q4gemv(const char* xclbin_path, const char* insts_path, int M, int K, int iters) {
  // q4gemv.py: 16 workers x 16 rows per tile; the output-element grouping
  // and the split-K chunking follow q4_out_tiles / q4_chunk_k above.
  const int QB = 32, ROWS = 16, WORKERS = 16;
  if (M % (WORKERS * ROWS) || q4_out_tiles(M) == 0)
    throw std::invalid_argument("M must be a multiple of 256 with a workable output tiling (see q4gemv.py)");
  if (K % 64) throw std::invalid_argument("K must be a multiple of 64");
  const int nb = K / QB, nb_chunk = q4_chunk_k(K) / QB, CHUNKS = K / q4_chunk_k(K);
  const size_t tb = size_t(nb_chunk) * (2 * 128 + ROWS * 2);
  if (size_t(K) + size_t(nb) * 4 > tb) throw std::invalid_argument("K too wide for one x slot");

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
    for (int j = 0; j < QB; ++j) xq[b * QB + j] = int8_t(std::lround(xs[j] * id));
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
  auto state = run.wait();
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
  const int nb_ref = std::getenv("SPIKE_Q4_REF_BLOCKS") ? int(parse_pos(std::getenv("SPIKE_Q4_REF_BLOCKS"), "blocks")) : nb;
  const int b_from = std::getenv("SPIKE_Q4_REF_FROM") ? int(std::strtol(std::getenv("SPIKE_Q4_REF_FROM"), nullptr, 10)) : 0;
  const int ref_mode = std::getenv("SPIKE_Q4_REF_MODE") ? int(parse_pos(std::getenv("SPIKE_Q4_REF_MODE"), "mode")) : 0;
  for (int r = 0; r < M; ++r) {
    double ref = 0, mag = 0;
    for (int b = b_from; b < nb_ref; ++b) {
      int32_t dot = 0;
      const int bx = ref_mode == 1 ? b % 64 : b;
      const int bw = ref_mode == 2 ? b % 64 : b;
      for (int j = 0; j < QB; ++j) dot += int32_t(wq[size_t(r) * K + bw * QB + j]) * xq[bx * QB + j];
      double term = double(dot) * bf16_to_f32(wd[size_t(r) * nb + b]) * xd[bx];
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
    q4_gen_weights(sh.M, sh.K, 2024u + unsigned(sh.M), wq, wd);
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

  std::vector<float> cur(2048);
  uint32_t s = 7;
  for (auto& v : cur) { s = s * 1664525u + 1013904223u; v = float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
  std::vector<int8_t> xq;
  std::vector<float> xd;
  auto run_token = [&]() {
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
      run.wait();
      L->bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      const float* y = L->bo_y.map<float*>();
      cur.assign(y, y + L->M);
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
        int in_s, M, K, out_s;
        if (std::sscanf(part.c_str(), "%d:%d:%d:%d", &in_s, &M, &K, &out_s) != 4)
          throw std::invalid_argument("bad SPIKE_Q4LAYER_SPEC stage: " + part);
        stages.push_back({in_s, M, K, out_s});
      }
      if (semi == std::string::npos) break;
      pos = semi + 1;
    }
  }
  const int ACT_SLOT = 8192, N_SLOTS = 8, TILE_BYTES = 18432;
  auto kern = open_mlir_aie(xclbin_path, insts_path);

  size_t w_bytes = 0;
  for (const auto& st : stages) w_bytes += size_t(st.M / 16) * (st.K / 2048) * TILE_BYTES;
  auto bo_w = host_bo(kern, w_bytes, 3);
  auto bo_y = host_bo(kern, size_t(N_SLOTS * ACT_SLOT) * 4, 4);
  auto* w = bo_w.map<uint8_t*>();
  std::memset(w, 0, w_bytes);

  // Host simulation state, kept per stage for verification.
  struct Sim { std::vector<int8_t> wq; std::vector<uint16_t> wd; };
  std::vector<Sim> sims;
  size_t base = 0;
  for (const auto& st : stages) {
    Sim s;
    q4_gen_weights(st.M, st.K, 2024u + unsigned(st.M * 31 + st.K), s.wq, s.wd);
    q4_pack_weights(s.wq, s.wd, st.M, st.K, w + base, false, false);
    base += size_t(st.M / 16) * (st.K / 2048) * TILE_BYTES;
    sims.push_back(std::move(s));
  }
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto* y = bo_y.map<float*>();
  std::memset(y, 0, size_t(N_SLOTS * ACT_SLOT) * 4);
  uint32_t s = 7;
  for (int i = 0; i < 2048; ++i) { s = s * 1664525u + 1013904223u; y[i] = float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto the_run = aie_run(kern);
  the_run.set_arg(3, bo_w);
  the_run.set_arg(4, bo_y);
  auto run_once = [&]() {
    the_run.start();
    return the_run.wait();
  };
  auto state = run_once();
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Host simulation: quantize each stage's input slot the way the core does
  // (f32 math, round half away from zero), then the Q4 GEMV in double.
  std::vector<float> slots(y, y + size_t(N_SLOTS * ACT_SLOT));
  int bad_total = 0;
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
      double err = std::fabs(double(y[size_t(st.out_slot) * ACT_SLOT + r]) - ref) / (mag > 0 ? mag : 1);
      if (!(err <= 1e-4) && r < 3) std::printf("    row %d: npu=%.6f ref=%.6f\n", r, y[size_t(st.out_slot) * ACT_SLOT + r], ref);
      if (!(err <= 1e-4)) ++bad;
      else worst = std::max(worst, err);
      if (std::isnan(err)) ++bad;
    }
    std::printf("  stage %zu (%dx%d -> slot %d): %d of %d rows off (worst ok %.2e) %s\n", si, st.M, st.K,
                st.out_slot, bad, st.M, worst, bad ? "FAIL" : "PASS");
    bad_total += bad;
  }
  std::printf("kernel %s, state=%d, fused layer verify: %s\n", kern.name.c_str(), int(state),
              bad_total ? "FAIL" : "PASS");

  std::vector<xrt::run> one;
  auto t = [&]() {
    std::vector<double> v;
    for (int i = 0; i < 3 + iters; ++i) {
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
