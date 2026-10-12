// XDNA Phase 0b spike, step 1: host-side XRT access and dispatch timing.
//
// Mirrors xrt-smi's NPU latency test (validate xclbin, DPU flow, all-zero
// instruction buffer, opcode 1), but records every run so we can report a
// distribution, and adds xrt::runlist chains of several lengths.
//
// Usage:
//   spike latency <validate.xclbin> <iterations>
//   spike latency-cold <validate.xclbin> <iterations>   (no warmup, as xrt-smi)
//   spike chain   <validate.xclbin> <iterations> <len> [<len> ...]
//   spike stream  <validate.xclbin> <iterations> <len> [<len> ...]
//   spike load    <any.xclbin>
//   spike gemv    <xclbin> <insts.bin> <M> <K> [iters]   (IRON int16 GEMV, verified)
//   spike memcpy  <xclbin> <insts.bin> <n_int32> [iters] (IRON memcpy, verified)
//   spike gemvchain <xclbin> <insts.bin> <M> <K> <iters> <L> [<L>...] (dependent GEMV chains)
//   spike q4gemv  <xclbin> <insts.bin> <M> <K> [iters] (Q4_0-style GEMV, iron/q4gemv, verified)
//   spike readbw  <xclbin> <insts.bin> <n_int32> <workers> [tile] [iters] (read-dominant, verified)

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

static void report(const char* label, std::vector<double> v, double divisor) {
  for (auto& x : v) x /= divisor;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) {
    size_t i = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return v[i];
  };
  double sum = 0;
  for (double x : v) sum += x;
  std::printf("%-28s n=%zu  min=%.1f  p50=%.1f  p90=%.1f  p99=%.1f  max=%.1f  mean=%.1f us\n", label, v.size(),
              v.front(), pct(0.50), pct(0.90), pct(0.99), v.back(), sum / v.size());
}

struct Ctx {
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

static Ctx open_ctx(const char* path) {
  Ctx c;
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
static xrt::run make_run(Ctx& c, std::vector<xrt::bo>& keep) {
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
  auto c = open_ctx(path);
  std::vector<xrt::bo> keep;
  auto run = make_run(c, keep);
  if (warm)
    for (int i = 0; i < 50; ++i) { run.start(); run.wait2(); }  // warmup
  std::vector<double> t;
  t.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait2();
    t.push_back(us_since(t0));
  }
  report(warm ? "single start+wait" : "single, no warmup", t, 1.0);
  return 0;
}

// Runlist: L runs submitted as one list, executed in order on the device.
static int cmd_chain(const char* path, int iters, const std::vector<int>& lens) {
  auto c = open_ctx(path);
  for (int len : lens) {
    std::vector<xrt::bo> keep;
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make_run(c, keep));
    xrt::runlist rl(c.hwctx);
    for (auto& r : runs) rl.add(r);
    for (int i = 0; i < 10; ++i) { rl.execute(); rl.wait(); }
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
      auto t0 = clk::now();
      rl.execute();
      rl.wait();
      t.push_back(us_since(t0));
    }
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
  auto c = open_ctx(path);
  for (int len : lens) {
    std::vector<xrt::bo> keep;
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make_run(c, keep));
    std::vector<double> t;
    for (int i = 0; i < iters + 10; ++i) {
      auto t0 = clk::now();
      for (auto& r : runs) r.start();
      for (auto& r : runs) r.wait2();
      if (i >= 10) t.push_back(us_since(t0));
    }
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
  std::printf("hw_context created in %.1f us\n", us_since(t0));
  auto name = dpu_kernel_name(x);
  if (!name.empty()) {
    xrt::kernel k(ctx, name);
    std::printf("kernel %s opened\n", name.c_str());
  }
  return 0;
}

// Run an IRON-built int16 GEMV (C[M] = A[M,K] . B[K], int32 out) the way the
// mlir-aie host tests submit it: kernel MLIR_AIE, args (opcode 3, instr bo,
// instr word count, A, B, C). Verifies against a CPU reference, then times it.
static int cmd_gemv(const char* xclbin_path, const char* insts_path, int M, int K, int iters) {
  FILE* f = std::fopen(insts_path, "rb");
  if (!f) throw std::runtime_error("cannot open insts file");
  std::vector<uint32_t> instr;
  uint32_t w;
  while (std::fread(&w, 4, 1, f) == 1) instr.push_back(w);
  std::fclose(f);

  auto dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  std::string kname;
  for (auto& k : x.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  xrt::kernel kernel(ctx, kname);
  std::printf("kernel %s, %zu instruction words\n", kname.c_str(), instr.size());

  auto bo_instr = xrt::bo(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(dev, size_t(M) * K * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b = xrt::bo(dev, size_t(K) * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(dev, size_t(M) * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * 4);
  auto* a = bo_a.map<int16_t*>();
  auto* b = bo_b.map<int16_t*>();
  auto* c = bo_c.map<int32_t*>();
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return int16_t(int((s >> 16) % 2001) - 1000); };
  for (size_t i = 0; i < size_t(M) * K; ++i) a[i] = rnd();
  for (int i = 0; i < K; ++i) b[i] = rnd();
  std::memset(c, 0, size_t(M) * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, uint64_t(3));
  run.set_arg(1, bo_instr);
  run.set_arg(2, uint32_t(instr.size()));
  run.set_arg(3, bo_a);
  run.set_arg(4, bo_b);
  run.set_arg(5, bo_c);
  run.start();
  auto state = run.wait();
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::printf("first run state=%d\n", int(state));

  int bad = 0;
  for (int i = 0; i < M; ++i) {
    int32_t ref = 0;
    for (int k = 0; k < K; ++k) ref += int32_t(a[size_t(i) * K + k]) * b[k];
    if (ref != c[i] && bad++ < 5) std::printf("  mismatch C[%d]: npu=%d ref=%d\n", i, c[i], ref);
  }
  std::printf("verify: %d of %d outputs wrong -> %s\n", bad, M, bad ? "FAIL" : "PASS");

  std::vector<double> t;
  for (int i = 0; i < iters + 10; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait2();
    if (i >= 10) t.push_back(us_since(t0));
  }
  report("gemv start+wait", t, 1.0);
  std::sort(t.begin(), t.end());
  double p50 = t[t.size() / 2];
  std::printf("weights %.1f KB, effective weight bandwidth at p50: %.2f GB/s\n", M * K * 2 / 1024.0,
              M * double(K) * 2 / (p50 * 1e3));
  return bad ? 1 : 0;
}

// Run an IRON memcpy design (args: opcode 3, instr bo, instr words, in, out)
// over n int32 elements. Verifies out == in, then reports DDR traffic
// (bytes read + bytes written) per unit time.
static int cmd_memcpy(const char* xclbin_path, const char* insts_path, size_t n, int iters) {
  FILE* f = std::fopen(insts_path, "rb");
  if (!f) throw std::runtime_error("cannot open insts file");
  std::vector<uint32_t> instr;
  uint32_t w;
  while (std::fread(&w, 4, 1, f) == 1) instr.push_back(w);
  std::fclose(f);

  auto dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  std::string kname;
  for (auto& k : x.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  xrt::kernel kernel(ctx, kname);

  auto bo_instr = xrt::bo(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(dev, n * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(dev, n * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * 4);
  auto* in = bo_in.map<int32_t*>();
  auto* out = bo_out.map<int32_t*>();
  for (size_t i = 0; i < n; ++i) in[i] = int32_t(i * 2654435761u);
  std::memset(out, 0, n * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, uint64_t(3));
  run.set_arg(1, bo_instr);
  run.set_arg(2, uint32_t(instr.size()));
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);
  run.start();
  auto state = run.wait();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += out[i] != in[i];
  std::printf("kernel %s, state=%d, verify: %zu of %zu words wrong -> %s\n", kname.c_str(), int(state), bad, n,
              bad ? "FAIL" : "PASS");

  std::vector<double> t;
  for (int i = 0; i < iters + 3; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait2();
    if (i >= 3) t.push_back(us_since(t0));
  }
  report("memcpy start+wait", t, 1.0);
  std::sort(t.begin(), t.end());
  double p50 = t[t.size() / 2], best = t.front();
  std::printf("%.1f MB each way; DDR read+write at p50: %.2f GB/s (best %.2f GB/s); read side alone: %.2f GB/s\n",
              n * 4 / 1048576.0, 2.0 * n * 4 / (p50 * 1e3), 2.0 * n * 4 / (best * 1e3), n * 4 / (p50 * 1e3));
  return bad ? 1 : 0;
}

// Run iron/readbw.py: n int32 words in, `workers` x 16 words out. Worker w
// sums the first word of each tile of its contiguous slice into its
// output word 0. Reports bytes read per unit time (writes are negligible).
static int cmd_readbw(const char* xclbin_path, const char* insts_path, size_t n, int workers, size_t tile, int iters) {
  const size_t out_words = 16;
  FILE* f = std::fopen(insts_path, "rb");
  if (!f) throw std::runtime_error("cannot open insts file");
  std::vector<uint32_t> instr;
  uint32_t w;
  while (std::fread(&w, 4, 1, f) == 1) instr.push_back(w);
  std::fclose(f);

  auto dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  std::string kname;
  for (auto& k : x.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  xrt::kernel kernel(ctx, kname);

  size_t n_out = size_t(workers) * out_words;
  auto bo_instr = xrt::bo(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(dev, n * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(dev, n_out * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * 4);
  auto* in = bo_in.map<int32_t*>();
  auto* out = bo_out.map<int32_t*>();
  for (size_t i = 0; i < n; ++i) in[i] = int32_t((i * 2654435761u) >> 8);
  std::memset(out, 0, n_out * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, uint64_t(3));
  run.set_arg(1, bo_instr);
  run.set_arg(2, uint32_t(instr.size()));
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);
  run.start();
  auto state = run.wait();
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  size_t per_worker = n / workers;
  int bad = 0;
  for (int wk = 0; wk < workers; ++wk) {
    uint32_t ref = 0;  // int32 wraparound, matching the core's add
    for (size_t t = 0; t < per_worker; t += tile) ref += uint32_t(in[wk * per_worker + t]);
    if (int32_t(ref) != out[wk * out_words] && bad++ < 4)
      std::printf("  worker %d: npu=%d ref=%d\n", wk, out[wk * out_words], int32_t(ref));
  }
  std::printf("kernel %s, state=%d, verify: %d of %d workers wrong -> %s\n", kname.c_str(), int(state), bad, workers,
              bad ? "FAIL" : "PASS");

  std::vector<double> t;
  for (int i = 0; i < iters + 3; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait2();
    if (i >= 3) t.push_back(us_since(t0));
  }
  report("readbw start+wait", t, 1.0);
  std::sort(t.begin(), t.end());
  double p50 = t[t.size() / 2], p90 = t[size_t(0.9 * (t.size() - 1) + 0.5)];
  std::printf("%.1f MB read: %.2f GB/s at p50, %.2f GB/s at p90, best %.2f GB/s\n", n * 4 / 1048576.0,
              n * 4 / (p50 * 1e3), n * 4 / (p90 * 1e3), n * 4 / (t.front() * 1e3));
  return bad ? 1 : 0;
}

// Chains of L dependent IRON GEMVs: run i+1 reads run i's output buffer as
// its input vector B (the first K int16 of the int32 C buffer; needs 4M >= 2K).
// Order comes only from in-order execution on one hardware context, which is
// what a backend would rely on. Each chain length is timed as one runlist and
// as L back-to-back start() calls. Only the first link's result is checked.
static int cmd_gemvchain(const char* xclbin_path, const char* insts_path, int M, int K, int iters,
                         const std::vector<int>& lens) {
  if (size_t(M) * 4 < size_t(K) * 2) throw std::runtime_error("need 4*M >= 2*K to chain C into B");
  FILE* f = std::fopen(insts_path, "rb");
  if (!f) throw std::runtime_error("cannot open insts file");
  std::vector<uint32_t> instr;
  uint32_t w;
  while (std::fread(&w, 4, 1, f) == 1) instr.push_back(w);
  std::fclose(f);

  auto dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  std::string kname;
  for (auto& k : x.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  xrt::kernel kernel(ctx, kname);

  auto bo_instr = xrt::bo(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a = xrt::bo(dev, size_t(M) * K * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b0 = xrt::bo(dev, size_t(K) * 2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * 4);
  auto* a = bo_a.map<int16_t*>();
  auto* b0 = bo_b0.map<int16_t*>();
  uint32_t s = 12345;
  auto rnd = [&]() { s = s * 1103515245u + 12345u; return int16_t(int((s >> 16) % 2001) - 1000); };
  for (size_t i = 0; i < size_t(M) * K; ++i) a[i] = rnd();
  for (int i = 0; i < K; ++i) b0[i] = rnd();
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b0.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int max_len = *std::max_element(lens.begin(), lens.end());
  std::vector<xrt::bo> bo_c;
  for (int i = 0; i < max_len; ++i) {
    bo_c.push_back(xrt::bo(dev, size_t(M) * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5)));
    std::memset(bo_c.back().map<void*>(), 0, size_t(M) * 4);
    bo_c.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  auto make = [&](int i) {
    xrt::run r(kernel);
    r.set_arg(0, uint64_t(3));
    r.set_arg(1, bo_instr);
    r.set_arg(2, uint32_t(instr.size()));
    r.set_arg(3, bo_a);
    if (i == 0) r.set_arg(4, bo_b0);
    else r.set_arg(4, bo_c[i - 1]);  // input vector = previous link's output
    r.set_arg(5, bo_c[i]);
    return r;
  };

  {  // check link 0
    auto r = make(0);
    r.start();
    r.wait2();
    bo_c[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* c = bo_c[0].map<int32_t*>();
    int bad = 0;
    for (int i = 0; i < M; ++i) {
      int32_t ref = 0;
      for (int k = 0; k < K; ++k) ref += int32_t(a[size_t(i) * K + k]) * b0[k];
      bad += ref != c[i];
    }
    std::printf("link 0 verify: %d of %d wrong -> %s\n", bad, M, bad ? "FAIL" : "PASS");
  }

  for (int len : lens) {
    std::vector<xrt::run> runs;
    for (int i = 0; i < len; ++i) runs.push_back(make(i));
    xrt::runlist rl(ctx);
    for (auto& r : runs) rl.add(r);
    std::vector<double> t_rl, t_st;
    for (int it = 0; it < iters + 3; ++it) {
      auto t0 = clk::now();
      rl.execute();
      rl.wait();
      if (it >= 3) t_rl.push_back(us_since(t0));
    }
    // Plain submission needs fresh run objects; a run in a runlist is owned by it.
    std::vector<xrt::run> runs2;
    for (int i = 0; i < len; ++i) runs2.push_back(make(i));
    for (int it = 0; it < iters + 3; ++it) {
      auto t0 = clk::now();
      for (auto& r : runs2) r.start();
      for (auto& r : runs2) r.wait2();
      if (it >= 3) t_st.push_back(us_since(t0));
    }
    char label[64];
    std::snprintf(label, sizeof label, "dep runlist L=%d (per run)", len);
    report(label, t_rl, len);
    std::snprintf(label, sizeof label, "dep stream L=%d (per run)", len);
    report(label, t_st, len);
  }
  return 0;
}

// Q4_0-style GEMV (iron/q4gemv): weights int4 (q - 8) with a bf16 scale per
// 32-weight block, x int8 with an f32 scale per 32-element block. The host
// quantizes random data the way llama.cpp's Q4_0 / Q8_0 do, packs one stream
// per worker in consumption order, and checks the NPU against a CPU reference
// that uses the same quantized values (so a mismatch is the kernel's, not
// quantization error).
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
  const int QB = 32, ROWS = 16, WORKERS = 16;
  const size_t tb = size_t(K / QB) * (2 * 128 + ROWS * 2);
  if (M % (WORKERS * ROWS) || K % QB) throw std::runtime_error("bad M or K");
  FILE* f = std::fopen(insts_path, "rb");
  if (!f) throw std::runtime_error("cannot open insts file");
  std::vector<uint32_t> instr;
  uint32_t word;
  while (std::fread(&word, 4, 1, f) == 1) instr.push_back(word);
  std::fclose(f);

  // Random weights and activations, quantized like Q4_0 / Q8_0.
  uint32_t s = 2024;
  auto frand = [&]() { s = s * 1664525u + 1013904223u; return float((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
  int nb = K / QB;
  std::vector<int8_t> wq(size_t(M) * K);    // q - 8, in [-8, 7]
  std::vector<uint16_t> wd(size_t(M) * nb);  // bf16 scales
  std::vector<float> wrow(K);
  for (int r = 0; r < M; ++r) {
    for (int k = 0; k < K; ++k) wrow[k] = frand();
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

  auto dev = xrt::device(0);
  xrt::xclbin x{std::string(xclbin_path)};
  std::string kname;
  for (auto& k : x.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  if (kname.empty()) throw std::runtime_error("no MLIR_AIE kernel in xclbin");
  dev.register_xclbin(x);
  xrt::hw_context ctx(dev, x.get_uuid());
  xrt::kernel kernel(ctx, kname);

  const size_t w_bytes = size_t(M / ROWS) * tb;
  auto bo_instr = xrt::bo(dev, instr.size() * 4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_w = xrt::bo(dev, w_bytes, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_x = xrt::bo(dev, tb, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_y = xrt::bo(dev, size_t(M) * 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  std::memcpy(bo_instr.map<void*>(), instr.data(), instr.size() * 4);

  // Pack: worker w owns rows [w*rpw, (w+1)*rpw); its tiles are contiguous.
  // A tile holds, for every block b, two 16x16 int4 B matrices (B[k][n] =
  // W[r0+n][b*32 + h*16 + k], element e = k*16+n, low nibble first), then the
  // 16 bf16 row scales of every block.
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
  auto* w = bo_w.map<uint8_t*>();
  std::memset(w, 0, w_bytes);
  for (int tile = 0; tile < M / ROWS; ++tile) {
    int r0 = tile * ROWS;
    uint8_t* tp = w + size_t(tile) * tb;
    for (int b = 0; b < nb; ++b) {
      uint8_t* bp = tp + size_t(b) * 256;  // weights for all blocks first, then all scales
      for (int h = 0; h < 2; ++h)
        for (int k = 0; k < 16; ++k)
          for (int n = 0; n < 16; ++n) {
            int e = bt ? n * 16 + k : k * 16 + n;
            uint8_t nib = uint8_t(wq[size_t(r0 + n) * K + b * QB + h * 16 + k]) & 0xF;
            bool high = ((e & 1) != 0) != hi_first;
            bp[h * 128 + e / 2] |= high ? uint8_t(nib << 4) : nib;
          }
      auto* sp = reinterpret_cast<uint16_t*>(tp + size_t(nb) * 256 + size_t(b) * ROWS * 2);
      for (int n = 0; n < 16; ++n) sp[n] = wd[size_t(r0 + n) * nb + b];
    }
  }
  auto* xs = bo_x.map<uint8_t*>();
  std::memset(xs, 0, tb);
  std::memcpy(xs, xq.data(), K);
  std::memcpy(xs + K, xd.data(), nb * 4);
  auto* y = bo_y.map<float*>();
  std::memset(y, 0, size_t(M) * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, uint64_t(3));
  run.set_arg(1, bo_instr);
  run.set_arg(2, uint32_t(instr.size()));
  run.set_arg(3, bo_w);
  run.set_arg(4, bo_x);
  run.set_arg(5, bo_y);
  run.start();
  auto state = run.wait();
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference from the same quantized values; error relative to the row's
  // magnitude scale (sum of |terms|), since f32 accumulation order differs.
  int bad = 0;
  double worst = 0;
  for (int r = 0; r < M; ++r) {
    double ref = 0, mag = 0;
    for (int b = 0; b < nb; ++b) {
      int32_t dot = 0;
      for (int j = 0; j < QB; ++j) dot += int32_t(wq[size_t(r) * K + b * QB + j]) * xq[b * QB + j];
      double term = double(dot) * bf16_to_f32(wd[size_t(r) * nb + b]) * xd[b];
      ref += term;
      mag += std::fabs(term);
    }
    double err = std::fabs(double(y[r]) - ref) / (mag > 0 ? mag : 1);
    worst = std::max(worst, err);
    if (err > 1e-4 && bad++ < 5) std::printf("  row %d: npu=%.6f ref=%.6f rel=%.2e\n", r, y[r], ref, err);
  }
  std::printf("kernel %s, state=%d, verify: %d of %d rows off by >1e-4 of row magnitude (worst %.2e) -> %s\n",
              kname.c_str(), int(state), bad, M, worst, bad ? "FAIL" : "PASS");

  // SPIKE_Q4_STREAM=L keeps L runs in flight (all started, then all waited),
  // so the NPU never idles between runs; times are reported per run.
  int depth = std::getenv("SPIKE_Q4_STREAM") ? std::atoi(std::getenv("SPIKE_Q4_STREAM")) : 1;
  std::vector<xrt::run> runs;
  for (int i = 0; i < std::max(depth, 1); ++i) {
    xrt::run r(kernel);
    r.set_arg(0, uint64_t(3));
    r.set_arg(1, bo_instr);
    r.set_arg(2, uint32_t(instr.size()));
    r.set_arg(3, bo_w);
    r.set_arg(4, bo_x);
    r.set_arg(5, bo_y);
    runs.push_back(r);
  }
  std::vector<double> t;
  for (int i = 0; i < iters + 3; ++i) {
    auto t0 = clk::now();
    for (auto& r : runs) r.start();
    for (auto& r : runs) r.wait2();
    if (i >= 3) t.push_back(us_since(t0) / runs.size());
  }
  report(depth > 1 ? "q4gemv streamed (per run)" : "q4gemv start+wait", t, 1.0);
  if (std::getenv("SPIKE_Q4_TRACE")) {  // per-run times in order, to see where the tail comes from
    for (size_t i = 0; i < t.size(); ++i) std::printf("%s%.0f", i % 20 ? " " : "\n  ", t[i]);
    std::printf("\n");
  }
  std::sort(t.begin(), t.end());
  double p50 = t[t.size() / 2], p90 = t[size_t(0.9 * (t.size() - 1) + 0.5)];
  std::printf("weights %.2f MB (4.5 bits/weight incl. bf16 scales): %.2f GB/s at p50, %.2f at p90, best %.2f\n",
              w_bytes / 1048576.0, w_bytes / (p50 * 1e3), w_bytes / (p90 * 1e3), w_bytes / (t.front() * 1e3));
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc >= 6 && std::string(argv[1]) == "q4gemv") {
    try {
      return cmd_q4gemv(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]), argc > 6 ? std::atoi(argv[6]) : 200);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc >= 8 && std::string(argv[1]) == "gemvchain") {
    try {
      std::vector<int> lens;
      for (int i = 7; i < argc; ++i) lens.push_back(std::atoi(argv[i]));
      return cmd_gemvchain(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), lens);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc >= 6 && std::string(argv[1]) == "readbw") {
    try {
      return cmd_readbw(argv[2], argv[3], std::strtoull(argv[4], nullptr, 10), std::atoi(argv[5]),
                        argc > 6 ? std::strtoull(argv[6], nullptr, 10) : 1024, argc > 7 ? std::atoi(argv[7]) : 50);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc >= 5 && std::string(argv[1]) == "memcpy") {
    try {
      return cmd_memcpy(argv[2], argv[3], std::strtoull(argv[4], nullptr, 10), argc > 5 ? std::atoi(argv[5]) : 50);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc >= 2 && std::string(argv[1]) == "gemv") {
    if (argc < 6) {
      std::fprintf(stderr, "usage: spike gemv <xclbin> <insts.bin> <M> <K> [iters]\n");
      return 2;
    }
    try {
      return cmd_gemv(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]), argc > 6 ? std::atoi(argv[6]) : 1000);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      return 1;
    }
  }
  if (argc < 3) {
    std::fprintf(stderr, "usage: spike latency|chain|stream|load <xclbin> [iters] [lens...]\n");
    return 2;
  }
  try {
    std::string cmd = argv[1];
    if (cmd == "load") return cmd_load(argv[2]);
    int iters = argc > 3 ? std::atoi(argv[3]) : 1000;
    std::vector<int> lens;
    for (int i = 4; i < argc; ++i) lens.push_back(std::atoi(argv[i]));
    if (cmd == "latency") return cmd_latency(argv[2], iters, true);
    if (cmd == "latency-cold") return cmd_latency(argv[2], iters, false);
    if (cmd == "chain") return cmd_chain(argv[2], iters, lens);
    if (cmd == "stream") return cmd_stream(argv[2], iters, lens);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 2;
}
