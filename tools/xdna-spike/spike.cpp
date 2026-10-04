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

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

#include <algorithm>
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

int main(int argc, char** argv) {
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
