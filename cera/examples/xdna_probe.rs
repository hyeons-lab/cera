//! Opens the AMD XDNA NPU through the XRT shim and loads an IRON `.xclbin`
//! with its instruction words. Loading is the support gate: a driver that
//! rejects the xclbin (wrong silicon, wrong XRT) fails here with its reason.
//!
//! Usage: `xdna_probe <kernel.xclbin> <insts.bin>`

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        println!("usage: xdna_probe <kernel.xclbin> <insts.bin>");
        return ExitCode::SUCCESS;
    }
    if args.len() != 3 {
        eprintln!("usage: xdna_probe <kernel.xclbin> <insts.bin>");
        return ExitCode::FAILURE;
    }
    let xclbin = match std::fs::read(&args[1]) {
        Ok(b) if b.is_empty() => {
            eprintln!("{}: xclbin file is empty", args[1]);
            return ExitCode::FAILURE;
        }
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let inst_bytes = match std::fs::read(&args[2]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {}: {e}", args[2]);
            return ExitCode::FAILURE;
        }
    };
    let instr = match cera::backend::xdna::kernels::try_inst_words(&inst_bytes) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{}: {e}", args[2]);
            return ExitCode::FAILURE;
        }
    };
    let device = match cera::backend::xdna::XdnaDevice::open(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xdna unavailable: {e}");
            return ExitCode::FAILURE;
        }
    };
    match device.bdf() {
        Ok(bdf) => println!("device: {bdf}"),
        Err(e) => println!("device: (identifier unavailable: {e})"),
    }
    match device.load_kernel(&xclbin, &instr) {
        Ok(_kernel) => {
            println!(
                "xclbin loaded: {} ({} bytes, {} instruction words)",
                args[1],
                xclbin.len(),
                instr.len()
            );
            println!("xdna probe: PASS");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("xdna unavailable: {e}");
            ExitCode::FAILURE
        }
    }
}
