//! Opens the AMD XDNA NPU through the XRT shim and loads an IRON `.xclbin`
//! with its instruction words. Loading is the support gate: a driver that
//! rejects the xclbin (wrong silicon, wrong XRT) fails here with its reason.
//!
//! Usage: `xdna_probe <kernel.xclbin> <insts.bin>`

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: xdna_probe <kernel.xclbin> <insts.bin>");
        return ExitCode::FAILURE;
    }
    let xclbin = match std::fs::read(&args[1]) {
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
    if inst_bytes.len() % 4 != 0 {
        eprintln!(
            "{}: instruction file is not a whole number of words",
            args[2]
        );
        return ExitCode::FAILURE;
    }
    let instr: Vec<u32> = (0..inst_bytes.len() / 4)
        .map(|i| {
            u32::from_le_bytes(
                inst_bytes[4 * i..4 * i + 4]
                    .try_into()
                    .expect("4-byte window"),
            )
        })
        .collect();
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
