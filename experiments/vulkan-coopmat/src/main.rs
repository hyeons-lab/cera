//! Cooperative-matrix capability probe.
//!
//! Enumerates every Vulkan physical device and, for those advertising
//! `VK_KHR_cooperative_matrix`, dumps the supported cooperative-matrix
//! configurations (tile M/N/K + A/B/C/Result component types + scope). This
//! replaces `vulkaninfo`, which is not shipped on the stock Pixel image, and tells
//! us whether to author the GEMM microbench with int8 or f16 coopmat.
//!
//! Pure `ash`; no device or shader is created — the query is a physical-device
//! function, guarded behind an extension-advertised check so it never dereferences
//! a null function pointer on adapters without the extension.

use ash::vk;
use std::ffi::CStr;

fn main() {
    unsafe { run() }
}

unsafe fn run() {
    let entry = match ash::Entry::load() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to load Vulkan loader (libvulkan.so): {e:?}");
            std::process::exit(1);
        }
    };

    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = entry
        .create_instance(&ci, None)
        .expect("vkCreateInstance failed");

    let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);

    let pdevs = instance
        .enumerate_physical_devices()
        .expect("enumerate_physical_devices failed");
    println!("physical devices: {}", pdevs.len());

    for pd in pdevs {
        let props = instance.get_physical_device_properties(pd);
        let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
        println!(
            "\n=== {name} (Vulkan {}.{}.{}) ===",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version),
        );

        let exts = instance
            .enumerate_device_extension_properties(pd)
            .unwrap_or_default();
        let has_cm = exts.iter().any(|e| {
            CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
        });
        println!("VK_KHR_cooperative_matrix advertised: {has_cm}");
        if !has_cm {
            continue;
        }

        match cm.get_physical_device_cooperative_matrix_properties(pd) {
            Ok(list) => {
                println!("cooperative-matrix configs: {}", list.len());
                for (i, p) in list.iter().enumerate() {
                    println!(
                        "  [{i:2}] M{:>3} x N{:>3} x K{:>3}  \
                         A={:?} B={:?} C={:?} Result={:?}  scope={:?}  sat_accum={}",
                        p.m_size,
                        p.n_size,
                        p.k_size,
                        p.a_type,
                        p.b_type,
                        p.c_type,
                        p.result_type,
                        p.scope,
                        p.saturating_accumulation != 0,
                    );
                }
            }
            Err(e) => println!("cooperative-matrix properties query failed: {e:?}"),
        }
    }

    instance.destroy_instance(None);
}
