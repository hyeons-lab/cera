//! Lists the cooperative-matrix configurations (VK_KHR_cooperative_matrix) a Vulkan adapter reports
//! through wgpu, and whether wgpu offers the feature on it.

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!(
        "EXPERIMENTAL_COOPERATIVE_MATRIX offered: {}",
        adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    );
    let props = adapter.cooperative_matrix_properties();
    println!("{} cooperative matrix configuration(s)", props.len());
    for p in props {
        println!("  {p:?}");
    }
}
