fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })).unwrap();
    let info = adapter.get_info();
    println!("backend: {:?}", info.backend);
    println!("name: {}", info.name);
    println!("device_type: {:?}", info.device_type);
    println!("vendor: 0x{:x}", info.vendor);
    println!("driver: {}", info.driver);
    println!("driver_info: {}", info.driver_info);
}
