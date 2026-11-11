//! Prints what a window surface supports for transparency (composite alpha) on each
//! GPU: whether a translucent, OS-blurred window can be presented with Vulkan.
use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App;
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = el.create_window(Window::default_attributes().with_visible(false).with_transparent(true)).unwrap();
        let ctx = brp_render::GpuContext::new(brp_render::GpuInstance::new().unwrap(), &w, None).unwrap();
        unsafe {
            let caps = ctx.surface_fn.get_physical_device_surface_capabilities(ctx.physical, ctx.surface).unwrap();
            let formats = ctx.surface_fn.get_physical_device_surface_formats(ctx.physical, ctx.surface).unwrap();
            println!("{}: composite alpha {:?}", ctx.device_name, caps.supported_composite_alpha);
            println!("formats: {:?}", formats.iter().map(|f| f.format).collect::<Vec<_>>());
            let _ = vk::CompositeAlphaFlagsKHR::OPAQUE;
        }
        el.exit();
    }
    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

fn main() {
    EventLoop::new().unwrap().run_app(&mut App).unwrap();
}
