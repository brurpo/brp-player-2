//! A window never waits for the GPU on its own thread (its mouse and keyboard would
//! wait too: a slider being dragged stopped following the mouse while effects
//! computed). When a frame's GPU work is not done yet, the window skips the frame and
//! this thread waits instead, then asks the window to draw again.

use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Asks the window to draw again (`Window::request_redraw`, callable from any thread).
pub type Waker = Arc<dyn Fn() + Send + Sync>;

pub struct GpuWaiter {
    tx: Option<crossbeam_channel::Sender<Option<vk::Fence>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// A wait is under way (one at a time is enough: the window draws again after it).
    pending: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl GpuWaiter {
    pub fn new(device: ash::Device) -> GpuWaiter {
        let (tx, rx) = crossbeam_channel::unbounded::<Option<vk::Fence>>();
        let pending = Arc::new(AtomicBool::new(false));
        let waker: Arc<Mutex<Option<Waker>>> = Arc::new(Mutex::new(None));
        let (p, w) = (pending.clone(), waker.clone());
        let thread = std::thread::Builder::new()
            .name("brp-gpu-wait".into())
            .spawn(move || {
                for msg in rx {
                    match msg {
                        // Wait for the frame's work (in steps, so a fence the window
                        // reset meanwhile does not hold this thread for long).
                        Some(f) => {
                            for _ in 0..100 {
                                match unsafe { device.wait_for_fences(&[f], true, 20_000_000) } {
                                    Err(vk::Result::TIMEOUT) => continue,
                                    _ => break,
                                }
                            }
                        }
                        // No swapchain image free yet: try again shortly.
                        None => std::thread::sleep(std::time::Duration::from_millis(2)),
                    }
                    p.store(false, Ordering::Release);
                    let wake = w.lock().unwrap().clone();
                    if let Some(wake) = wake {
                        wake();
                    }
                }
            })
            .ok();
        GpuWaiter { tx: Some(tx), thread, pending, waker }
    }

    pub fn set_waker(&self, w: Waker) {
        *self.waker.lock().unwrap() = Some(w);
    }

    /// Wake the window once `fence` is signalled (None: in a moment).
    pub fn wake_when(&self, fence: Option<vk::Fence>) {
        if !self.pending.swap(true, Ordering::AcqRel) {
            if let Some(tx) = &self.tx {
                let _ = tx.send(fence);
            }
        }
    }

    /// End the thread (before the fences it may wait on are destroyed).
    pub fn stop(&mut self) {
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for GpuWaiter {
    fn drop(&mut self) {
        self.stop();
    }
}
