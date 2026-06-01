//! Vite child-process supervisor.
//!
//! Runs `pnpm exec vite --port <port> --strictPort` (or `--no-vite` to skip).
//! We don't reach for Node.js APIs — Vite is invoked through the local
//! `node_modules/.bin/vite` binary so the user's existing pnpm-managed
//! deps are reused.

use std::path::PathBuf;
use std::process::Stdio;

use tokio::process::{Child, Command};

pub struct ViteHandle {
    child: Child,
    port: u16,
}

impl ViteHandle {
    pub async fn shutdown(&mut self) {
        // Be polite: SIGTERM first, escalate to kill if it lingers.
        #[cfg(unix)]
        {
            if let Some(pid) = self.child.id() {
                // SIGTERM
                let _ = unsafe { libc_kill(pid as i32, 15) };
            }
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(3), self.child.wait()).await;
        let _ = self.child.kill().await;
        tracing::info!("[vite:{}] terminated", self.port);
    }
}

/// On unix, send a signal via the libc syscall. We don't link libc directly
/// to keep deps small; this calls through the syscall ABI.
#[cfg(unix)]
unsafe fn libc_kill(pid: i32, sig: i32) -> i32 {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    unsafe { kill(pid, sig) }
}

pub async fn spawn_vite(web_dir: PathBuf, port: u16) -> std::io::Result<ViteHandle> {
    tracing::info!("[vite:{port}] spawning in {}", web_dir.display());
    let mut cmd = Command::new("pnpm");
    cmd.current_dir(&web_dir)
        .arg("exec")
        .arg("vite")
        .arg("--port")
        .arg(port.to_string())
        .arg("--strictPort")
        // Vite needs to know which host to advertise to HMR clients —
        // since our devserver loads pages from :25321 but HMR connects
        // directly to :5173, set clientPort here too.
        .env("VITE_HMR_CLIENT_PORT", port.to_string())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let child = cmd.spawn()?;
    // Best-effort wait for Vite to be ready by sleeping briefly. The proxy
    // surfaces 502s with a clear message if Vite hasn't bound yet on the
    // first request.
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    Ok(ViteHandle { child, port })
}
