//! The desktop shell. It only hosts the UI and keeps the local service running;
//! all file access stays in the service so the UI can later run on the web.

use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};

/// Must match the default in `service/pixora_service/__main__.py` and `src/api.ts`.
const SERVICE_PORT: u16 = 8732;

/// The service process, when this shell started it.
struct Service(Mutex<Option<Child>>);

fn service_is_up() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], SERVICE_PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn service_command() -> Option<Command> {
    if let Ok(exe) = std::env::var("PIXORA_SERVICE_EXE") {
        return Some(Command::new(exe));
    }

    // Development: run the service from the repository's virtual environment.
    #[cfg(debug_assertions)]
    {
        let python = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../service/.venv/Scripts/python.exe");
        if python.exists() {
            let mut cmd = Command::new(python);
            cmd.args(["-m", "pixora_service"]);
            // Model files downloaded for development live in the repository folder.
            let models = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../service/.models");
            if models.exists() {
                cmd.arg("--models-dir").arg(models);
            }
            return Some(cmd);
        }
    }

    // Installed: a packaged service next to the app (built in the release phase).
    let exe = std::env::current_exe()
        .ok()?
        .parent()?
        .join("pixora-service.exe");
    exe.exists().then(|| Command::new(exe))
}

fn start_service() -> Option<Child> {
    if service_is_up() {
        // Already running, for example started by hand during development.
        return None;
    }
    let Some(mut cmd) = service_command() else {
        eprintln!("pixora: no local service found to start");
        return None;
    };
    cmd.args(["--port", &SERVICE_PORT.to_string()]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    match cmd.spawn() {
        Ok(child) => Some(child),
        Err(err) => {
            eprintln!("pixora: could not start the local service: {err}");
            None
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(Service(Mutex::new(None)))
        .setup(|app| {
            // The UI shows "service is not running" until the port answers, so
            // there is no need to block the window on startup.
            *app.state::<Service>().0.lock().unwrap() = start_service();
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Pixora shell")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Scan progress lives in the database, so stopping here is safe:
                // the next start resumes where this one left off.
                if let Some(mut child) = app.state::<Service>().0.lock().unwrap().take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
