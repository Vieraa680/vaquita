#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use tauri::{LogicalSize, Manager, Size};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Serialize)]
struct ChatRequest {
    prompt: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct IntentInfo {
    label: String,
    confidence: f64,
    is_confident: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ChatResponse {
    response: Option<String>,
    intent: Option<IntentInfo>,
    error: Option<String>,
}

#[derive(Serialize)]
struct VoiceRequest {
    record_seconds: f64,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct VoiceResponse {
    transcribed_prompt: Option<String>,
    response: Option<String>,
    intent: Option<IntentInfo>,
    error: Option<String>,
}

#[tauri::command]
async fn send_query(prompt: String) -> Result<ChatResponse, String> {
    let client = reqwest::Client::new();
    let body = ChatRequest { prompt };

    let resp = client
        .post("http://127.0.0.1:8765/api/chat")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Cannot reach Vaquita daemon at 127.0.0.1:8765: {e}"))?;

    let result = resp
        .json::<ChatResponse>()
        .await
        .map_err(|e| format!("Failed to parse daemon response: {e}"))?;

    Ok(result)
}

#[tauri::command]
async fn record_voice(seconds: Option<f64>) -> Result<VoiceResponse, String> {
    let client = reqwest::Client::new();
    let body = VoiceRequest {
        record_seconds: seconds.unwrap_or(4.0),
    };

    let resp = client
        .post("http://127.0.0.1:8765/api/voice")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Cannot reach Vaquita daemon at 127.0.0.1:8765: {e}"))?;

    let result = resp
        .json::<VoiceResponse>()
        .await
        .map_err(|e| format!("Failed to parse voice response: {e}"))?;

    Ok(result)
}

#[tauri::command]
fn hide_window(app: tauri::AppHandle) {
    hide_hud(&app);
}

#[tauri::command]
fn set_window_height(app: tauri::AppHandle, height: f64) {
    if let Some(window) = app.get_webview_window("main") {
        let clamped = height.clamp(68.0, 560.0);
        let _ = window.set_size(Size::Logical(LogicalSize {
            width: 640.0,
            height: clamped,
        }));
    }
}

fn show_hud(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn hide_hud(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn toggle_hud(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let is_visible = window.is_visible().unwrap_or(false);
        if is_visible {
            hide_hud(app);
        } else {
            show_hud(app);
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let handle = app.handle().clone();

            // Single-port IPC trigger on 127.0.0.1:8766 for instant Wayland / Hyprland keybinds
            let handle_ipc = handle.clone();
            tauri::async_runtime::spawn(async move {
                if let Ok(listener) = TcpListener::bind("127.0.0.1:8766").await {
                    while let Ok((mut stream, _)) = listener.accept().await {
                        let mut buf = [0u8; 128];
                        let n = stream.read(&mut buf).await.unwrap_or(0);
                        let req_str = String::from_utf8_lossy(&buf[..n]);
                        if req_str.contains("/show") {
                            show_hud(&handle_ipc);
                        } else if req_str.contains("/hide") {
                            hide_hud(&handle_ipc);
                        } else {
                            toggle_hud(&handle_ipc);
                        }
                        let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK";
                        let _ = stream.write_all(resp).await;
                    }
                }
            });

            // Register global shortcuts for environments that support direct grabbing
            for shortcut in ["Super+Space", "Ctrl+Space", "Alt+Space", "Ctrl+Alt+Space"] {
                let handle_clone = handle.clone();
                let _ = app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        toggle_hud(&handle_clone);
                    }
                });
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            send_query,
            record_voice,
            hide_window,
            set_window_height
        ])
        .run(tauri::generate_context!())
        .expect("error while running Vaquita application");
}
