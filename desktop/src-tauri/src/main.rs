#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

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
fn hide_window(window: tauri::Window) {
    let _ = window.hide();
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![send_query, hide_window])
        .run(tauri::generate_context!())
        .expect("error while running Vaquita HUD application");
}
