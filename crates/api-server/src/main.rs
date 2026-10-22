use api_server::AppServerState;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let state = AppServerState::new();
    let listener = TcpListener::bind(&addr).await?;
    println!("Calorie-Counter AI Server & Test UI listening on http://{}", addr);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let app = state.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 65536];
            let n = match socket.read(&mut buf).await {
                Ok(n) if n > 0 => n,
                _ => return,
            };
            let raw_req = String::from_utf8_lossy(&buf[..n]);
            let mut lines = raw_req.lines();
            let first_line = lines.next().unwrap_or("");
            let parts: Vec<&str> = first_line.split_whitespace().collect();
            if parts.len() < 2 {
                return;
            }
            let method = parts[0];
            let path = parts[1];
            let body = raw_req
                .split("\r\n\r\n")
                .nth(1)
                .or_else(|| raw_req.split("\n\n").nth(1))
                .unwrap_or("");

            let (status, content_type, resp_body) =
                app.handle_http_request(method, path, body).await;
            let status_text = match status {
                200 => "OK",
                201 => "Created",
                400 => "Bad Request",
                404 => "Not Found",
                _ => "OK",
            };
            let http_resp = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                status,
                status_text,
                content_type,
                resp_body.len(),
                resp_body
            );
            let _ = socket.write_all(http_resp.as_bytes()).await;
        });
    }
}
