//! A very small HTTP/1.1 server: enough to speak the OpenAI API and no more.
//!
//! Hand-rolled rather than pulled in, because the whole surface needed is
//! "parse a request line, read a JSON body, write a response or a stream of
//! them". Everything is bounded — headers, body, and the request line — since
//! this is the only code in the project that reads from a socket.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

const MAX_LINE: usize = 8 * 1024;
const MAX_HEADERS: usize = 64;
const MAX_BODY: usize = 16 * 1024 * 1024;

pub struct Request {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

impl Request {
    /// Path without its query string.
    pub fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or(&self.path)
    }
}

/// The response side of one connection.
pub struct Conn {
    stream: TcpStream,
    pub cors: bool,
}

impl Conn {
    fn head(&mut self, status: u16, content_type: &str, extra: &str) -> io::Result<()> {
        let reason = match status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            405 => "Method Not Allowed",
            413 => "Payload Too Large",
            503 => "Service Unavailable",
            _ => "Error",
        };
        let cors = if self.cors {
            "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        } else {
            ""
        };
        write!(
            self.stream,
            "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n{cors}{extra}Connection: close\r\n\r\n"
        )
    }

    pub fn json(&mut self, status: u16, body: &str) -> io::Result<()> {
        self.head(status, "application/json", &format!("Content-Length: {}\r\n", body.len()))?;
        self.stream.write_all(body.as_bytes())?;
        self.stream.flush()
    }

    pub fn text(&mut self, status: u16, body: &str) -> io::Result<()> {
        self.head(status, "text/plain; charset=utf-8", &format!("Content-Length: {}\r\n", body.len()))?;
        self.stream.write_all(body.as_bytes())?;
        self.stream.flush()
    }

    /// An error in the shape OpenAI clients expect to find it.
    pub fn error(&mut self, status: u16, message: &str) -> io::Result<()> {
        let kind = if status == 503 { "server_error" } else { "invalid_request_error" };
        let body = serde_json::json!({"error": {"message": message, "type": kind}});
        self.json(status, &body.to_string())
    }

    /// Begin a server-sent-event stream.
    pub fn sse_open(&mut self) -> io::Result<()> {
        self.head(200, "text/event-stream", "Cache-Control: no-cache\r\n")?;
        self.stream.flush()
    }

    pub fn sse(&mut self, data: &str) -> io::Result<()> {
        write!(self.stream, "data: {data}\n\n")?;
        self.stream.flush()
    }

    pub fn sse_close(&mut self) -> io::Result<()> {
        write!(self.stream, "data: [DONE]\n\n")?;
        self.stream.flush()
    }
}

fn read_line(r: &mut impl BufRead) -> io::Result<String> {
    let mut line = String::new();
    let n = r.take(MAX_LINE as u64).read_line(&mut line)?;
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "closed"));
    }
    if n >= MAX_LINE {
        return Err(io::Error::other("header line too long"));
    }
    Ok(line.trim_end().to_string())
}

fn parse(stream: &TcpStream) -> io::Result<Request> {
    let mut r = BufReader::new(stream);
    let start = read_line(&mut r)?;
    let mut parts = start.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || path.is_empty() {
        return Err(io::Error::other("malformed request line"));
    }

    let mut len = 0usize;
    for _ in 0..MAX_HEADERS {
        let line = read_line(&mut r)?;
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().map_err(|_| io::Error::other("bad content-length"))?;
            } else if k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked") {
                return Err(io::Error::other("chunked request bodies are not supported"));
            }
        }
    }
    if len > MAX_BODY {
        return Err(io::Error::other("body too large"));
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, body })
}

pub fn bind(host: &str, port: u16) -> io::Result<TcpListener> {
    TcpListener::bind((host, port))
}

/// Serve until the listener dies. One thread per connection; the handler is
/// shared, so whatever serialisation the handler needs is the handler's to do.
pub fn run<H>(listener: TcpListener, handler: H)
where
    H: Fn(&Request, &mut Conn) + Send + Sync + 'static,
{
    let handler = Arc::new(handler);
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let handler = handler.clone();
        std::thread::spawn(move || {
            let mut conn = Conn {
                stream: match stream.try_clone() {
                    Ok(s) => s,
                    Err(_) => return,
                },
                cors: false,
            };
            match parse(&stream) {
                Ok(req) => handler(&req, &mut conn),
                // A malformed request gets one line and the door.
                Err(e) if e.kind() != io::ErrorKind::UnexpectedEof => {
                    let _ = conn.error(400, &e.to_string());
                }
                Err(_) => {}
            }
        });
    }
}
