
#!/usr/bin/env python3
"""Cybertech - the command and background service for The Cybertech Engine.

An Ollama-style front end. One program that can

    cybertech pull qwen2.5:3b      download a model
    cybertech list                 show downloaded models
    cybertech run qwen2.5:3b       chat in the terminal (pulls first if needed)
    cybertech rm qwen2.5:3b        delete a model
    cybertech serve                run the background service (default when double-clicked)

The service listens on http://127.0.0.1:11434 and speaks the Ollama API
(/api/tags /api/pull /api/chat /api/generate ...) and the OpenAI API (/v1/...),
loading whichever pulled model a request names. The models themselves are run by
cybertech-engine, which must sit in the same folder.

Standard library only, so it packages into a single exe with PyInstaller.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "0.1.0"
HOST = os.environ.get("CYBERTECH_HOST", "127.0.0.1")
PORT = int(os.environ.get("CYBERTECH_PORT", "11434"))
KEEP_ALIVE = int(os.environ.get("CYBERTECH_KEEP_ALIVE", "600"))      # seconds idle before unloading
LOAD_TIMEOUT = int(os.environ.get("CYBERTECH_LOAD_TIMEOUT", "900"))  # seconds allowed to load a model
DEFAULT_QUANT = "Q4_K_M"
WIN = os.name == "nt"

# Short Ollama-style names -> names the engine's `pull` understands.
EXTRA_ALIASES = {
    "qwen2.5": "qwen2.5:3b",
    "qwen2.5-coder": "qwen2.5-coder:7b",
    "llama3.2": "llama3.2:3b",
    "llama3.1": "llama3.1:8b",
}


# ----------------------------------------------------------------------------- paths

def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def engine_path():
    env = os.environ.get("CYBERTECH_ENGINE")
    if env and os.path.exists(env):
        return env
    exe = "cybertech-engine.exe" if WIN else "cybertech-engine"
    for d in (app_dir(), os.path.join(app_dir(), "engine")):
        p = os.path.join(d, exe)
        if os.path.exists(p):
            return p
    return shutil.which("cybertech-engine")


def models_dir():
    d = os.environ.get("CYBERTECH_MODELS")
    if d:
        return d
    if WIN and os.environ.get("LOCALAPPDATA"):
        return os.path.join(os.environ["LOCALAPPDATA"], "CybertechEngine", "models")
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        return os.path.join(home, ".cybertech-engine", "models")
    return "models_gguf"


def sanitize(name):  # must match pull.rs
    return "".join(c if (c.isascii() and c.isalnum()) or c in "._-" else "-" for c in name)


def folder_for(name, quant=None):  # must match pull.rs dir_name
    q = quant or DEFAULT_QUANT
    if q.lower() == DEFAULT_QUANT.lower():
        return sanitize(name)
    return f"{sanitize(name)}-{sanitize(q)}"


def gguf_files(folder):
    try:
        return sorted(f for f in os.listdir(folder) if f.lower().endswith(".gguf"))
    except OSError:
        return []


def registry_path():
    return os.path.join(models_dir(), "_names.json")


def load_registry():
    try:
        with open(registry_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_registry(reg):
    try:
        os.makedirs(models_dir(), exist_ok=True)
        with open(registry_path(), "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=1)
    except OSError:
        pass


def list_models():
    """[{folder, name, size, modified}] for every folder in the models dir holding a .gguf."""
    root, reg, out = models_dir(), load_registry(), []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for e in entries:
        p = os.path.join(root, e)
        files = gguf_files(p) if os.path.isdir(p) else []
        if not files:
            continue
        size = sum(os.path.getsize(os.path.join(p, f)) for f in files)
        out.append({"folder": e, "name": reg.get(e, e), "size": size,
                    "modified": datetime.fromtimestamp(os.path.getmtime(p), timezone.utc)})
    return out


def resolve(name):
    """A user-given model name (qwen2.5:3b, qwen2.5-3b, qwen2.5:3b:latest ...) -> folder, or None."""
    if not name:
        return None
    n = name.strip()
    if n.lower().endswith(":latest"):
        n = n[: -len(":latest")]
    n = EXTRA_ALIASES.get(n.lower(), n)
    models = list_models()
    folders = {m["folder"] for m in models}
    for cand in (n, sanitize(n)):
        if cand in folders:
            return cand
    low = n.lower()
    for m in models:
        if m["name"].lower() == low or m["folder"].lower() == sanitize(n).lower():
            return m["folder"]
    return None


def canonical(name):
    n = name.strip()
    if n.lower().endswith(":latest"):
        n = n[: -len(":latest")]
    return EXTRA_ALIASES.get(n.lower(), n)


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def child_env():
    env = os.environ.copy()
    # On an Intel/AMD PC the engine's default CPU mode expands a 3B model to ~12 GB of RAM;
    # 8-bit weights need about a quarter of that. Override with CYBERTECH_PREC=f16|f32.
    env.setdefault("OJAS_CPU_PREC", os.environ.get("CYBERTECH_PREC", "q8"))
    return env


# ----------------------------------------------------------------------------- pulling

PERCENT = re.compile(rb"(\d{1,3}(?:\.\d)?)%")


def pull_model(name, quant=None, on_progress=None):
    """Run `cybertech-engine pull`. Returns (ok, message). on_progress(status, percent)."""
    eng = engine_path()
    if not eng:
        return False, "cybertech-engine was not found next to cybertech"
    name = canonical(name)
    cmd = [eng, "pull", name] + ([quant] if quant else [])
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                             env=child_env(), stdin=subprocess.DEVNULL)
    except OSError as e:
        return False, f"could not start the engine: {e}"
    tail, last, buf = b"", -1.0, b""
    while True:
        chunk = p.stdout.read(256)
        if not chunk:
            break
        tail = (tail + chunk)[-2000:]
        buf = (buf + chunk)[-300:]
        m = PERCENT.findall(buf)
        if m and on_progress:
            pct = float(m[-1])
            if pct != last:
                last = pct
                on_progress("downloading", pct)
    rc = p.wait()
    if rc != 0:
        text = tail.decode("utf-8", "replace").replace("\r", "\n").strip().splitlines()
        return False, (text[-1] if text else f"pull failed (exit {rc})")
    folder = folder_for(name, quant)
    if not gguf_files(os.path.join(models_dir(), folder)):
        return False, "download finished but no model file was found"
    reg = load_registry()
    reg[folder] = name if not quant or quant.lower() == DEFAULT_QUANT.lower() else f"{name}:{quant}"
    save_registry(reg)
    return True, "success"


def remove_model(folder):
    d = os.path.join(models_dir(), folder)
    if not folder or ".." in folder or not os.path.isdir(d):
        return False
    shutil.rmtree(d, ignore_errors=True)
    reg = load_registry()
    if reg.pop(folder, None) is not None:
        save_registry(reg)
    return True


# ----------------------------------------------------------------------------- backend

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def log_path():
    return os.path.join(tempfile.gettempdir(), "cybertech-engine.log")


def log_tail(n=12):
    try:
        with open(log_path(), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 6000))
            lines = f.read().decode("utf-8", "replace").replace("\r", "\n").strip().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


class Backend:
    """Keeps at most one `cybertech-engine serve` process running, for the model last asked for."""

    def __init__(self):
        self.proc = None
        self.folder = None
        self.port = None
        self.last_used = time.time()
        self.busy = 0
        self.lock = threading.RLock()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=10)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            self.proc, self.folder, self.port = None, None, None

    def ensure(self, folder):
        with self.lock:
            self.last_used = time.time()
            if self.alive() and self.folder == folder:
                return self.port
            self.stop()
            eng = engine_path()
            if not eng:
                raise RuntimeError("cybertech-engine was not found next to cybertech")
            port = free_port()
            cmd = [eng, "serve", folder, "--host", "127.0.0.1", "--port", str(port)]
            logf = open(log_path(), "ab")
            logf.write(f"\n=== {iso_now()} {' '.join(cmd)}\n".encode())
            logf.flush()
            flags = 0x08000000 if WIN else 0  # CREATE_NO_WINDOW
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=child_env(),
                                         stdin=subprocess.DEVNULL, creationflags=flags)
            self.folder, self.port = folder, port
            deadline = time.time() + LOAD_TIMEOUT
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    tail = log_tail()
                    self.proc, self.folder, self.port = None, None, None
                    raise RuntimeError("the engine stopped while loading the model:\n" + tail)
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                        if r.status == 200:
                            self.last_used = time.time()
                            return port
                except (urllib.error.URLError, OSError):
                    pass
                time.sleep(0.5)
            self.stop()
            raise RuntimeError("timed out loading the model")

    def reaper(self):
        while True:
            time.sleep(15)
            with self.lock:
                if self.alive() and self.busy == 0 and KEEP_ALIVE > 0 \
                        and time.time() - self.last_used > KEEP_ALIVE:
                    self.stop()


BACKEND = Backend()
GEN_LOCK = threading.Lock()  # the engine answers one request at a time; also stops model swaps mid-reply


def backend_post(port, path, payload_bytes):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=payload_bytes, method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=3600)


def sse_events(resp):
    """Yield parsed JSON objects from an OpenAI-style server-sent-event stream."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


# ----------------------------------------------------------------------------- HTTP

def map_options(opts, payload):
    opts = opts or {}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                     ("seed", "seed"), ("repeat_penalty", "repeat_penalty"),
                     ("repeat_last_n", "repeat_last_n"), ("stop", "stop")):
        if opts.get(src) is not None:
            payload[dst] = opts[src]
    n = opts.get("num_predict")
    if isinstance(n, int) and n > 0:
        payload["max_tokens"] = n


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # streams are ended by closing the connection
    server_version = "Cybertech/" + VERSION

    def log_message(self, fmt, *args):
        if os.environ.get("CYBERTECH_VERBOSE"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, code, msg):
        self._send(code, {"error": msg})

    def _stream_head(self, ctype="application/x-ndjson"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self._cors()
        self.end_headers()

    def _line(self, obj):
        self.wfile.write(json.dumps(obj).encode() + b"\n")
        self.wfile.flush()

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if not raw.strip():
            return {}, raw
        try:
            return json.loads(raw), raw
        except ValueError:
            return None, raw

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # -- routes
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/":
            self._send(200, b"Cybertech Engine is running", "text/plain; charset=utf-8")
        elif path == "/api/version":
            self._send(200, {"version": "0.6.0", "cybertech": VERSION})
        elif path == "/api/tags":
            self._send(200, {"models": [self._tag(m) for m in list_models()]})
        elif path == "/api/ps":
            ms = []
            if BACKEND.alive():
                for m in list_models():
                    if m["folder"] == BACKEND.folder:
                        t = self._tag(m)
                        t["expires_at"] = iso_now()
                        ms.append(t)
            self._send(200, {"models": ms})
        elif path in ("/v1/models", "/models"):
            self._send(200, {"object": "list", "data": [
                {"id": m["name"], "object": "model", "created": int(m["modified"].timestamp()),
                 "owned_by": "cybertech"} for m in list_models()]})
        elif path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._error(404, f"no route for GET {path}")

    def do_DELETE(self):
        path = self.path.split("?")[0].rstrip("/")
        if path != "/api/delete":
            return self._error(404, f"no route for DELETE {path}")
        self._delete()

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        body, raw = self._json_body()
        if body is None:
            return self._error(400, "invalid JSON")
        try:
            if path == "/api/chat":
                self.api_chat(body, generate=False)
            elif path == "/api/generate":
                self.api_chat(body, generate=True)
            elif path == "/api/pull":
                self.api_pull(body)
            elif path == "/api/delete":
                self._delete(body)
            elif path == "/api/show":
                self.api_show(body)
            elif path in ("/v1/chat/completions", "/v1/completions", "/chat/completions", "/completions"):
                self.openai_proxy(path if path.startswith("/v1") else "/v1" + path, body)
            elif path in ("/api/embed", "/api/embeddings", "/api/create", "/api/copy", "/api/push"):
                self._error(501, "not supported by The Cybertech Engine")
            else:
                self._error(404, f"no route for POST {path}")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        except Exception as e:  # keep the service alive whatever a request does
            try:
                self._error(500, str(e))
            except Exception:
                pass

    # -- Ollama API
    @staticmethod
    def _tag(m):
        return {"name": m["name"], "model": m["name"], "modified_at": m["modified"].isoformat(),
                "size": m["size"], "digest": "",
                "details": {"format": "gguf", "family": "", "families": None,
                            "parameter_size": "", "quantization_level": ""}}

    def api_show(self, body):
        folder = resolve(body.get("model") or body.get("name"))
        if not folder:
            return self._error(404, "model not found")
        self._send(200, {"modelfile": "", "parameters": "", "template": "", "license": "",
                         "details": {"format": "gguf"}, "model_info": {}})

    def _delete(self, body=None):
        if body is None:
            body, _ = self._json_body()
        folder = resolve((body or {}).get("model") or (body or {}).get("name"))
        if not folder:
            return self._error(404, "model not found")
        if BACKEND.folder == folder:
            BACKEND.stop()
        remove_model(folder)
        self._send(200, b"", "text/plain")

    def api_pull(self, body):
        name = body.get("model") or body.get("name")
        if not name:
            return self._error(400, "missing model name")
        stream = body.get("stream", True)
        if stream:
            self._stream_head()
            self._line({"status": "pulling manifest"})

            def progress(status, pct):
                try:
                    self._line({"status": status, "digest": "", "total": 100, "completed": int(pct)})
                except OSError:
                    pass
            ok, msg = pull_model(name, on_progress=progress)
            self._line({"status": "success"} if ok else {"error": msg})
        else:
            ok, msg = pull_model(name)
            self._send(200, {"status": "success"}) if ok else self._error(500, msg)

    def api_chat(self, body, generate):
        name = body.get("model")
        folder = resolve(name)
        if not folder:
            return self._error(404, f"model '{name}' not found, try pulling it first")
        stream = body.get("stream", True)
        if generate:
            if body.get("raw"):
                path, payload = "/v1/completions", {"prompt": body.get("prompt", "")}
            else:
                msgs = []
                if body.get("system"):
                    msgs.append({"role": "system", "content": body["system"]})
                msgs.append({"role": "user", "content": body.get("prompt", "")})
                path, payload = "/v1/chat/completions", {"messages": msgs}
        else:
            msgs = [{"role": m.get("role", "user"), "content": m.get("content") or ""}
                    for m in body.get("messages", [])]
            if body.get("system"):
                msgs.insert(0, {"role": "system", "content": body["system"]})
            path, payload = "/v1/chat/completions", {"messages": msgs}
        payload["stream"] = True
        map_options(body.get("options"), payload)
        fmt = body.get("format")
        if fmt == "json":
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(fmt, dict):
            payload["json_schema"] = fmt

        def piece_of(ev):
            ch = (ev.get("choices") or [{}])[0]
            if path.endswith("chat/completions"):
                return (ch.get("delta") or {}).get("content") or "", ch.get("finish_reason")
            return ch.get("text") or "", ch.get("finish_reason")

        def wrap(text, done=False, extra=None):
            if generate:
                o = {"model": name, "created_at": iso_now(), "response": text, "done": done}
            else:
                o = {"model": name, "created_at": iso_now(),
                     "message": {"role": "assistant", "content": text}, "done": done}
            if extra:
                o.update(extra)
            return o

        with GEN_LOCK:
            BACKEND.busy += 1
            t0 = time.time()
            try:
                try:
                    port = BACKEND.ensure(folder)
                    resp = backend_post(port, path, json.dumps(payload).encode())
                except urllib.error.HTTPError as e:
                    return self._error(500, e.read().decode("utf-8", "replace"))
                except Exception as e:
                    return self._error(500, str(e))
                full, count, reason, usage = [], 0, "stop", {}
                if stream:
                    self._stream_head()
                with resp:
                    for ev in sse_events(resp):
                        text, fin = piece_of(ev)
                        if ev.get("usage"):
                            usage = ev["usage"]
                        if fin:
                            reason = "length" if fin == "length" else "stop"
                        if text:
                            count += 1
                            if stream:
                                self._line(wrap(text))
                            else:
                                full.append(text)
                dur = int((time.time() - t0) * 1e9)
                extra = {"done_reason": reason, "total_duration": dur, "load_duration": 0,
                         "prompt_eval_count": usage.get("prompt_tokens", 0),
                         "eval_count": usage.get("completion_tokens", count), "eval_duration": dur}
                if stream:
                    self._line(wrap("", True, extra))
                else:
                    self._send(200, wrap("".join(full), True, extra))
            finally:
                BACKEND.busy -= 1
                BACKEND.last_used = time.time()

    # -- OpenAI API (passed straight through)
    def openai_proxy(self, path, body):
        name = body.get("model")
        folder = resolve(name)
        if not folder:
            models = list_models()
            if len(models) == 1:        # apps that send a placeholder model name still work
                folder = models[0]["folder"]
            else:
                return self._error(404, f"model '{name}' not found, try pulling it first")
        with GEN_LOCK:
            BACKEND.busy += 1
            try:
                try:
                    port = BACKEND.ensure(folder)
                    resp = backend_post(port, path, json.dumps(body).encode())
                except urllib.error.HTTPError as e:
                    return self._send(e.code, e.read() or b"{}")
                except Exception as e:
                    return self._error(500, str(e))
                with resp:
                    ctype = resp.headers.get("Content-Type", "application/json")
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Cache-Control", "no-cache")
                        self._cors()
                        self.end_headers()
                        while True:
                            chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    else:
                        self._send(200, resp.read(), ctype)
            finally:
                BACKEND.busy -= 1
                BACKEND.last_used = time.time()


# ----------------------------------------------------------------------------- commands

def cmd_serve():
    eng = engine_path()
    print(f"The Cybertech Engine  (cybertech {VERSION})")
    print(f"  engine : {eng or 'NOT FOUND - put cybertech-engine.exe next to this program'}")
    print(f"  models : {models_dir()}")
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
    except OSError as e:
        print(f"\nCannot listen on {HOST}:{PORT}: {e}")
        print("Another program (maybe Ollama or another Cybertech window) is using that port.")
        print("Close it, or choose another port:  set CYBERTECH_PORT=11500")
        return 1
    server.daemon_threads = True
    threading.Thread(target=BACKEND.reaper, daemon=True).start()
    print(f"\nRunning on http://{HOST}:{PORT}   (Ollama API: /api/...   OpenAI API: /v1/...)")
    print("Pull a model in another window:  cybertech pull qwen2.5:3b")
    print("Leave this window open. Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        BACKEND.stop()
    return 0


def cmd_pull(args):
    if not args:
        print("usage: cybertech pull <name> [quant]     e.g. cybertech pull qwen2.5:3b")
        return 2
    eng = engine_path()
    if not eng:
        print("cybertech-engine was not found next to cybertech")
        return 1
    name, quant = canonical(args[0]), (args[1] if len(args) > 1 else None)
    rc = subprocess.call([eng, "pull", name] + ([quant] if quant else []), env=child_env())
    if rc == 0 and gguf_files(os.path.join(models_dir(), folder_for(name, quant))):
        reg = load_registry()
        reg[folder_for(name, quant)] = name if not quant or quant.lower() == DEFAULT_QUANT.lower() \
            else f"{name}:{quant}"
        save_registry(reg)
        print(f"\nReady. Chat with it:  cybertech run {name}")
    return rc


def cmd_list():
    ms = list_models()
    if not ms:
        print("No models yet. Try:  cybertech pull qwen2.5:3b")
        print(f"(models are stored in {models_dir()})")
        return 0
    print(f"{'NAME':<40} {'SIZE':>9}  MODIFIED")
    for m in ms:
        print(f"{m['name']:<40} {m['size'] / 1e9:>6.1f} GB  {m['modified'].astimezone():%Y-%m-%d %H:%M}")
    return 0


def cmd_rm(args):
    if not args:
        print("usage: cybertech rm <name>")
        return 2
    folder = resolve(args[0])
    if not folder or not remove_model(folder):
        print(f"no downloaded model named {args[0]!r} (see: cybertech list)")
        return 1
    print(f"deleted {args[0]}")
    return 0


def cmd_run(args):
    if not args:
        print("usage: cybertech run <name>")
        return 2
    eng = engine_path()
    if not eng:
        print("cybertech-engine was not found next to cybertech")
        return 1
    folder = resolve(args[0])
    if not folder:
        print(f"{args[0]} is not downloaded yet, pulling it first...")
        if cmd_pull([args[0]]) != 0:
            return 1
        folder = resolve(args[0])
        if not folder:
            return 1
    return subprocess.call([eng, "chat", folder] + args[1:], env=child_env())


# ----------------------------------------------------------------------------- window

SUGGESTED = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5-coder:7b", "llama3.2:3b", "llama3.1:8b"]
_reaper_started = False


class Service:
    """The HTTP service, started and stopped from the window."""

    def __init__(self):
        self.server = None

    def running(self):
        return self.server is not None

    def start(self):
        global _reaper_started
        if self.server:
            return None
        try:
            srv = ThreadingHTTPServer((HOST, PORT), Handler)
        except OSError as e:
            return str(e)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        if not _reaper_started:
            _reaper_started = True
            threading.Thread(target=BACKEND.reaper, daemon=True).start()
        self.server = srv
        return None

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        BACKEND.stop()


def build_gui(root):
    """Builds the Cybertech Engine window on `root`. Returns the object holding its widgets."""
    import queue
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    class App:
        pass

    app = App()
    app.root, app.service, app.q = root, Service(), queue.Queue()
    app.history, app.busy_pull, app.busy_chat = [], False, False
    root.title("Cybertech Engine")
    root.geometry("780x580")
    root.minsize(660, 480)

    # -- top bar: service status
    top = ttk.Frame(root, padding=(14, 12, 14, 6))
    top.pack(fill="x")
    app.dot = tk.Label(top, text="●", font=("Segoe UI", 16))
    app.dot.pack(side="left")
    box = ttk.Frame(top)
    box.pack(side="left", padx=8)
    app.status = tk.StringVar()
    ttk.Label(box, textvariable=app.status, font=("Segoe UI", 11, "bold")).pack(anchor="w")
    app.sub = tk.StringVar()
    ttk.Label(box, textvariable=app.sub, foreground="#666").pack(anchor="w")
    app.toggle = ttk.Button(top, width=14)
    app.toggle.pack(side="right")
    ttk.Button(top, text="Copy address", width=14, command=lambda: (root.clipboard_clear(),
               root.clipboard_append(f"http://{HOST}:{PORT}"))).pack(side="right", padx=8)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=14, pady=(4, 8))

    # -- models tab
    mt = ttk.Frame(nb, padding=12)
    nb.add(mt, text="  Models  ")
    cols = ("name", "size")
    app.tree = ttk.Treeview(mt, columns=cols, show="headings", height=8, selectmode="browse")
    app.tree.heading("name", text="Downloaded models")
    app.tree.heading("size", text="Size")
    app.tree.column("name", width=420)
    app.tree.column("size", width=100, anchor="e")
    app.tree.pack(fill="both", expand=True)
    bar = ttk.Frame(mt)
    bar.pack(fill="x", pady=(6, 14))
    app.del_btn = ttk.Button(bar, text="Delete selected")
    app.del_btn.pack(side="left")
    ttk.Label(mt, text="Download a model", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    pr = ttk.Frame(mt)
    pr.pack(fill="x", pady=4)
    app.pull_name = ttk.Combobox(pr, values=SUGGESTED, width=44)
    app.pull_name.set(SUGGESTED[0])
    app.pull_name.pack(side="left")
    app.pull_btn = ttk.Button(pr, text="Download")
    app.pull_btn.pack(side="left", padx=8)
    app.prog = ttk.Progressbar(mt, maximum=100)
    app.prog.pack(fill="x", pady=(4, 2))
    app.pull_msg = tk.StringVar(value="Type a name above (or a Hugging Face owner/repo) and press Download.")
    ttk.Label(mt, textvariable=app.pull_msg, foreground="#666", wraplength=700).pack(anchor="w")

    # -- chat tab
    ct = ttk.Frame(nb, padding=12)
    nb.add(ct, text="  Chat  ")
    cr = ttk.Frame(ct)
    cr.pack(fill="x")
    ttk.Label(cr, text="Model").pack(side="left")
    app.chat_model = ttk.Combobox(cr, width=40, state="readonly")
    app.chat_model.pack(side="left", padx=8)
    ttk.Button(cr, text="New chat", command=lambda: new_chat()).pack(side="right")
    app.log = ScrolledText(ct, wrap="word", state="disabled", height=14, font=("Segoe UI", 10))
    app.log.pack(fill="both", expand=True, pady=8)
    app.log.tag_config("you", foreground="#1a56c4", font=("Segoe UI", 10, "bold"))
    app.log.tag_config("bot", foreground="#222")
    app.log.tag_config("note", foreground="#888")
    ir = ttk.Frame(ct)
    ir.pack(fill="x")
    app.entry = ttk.Entry(ir)
    app.entry.pack(side="left", fill="x", expand=True)
    app.send_btn = ttk.Button(ir, text="Send")
    app.send_btn.pack(side="left", padx=(8, 0))

    # -- behaviour
    def say(text, tag="bot"):
        app.log.config(state="normal")
        app.log.insert("end", text, tag)
        app.log.see("end")
        app.log.config(state="disabled")

    def refresh_models():
        for i in app.tree.get_children():
            app.tree.delete(i)
        ms = list_models()
        for m in ms:
            app.tree.insert("", "end", iid=m["folder"], values=(m["name"], f"{m['size'] / 1e9:.1f} GB"))
        names = [m["name"] for m in ms]
        app.chat_model["values"] = names
        if names and app.chat_model.get() not in names:
            app.chat_model.set(names[0])
        if not names:
            app.chat_model.set("")
    app.refresh_models = refresh_models

    def refresh_status():
        if app.service.running():
            app.dot.config(fg="#1a9a3a")
            app.status.set("Running")
            app.sub.set(f"http://{HOST}:{PORT}  •  apps connect here")
            app.toggle.config(text="Stop service")
        else:
            app.dot.config(fg="#b3b3b3")
            app.status.set("Stopped")
            app.sub.set("Press Start so apps can use your models.")
            app.toggle.config(text="Start service")
        if not engine_path():
            app.dot.config(fg="#c0392b")
            app.sub.set("cybertech-engine.exe was not found next to this program.")
    app.refresh_status = refresh_status

    def toggle():
        if app.service.running():
            app.service.stop()
        else:
            err = app.service.start()
            if err:
                messagebox.showerror("Cybertech Engine",
                                     f"Cannot listen on port {PORT}.\n\n{err}\n\nAnother program (Ollama?) "
                                     f"may be using it. Close it, or set CYBERTECH_PORT to another number.")
        refresh_status()
    app.toggle.config(command=toggle)
    app.toggle_service = toggle

    def do_pull():
        name = app.pull_name.get().strip()
        if not name or app.busy_pull:
            return
        if not engine_path():
            messagebox.showerror("Cybertech Engine", "cybertech-engine.exe was not found next to this program.")
            return
        app.busy_pull = True
        app.pull_btn.config(state="disabled")
        app.prog.config(value=0)
        app.pull_msg.set(f"Downloading {name} ... (large files take a while; closing the window pauses it)")

        def work():
            ok, msg = pull_model(name, on_progress=lambda s, p: app.q.put(("pull", p)))
            app.q.put(("pull_done", ok, msg, name))
        threading.Thread(target=work, daemon=True).start()
    app.pull_btn.config(command=do_pull)
    app.do_pull = do_pull

    def do_delete():
        sel = app.tree.selection()
        if not sel:
            return
        name = app.tree.item(sel[0], "values")[0]
        if messagebox.askyesno("Delete model", f"Delete {name} from this computer?"):
            if BACKEND.folder == sel[0]:
                BACKEND.stop()
            remove_model(sel[0])
            refresh_models()
    app.del_btn.config(command=do_delete)

    def new_chat():
        app.history = []
        app.log.config(state="normal")
        app.log.delete("1.0", "end")
        app.log.config(state="disabled")
        say("New chat. The first reply loads the model and can take a minute.\n\n", "note")
    app.new_chat = new_chat

    def send(_event=None):
        text = app.entry.get().strip()
        model = app.chat_model.get()
        if not text or app.busy_chat:
            return
        if not model:
            messagebox.showinfo("Cybertech Engine", "Download a model first (Models tab).")
            return
        if not app.service.running():
            messagebox.showinfo("Cybertech Engine", "Start the service first (top right).")
            return
        app.entry.delete(0, "end")
        app.busy_chat = True
        app.send_btn.config(state="disabled")
        app.history.append({"role": "user", "content": text})
        say("You: ", "you")
        say(text + "\n", "bot")
        say("Cybertech: ", "you")
        msgs = list(app.history)

        def work():
            reply = []
            try:
                req = urllib.request.Request(
                    f"http://{HOST}:{PORT}/api/chat", method="POST",
                    data=json.dumps({"model": model, "messages": msgs, "stream": True}).encode())
                with urllib.request.urlopen(req, timeout=3600) as r:
                    for line in r:
                        if not line.strip():
                            continue
                        o = json.loads(line)
                        if o.get("error"):
                            raise RuntimeError(o["error"])
                        t = (o.get("message") or {}).get("content") or ""
                        if t:
                            reply.append(t)
                            app.q.put(("tok", t))
                app.q.put(("chat_done", "".join(reply)))
            except urllib.error.HTTPError as e:
                app.q.put(("chat_err", e.read().decode("utf-8", "replace")))
            except Exception as e:
                app.q.put(("chat_err", str(e)))
        threading.Thread(target=work, daemon=True).start()
    app.send_btn.config(command=send)
    app.entry.bind("<Return>", send)
    app.send = send

    def pump():
        try:
            while True:
                item = app.q.get_nowait()
                k = item[0]
                if k == "pull":
                    app.prog.config(value=item[1])
                    app.pull_msg.set(f"Downloading ... {item[1]:.0f}%")
                elif k == "pull_done":
                    app.busy_pull = False
                    app.pull_btn.config(state="normal")
                    if item[1]:
                        app.prog.config(value=100)
                        app.pull_msg.set(f"{item[3]} is ready. Use it in the Chat tab or from any app.")
                        refresh_models()
                    else:
                        app.pull_msg.set("Download failed: " + item[2] + "   (press Download again to resume)")
                elif k == "tok":
                    say(item[1])
                elif k == "chat_done":
                    say("\n\n")
                    app.history.append({"role": "assistant", "content": item[1]})
                    app.busy_chat = False
                    app.send_btn.config(state="normal")
                elif k == "chat_err":
                    say("\n[error] " + item[1] + "\n\n", "note")
                    if app.history and app.history[-1]["role"] == "user":
                        app.history.pop()
                    app.busy_chat = False
                    app.send_btn.config(state="normal")
        except queue.Empty:
            pass
        root.after(100, pump)
    pump()

    def on_close():
        app.service.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_models()
    new_chat()
    err = app.service.start()
    if err:
        messagebox.showwarning("Cybertech Engine",
                               f"Could not start the service on port {PORT}.\n\n{err}\n\nAnother program "
                               f"(Ollama?) may be using it. Close it, or set CYBERTECH_PORT to another number.")
    refresh_status()
    return app


def cmd_gui():
    try:
        import tkinter as tk
    except ImportError:
        return cmd_serve()
    root = tk.Tk()
    build_gui(root)
    root.mainloop()
    return 0


def usage():
    print(f"""The Cybertech Engine - cybertech {VERSION}

  cybertech pull <name> [quant]   download a model   (qwen2.5:3b  qwen2.5:7b  qwen2.5-coder:7b
                                  llama3.2:3b  llama3.1:8b  or any Hugging Face GGUF: owner/repo)
  cybertech list                  show downloaded models
  cybertech run <name>            chat in this window
  cybertech rm <name>             delete a model
  cybertech                       open the window (also starts the service)
  cybertech serve                 run the service without a window, on http://{HOST}:{PORT}
  cybertech version
""")


def main(argv):
    if sys.stdout is None:  # windowed exe: no console to print to
        sys.stdout = sys.stderr = open(os.devnull, "w")
    if not argv:
        return cmd_gui()  # double-clicked: open the window (it also starts the service)
    if argv[0].lower() == "gui":
        return cmd_gui()
    c, rest = argv[0].lower(), argv[1:]
    if c in ("serve", "start"):
        return cmd_serve()
    if c in ("pull", "download"):
        return cmd_pull(rest)
    if c in ("list", "ls"):
        return cmd_list()
    if c in ("rm", "remove", "delete"):
        return cmd_rm(rest)
    if c in ("run", "chat"):
        return cmd_run(rest)
    if c in ("version", "--version", "-v"):
        print(f"cybertech {VERSION}")
        return 0
    usage()
    return 0 if c in ("help", "-h", "--help") else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
