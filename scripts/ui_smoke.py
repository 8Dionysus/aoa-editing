#!/usr/bin/env python3
"""Drive the real workbench in headless Chromium through the DevTools protocol."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aoa_editing.config import LEGACY_ROOT_VARIABLES, Settings


class SmokeError(RuntimeError):
    pass


class CDP:
    def __init__(self, url: str):
        parsed = urlparse(url)
        if parsed.hostname is None or parsed.port is None:
            raise SmokeError(f"invalid debugger URL: {url}")
        self.socket = socket.create_connection((parsed.hostname, parsed.port), timeout=10)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {parsed.path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.socket.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise SmokeError(f"websocket upgrade failed: {response[:500]!r}")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        )
        if expected not in response:
            raise SmokeError("websocket accept key mismatch")
        self.next_id = 1

    def close(self) -> None:
        self.socket.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        call_id = self.next_id
        self.next_id += 1
        self._send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._receive())
            if message.get("id") == call_id:
                if "error" in message:
                    raise SmokeError(f"CDP {method} failed: {message['error']}")
                return dict(message.get("result", {}))

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise SmokeError(str(remote.get("description", "browser evaluation failed")))
        return remote.get("value")

    def _send(self, payload: str) -> None:
        encoded = payload.encode("utf-8")
        mask = secrets.token_bytes(4)
        length = len(encoded)
        if length < 126:
            header = bytes((0x81, 0x80 | length))
        elif length < 65536:
            header = bytes((0x81, 0x80 | 126)) + struct.pack("!H", length)
        else:
            header = bytes((0x81, 0x80 | 127)) + struct.pack("!Q", length)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(encoded))
        self.socket.sendall(header + mask + masked)

    def _receive(self) -> str:
        while True:
            first, second = _read_exact(self.socket, 2)
            opcode = first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", _read_exact(self.socket, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", _read_exact(self.socket, 8))[0]
            mask = _read_exact(self.socket, 4) if second & 0x80 else b""
            payload = _read_exact(self.socket, length)
            if mask:
                payload = bytes(
                    value ^ mask[index % 4] for index, value in enumerate(payload)
                )
            if opcode == 0x8:
                raise SmokeError("browser closed the debugger socket")
            if opcode == 0x9:
                self.socket.sendall(bytes((0x8A, len(payload))) + payload)
                continue
            if opcode == 0x1:
                return payload.decode("utf-8")


def _read_exact(connection: socket.socket, size: int) -> bytes:
    payload = b""
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise SmokeError("unexpected debugger socket EOF")
        payload += chunk
    return payload


def _free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def _wait(predicate: Any, description: str, timeout: float = 20) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as error:  # service/browser may still be starting
            last_error = error
        time.sleep(0.1)
    detail = f": {last_error}" if last_error else ""
    raise SmokeError(f"timed out waiting for {description}{detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(source_home=repo)
    python = Path(sys.executable)
    chromium = shutil.which("chromium-browser") or shutil.which("chromium")
    if chromium is None:
        raise SmokeError("Chromium is required for the mandatory UI smoke")
    settings.tmp_root.mkdir(parents=True, exist_ok=True)
    output = (
        args.output
        or Path(tempfile.mkdtemp(prefix="ui-smoke-", dir=settings.tmp_root))
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="aoa-ui-smoke-", dir=output))
    server_port = _free_port()
    debugger_port = _free_port()
    environment = os.environ.copy()
    for variable in LEGACY_ROOT_VARIABLES:
        environment.pop(variable, None)
    environment.update(
        {
            "AOA_EDITING_HOME": str(workspace),
            "AOA_EDITING_BIND": "127.0.0.1",
            "AOA_EDITING_PORT": str(server_port),
        }
    )
    server = subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "aoa_editing.api.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--log-level",
            "warning",
        ],
        cwd=repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    browser: subprocess.Popen[str] | None = None
    cdp: CDP | None = None
    started = time.monotonic()
    try:
        _wait(lambda: _json(f"http://127.0.0.1:{server_port}/api/health"), "API")
        profile = workspace / "chromium-profile"
        browser = subprocess.Popen(
            [
                chromium,
                "--headless=new",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={debugger_port}",
                f"--user-data-dir={profile}",
                f"http://127.0.0.1:{server_port}/",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def debugger_url() -> str | None:
            pages = _json(f"http://127.0.0.1:{debugger_port}/json/list")
            page = next(
                (
                    item
                    for item in pages
                    if item.get("type") == "page"
                    and str(item.get("url", "")).startswith(
                        f"http://127.0.0.1:{server_port}/"
                    )
                ),
                None,
            )
            return str(page["webSocketDebuggerUrl"]) if page else None

        cdp = CDP(_wait(debugger_url, "Chromium debugger"))
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        _wait(
            lambda: cdp.evaluate(
                "document.readyState === 'complete' "
                "&& document.querySelector('#new-project-button') !== null"
            ),
            "workbench document",
        )
        submitted = cdp.evaluate(
            """
            (() => {
              document.querySelector('#new-project-button').click();
              const form = document.querySelector('#project-form');
              form.elements.name.value = 'UI Smoke Fixture';
              form.elements.scenario.value = 'still.motion';
              form.elements.intent.value = 'Browser-created reversible edit';
              form.elements.duration.value = '1';
              form.elements.mood.value = 'restrained';
              form.requestSubmit();
              return true;
            })()
            """
        )
        if submitted is not True:
            raise SmokeError("project form was not submitted")
        try:
            _wait(
                lambda: cdp.evaluate(
                    "document.querySelector('#project-title')?.textContent "
                    "=== 'UI Smoke Fixture'"
                ),
                "UI-created project selection",
            )
        except SmokeError as error:
            browser_state = cdp.evaluate(
                """
                (() => ({
                  title: document.querySelector('#project-title')?.textContent,
                  toast: document.querySelector('#toast')?.textContent,
                  projectList: document.querySelector('#project-list')?.textContent,
                  dialogOpen: document.querySelector('#project-dialog')?.open,
                  busy: !document.querySelector('#busy')?.classList.contains('hidden')
                }))()
                """
            )
            raise SmokeError(f"{error}; browser state={browser_state}") from error
        cdp.evaluate(
            """
            (() => {
              const form = document.querySelector('#brief-revision-form');
              form.elements.intent.value = 'Browser-revised reversible edit';
              form.elements.rationale.value = 'Browser smoke revision';
              form.requestSubmit();
              return true;
            })()
            """
        )
        _wait(
            lambda: cdp.evaluate(
                "document.querySelector('#tab-edit')?.textContent.includes('rev 2')"
            ),
            "brief revision through UI",
        )
        cdp.evaluate(
            """
            (() => {
              const form = document.querySelector('#style-confirm-form');
              form.elements.preference.value = 'motion';
              form.elements.value.value = 'restrained';
              form.requestSubmit();
              return true;
            })()
            """
        )
        _wait(
            lambda: len(_json(f"http://127.0.0.1:{server_port}/api/style-profiles")) == 1,
            "explicit style confirmation",
        )
        tab_ok = cdp.evaluate(
            """
            (() => {
              document.querySelector('[data-tab="evidence"]').click();
              return document.querySelector('#tab-evidence').classList.contains('active')
                && document.querySelector('#tab-evidence').textContent.includes('Evidence ledger');
            })()
            """
        )
        if tab_ok is not True:
            raise SmokeError("tab interaction did not update the workbench")
        cdp.evaluate("window.scrollTo(0, 0); true")
        screenshot = cdp.call("Page.captureScreenshot", {"format": "png"})["data"]
        screenshot_path = output / "ui-smoke.png"
        screenshot_path.write_bytes(base64.b64decode(screenshot))
        projects = _json(f"http://127.0.0.1:{server_port}/api/projects")
        if len(projects) != 1 or projects[0]["name"] != "UI Smoke Fixture":
            raise SmokeError("browser action did not persist exactly one project")
        receipt = {
            "schema": "aoa_editing_ui_smoke_v1",
            "overall": "pass",
            "browser": subprocess.run(
                [chromium, "--version"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "checks": {
                "real_browser": True,
                "javascript_executed": True,
                "project_created_through_form": True,
                "project_selected": True,
                "brief_revision": True,
                "explicit_style_confirmation": True,
                "tab_interaction": True,
                "api_persistence": True,
            },
            "screenshot": str(screenshot_path),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        receipt_path = output / "ui-smoke.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))
    finally:
        if cdp is not None:
            cdp.close()
        if browser is not None:
            browser.terminate()
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.kill()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"UI smoke failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
