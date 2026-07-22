#!/usr/bin/env python3
"""Read-only real-Chromium smoke for an attached Reference Workspace v2."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ui_smoke import CDP, SmokeError, _free_port, _json, _wait

from aoa_editing.config import LEGACY_ROOT_VARIABLES, Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    settings = Settings.from_env(source_home=repo)
    chromium = shutil.which("chromium-browser") or shutil.which("chromium")
    if chromium is None:
        raise SmokeError("Chromium is required for the reference UI smoke")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings.tmp_root.mkdir(parents=True, exist_ok=True)
    run_tmp = Path(tempfile.mkdtemp(prefix="reference-ui-smoke-", dir=settings.tmp_root))
    server_port = _free_port()
    debugger_port = _free_port()
    environment = os.environ.copy()
    for variable in LEGACY_ROOT_VARIABLES:
        environment.pop(variable, None)
    environment.update(
        {
            "AOA_EDITING_HOME": str(settings.editing_home),
            "AOA_EDITING_BIND": "127.0.0.1",
            "AOA_EDITING_PORT": str(server_port),
        }
    )
    server = subprocess.Popen(
        [
            sys.executable,
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
                f"--user-data-dir={run_tmp / 'chromium-profile'}",
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
                "&& document.querySelector('[data-project]') !== null"
            ),
            "project list",
        )
        selected = cdp.evaluate(
            f"""
            (() => {{
              const project = document.querySelector(
                '[data-project="{args.project_id}"]'
              );
              if (!project) return false;
              project.click();
              return true;
            }})()
            """
        )
        if selected is not True:
            raise SmokeError(f"project is absent from workbench: {args.project_id}")
        _wait(
            lambda: cdp.evaluate(
                "document.querySelector('#workspace-status')?.textContent "
                "&& document.querySelector('[data-tab=\"reference\"]') !== null"
            ),
            "selected project",
        )
        cdp.evaluate(
            "document.querySelector('[data-tab=\"reference\"]').click(); true"
        )
        _wait(
            lambda: cdp.evaluate(
                "document.querySelector('#tab-reference.motion-chart') !== null "
                "|| document.querySelector('#tab-reference .motion-chart') !== null"
            ),
            "reference motion chart",
            timeout=30,
        )
        initial = cdp.evaluate(
            """
            (() => ({
              active: document.querySelector('#tab-reference').classList.contains('active'),
              players: document.querySelectorAll('#tab-reference video').length,
              charts: document.querySelectorAll('#tab-reference svg').length,
              plots: document.querySelectorAll('#tab-reference .plot-grid img').length,
              phases: document.querySelectorAll('#tab-reference .phase-pills button').length,
              interpretations: document.querySelectorAll(
                '#tab-reference .interpretation'
              ).length,
              tangent: document.querySelector('#tangent-strength') !== null,
              perItemReview: document.querySelector(
                '[data-action="reference-language-preview"]'
              ) !== null,
              referenceSource: document.querySelector('#reference-player')?.currentSrc,
              candidateSource: document.querySelector('#candidate-player')?.currentSrc
            }))()
            """
        )
        expected = {
            "active": True,
            "players": 6,
            "charts": 2,
            "plots": 8,
            "phases": 5,
            "interpretations": 4,
            "tangent": True,
            "perItemReview": True,
        }
        for key, minimum in expected.items():
            actual = initial.get(key)
            if isinstance(minimum, bool):
                if actual is not minimum:
                    raise SmokeError(f"reference UI check failed: {key}={actual!r}")
            elif not isinstance(actual, int) or actual < minimum:
                raise SmokeError(f"reference UI check failed: {key}={actual!r}")
        if not str(initial.get("referenceSource", "")).endswith(
            "/artifacts/reference_media"
        ):
            raise SmokeError("reference player does not use the allowlisted media route")
        if not str(initial.get("candidateSource", "")).endswith(
            "/artifacts/candidate_media"
        ):
            raise SmokeError("candidate player does not use the allowlisted media route")

        interaction = cdp.evaluate(
            """
            (() => {
              const derivative = document.querySelector('#reference-derivative');
              derivative.value = 'jerk';
              derivative.dispatchEvent(new Event('change', {bubbles:true}));
              const slider = document.querySelector('#reference-frame');
              slider.value = '166';
              slider.dispatchEvent(new Event('input', {bubbles:true}));
              const opacity = document.querySelector('#reference-opacity');
              opacity.value = '.27';
              opacity.dispatchEvent(new Event('input', {bubbles:true}));
              return {
                frame: document.querySelector('#reference-frame')?.value,
                frameLabel: document.querySelector(
                  '#reference-frame-label'
                )?.textContent,
                derivative: document.querySelector(
                  '#reference-derivative'
                )?.value,
                overlayOpacity: document.querySelector(
                  '#reference-player'
                )?.style.opacity,
                residual: document.querySelector(
                  '#reference-flow-residual'
                )?.getAttribute('src')
              };
            })()
            """
        )
        _wait(
            lambda: cdp.evaluate(
                "document.querySelector('#reference-derivative')?.value === 'jerk'"
            ),
            "derivative interaction",
        )
        screenshot = cdp.call(
            "Page.captureScreenshot",
            {"format": "png"},
        )["data"]
        screenshot_path = output / "reference-ui-smoke.png"
        screenshot_path.write_bytes(base64.b64decode(screenshot))
        receipt = {
            "schema": "aoa_editing_reference_ui_smoke_v2",
            "overall": "pass",
            "project_id": args.project_id,
            "browser": subprocess.run(
                [chromium, "--version"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "checks": {
                "real_browser": True,
                "reference_tab_active": initial["active"],
                "allowlisted_reference_route": True,
                "allowlisted_candidate_route": True,
                "synchronized_player_surfaces": initial["players"] >= 3,
                "transform_and_confidence_charts": initial["charts"] >= 2,
                "all_diagnostic_plot_roles": initial["plots"] >= 8,
                "phase_markers": initial["phases"] == 5,
                "variant_interpretations": initial["interpretations"] >= 4,
                "editable_tangent_controls": initial["tangent"],
                "per_item_review_surface": initial["perItemReview"],
                "frame_scrub_interaction": interaction["frame"] == "166",
                "derivative_switch_interaction": interaction["derivative"] == "jerk",
            },
            "interaction": interaction,
            "screenshot": str(screenshot_path),
            "duration_seconds": round(time.monotonic() - started, 3),
            "mutated_project_state": False,
        }
        receipt_path = output / "reference-ui-smoke.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
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
        shutil.rmtree(run_tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Reference UI smoke failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
