"""
Headless SOC REST API Server for Frontend & UI Designers.

Provides high-throughput JSON endpoints with full CORS support so frontend
designers can develop and connect arbitrary web frameworks (React, Vue, Next.js,
Vite, Svelte, or static HTML/CSS) to the GNN explainability engine.

Usage:
    venv/bin/python scripts/soc_server.py [--port 8080] [--host 127.0.0.1]
"""

import argparse
import http.server
import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT_DIR / "output" / "reports" / "soc_dashboard_data.json"
WEB_DIR = ROOT_DIR / "web"


class SOCAPIHandler(http.server.BaseHTTPRequestHandler):
    """
    Headless REST API Request Handler with full CORS enablement.
    """

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # 1. API Endpoints
        if path.startswith("/api/"):
            self.handle_api(path, params)
            return

        # 2. If web/ directory exists and contains requested file, serve it
        if WEB_DIR.exists():
            file_path = WEB_DIR / path.lstrip("/")
            if file_path.is_file():
                self.serve_static_file(file_path)
                return
            index_path = WEB_DIR / "index.html"
            if (path == "/" or path == "") and index_path.is_file():
                self.serve_static_file(index_path)
                return

        # 3. Default root response: API discovery directory
        self.send_json_response(
            {
                "service": "CYBER-GNN Forensic SOC REST API",
                "version": "2.4",
                "status": "ready",
                "description": "Backend explainability and anomaly scoring API for frontend SOC dashboards.",
                "dataset": "CMU CERT Insider Threat Testbed (r4.2)",
                "endpoints": {
                    "GET /api/overview": "Global KPIs and system metadata",
                    "GET /api/incidents": "List of triaged threat alerts (supports ?severity=CRITICAL&limit=10)",
                    "GET /api/incidents/<incident_id>": "Full explainability dossier for a specific incident",
                    "GET /api/users": "List of monitored employees and risk summaries",
                    "GET /api/users/<user_id>": "Detailed 501-day behavioral timeline and profile for an employee",
                    "GET /api/benchmarks": "Cross-paradigm model comparison metrics",
                },
                "documentation": "docs/DESIGNER_INTEGRATION_GUIDE.md",
            }
        )

    def handle_api(self, path: str, params: dict):
        if not DATA_FILE.exists():
            self.send_json_response(
                {"error": "Dataset not compiled. Run 'python scripts/export_soc_data.py' first."},
                status=503,
            )
            return

        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        if path == "/api/overview" or path == "/api/kpi":
            self.send_json_response(
                {"kpi": data.get("kpi", {}), "metadata": data.get("metadata", {})}
            )
            return

        if path == "/api/incidents":
            incidents = data.get("incidents", [])
            if "severity" in params:
                sev = params["severity"][0].upper()
                incidents = [i for i in incidents if i.get("severity") == sev]
            if "user" in params:
                uid = params["user"][0]
                incidents = [i for i in incidents if i.get("user_id") == uid]
            if "scenario" in params:
                try:
                    sc = int(params["scenario"][0])
                    incidents = [
                        i for i in incidents if i.get("scenario") and i["scenario"].get("scenario_id") == sc
                    ]
                except ValueError:
                    pass
            if "limit" in params:
                try:
                    limit = int(params["limit"][0])
                    incidents = incidents[:limit]
                except ValueError:
                    pass
            self.send_json_response({"total": len(incidents), "incidents": incidents})
            return

        if path.startswith("/api/incidents/"):
            inc_id = path.replace("/api/incidents/", "").strip()
            match = next((i for i in data.get("incidents", []) if i["incident_id"] == inc_id), None)
            if match:
                self.send_json_response(match)
            else:
                self.send_json_response({"error": f"Incident '{inc_id}' not found."}, status=404)
            return

        if path == "/api/users":
            users = data.get("users", [])
            summary = [
                {
                    "user_id": u["user_id"],
                    "is_malicious": u["is_malicious"],
                    "malicious_days_count": u.get("malicious_days_count", 0),
                    "profile": u["profile"],
                    "max_temporal_risk": u["max_temporal_risk"],
                    "max_static_risk": u["max_static_risk"],
                }
                for u in users
            ]
            self.send_json_response({"total": len(summary), "users": summary})
            return

        if path.startswith("/api/users/"):
            uid = path.replace("/api/users/", "").strip()
            match = next((u for u in data.get("users", []) if u["user_id"] == uid), None)
            if match:
                self.send_json_response(match)
            else:
                self.send_json_response({"error": f"User '{uid}' not found."}, status=404)
            return

        if path == "/api/benchmarks":
            self.send_json_response({"benchmarks": data.get("benchmarks", [])})
            return

        self.send_json_response({"error": f"Unknown endpoint: {path}"}, status=404)

    def serve_static_file(self, file_path: Path):
        import mimetypes

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, payload: dict, status: int = 200):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="CYBER-GNN Headless SOC REST API Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    # Compile dataset if not already present
    if not DATA_FILE.exists():
        print(f"[api_server] Dataset {DATA_FILE} not found! Compiling via export_soc_data.py...")
        from scripts.export_soc_data import main as export_main

        export_main()

    server_address = (args.host, args.port)
    httpd = http.server.HTTPServer(server_address, SOCAPIHandler)

    print("=" * 74)
    print("      CYBER-GNN // FORENSIC EXPLAINABILITY REST API FOR DESIGNERS")
    print("=" * 74)
    print(f" API Base URL:   http://{args.host}:{args.port}/")
    print(f" Endpoints:      http://{args.host}:{args.port}/api/incidents")
    print(f" Data Source:    {DATA_FILE}")
    print(f" CORS Enabled:   Access-Control-Allow-Origin: *")
    print(" Press Ctrl+C to terminate server.")
    print("=" * 74)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[api_server] Shutting down REST API server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
