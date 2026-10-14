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
DATA_FILE_14D = ROOT_DIR / "output" / "reports" / "soc_dashboard_data.json"
DATA_FILE_60D = ROOT_DIR / "output" / "reports" / "soc_dashboard_data_60d.json"
WEB_DIR = ROOT_DIR / "web"

# In-memory dataset cache
_DATA_CACHE = {}


def get_dataset(model_id: str = "14d") -> dict:
    is_60d = model_id in ["60d", "multiscale"]
    cache_key = "60d" if is_60d else "14d"
    target_file = DATA_FILE_60D if is_60d else DATA_FILE_14D

    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    if not target_file.exists():
        print(f"[api_server] Target dataset {target_file} not found! Compiling via export_soc_data.py...")
        from scripts.export_soc_data import export_model_dataset

        export_model_dataset("60d" if is_60d else "14d")

    if target_file.exists():
        with open(target_file, "r") as f:
            data = json.load(f)
            _DATA_CACHE[cache_key] = data
            return data

    return {}


class SOCAPIHandler(http.server.BaseHTTPRequestHandler):
    """
    Headless REST API Request Handler with full CORS enablement.
    Supports both 14-day production default model and 60-day strategic multi-scale model.
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
                "version": "2.5",
                "status": "ready",
                "description": "Backend explainability and anomaly scoring API for frontend SOC dashboards.",
                "dataset": "CMU CERT Insider Threat Testbed (r4.2)",
                "default_model": "14d",
                "query_parameter": "?model=14d (default) or ?model=60d (optional strategic drift)",
                "endpoints": {
                    "GET /api/models": "List available temporal detection models with benchmark stats",
                    "GET /api/overview": "Global KPIs and system metadata (supports ?model=14d|60d)",
                    "GET /api/incidents": "List of triaged threat alerts (supports ?model=14d|60d&severity=CRITICAL&limit=10)",
                    "GET /api/incidents/<incident_id>": "Full explainability dossier for an incident (supports ?model=14d|60d)",
                    "GET /api/users": "List of monitored employees and risk summaries (supports ?model=14d|60d)",
                    "GET /api/users/<user_id>": "Detailed 501-day timeline for an employee (supports ?model=14d|60d)",
                    "GET /api/benchmarks": "Cross-paradigm model comparison metrics",
                },
                "documentation": "docs/DESIGNER_INTEGRATION_GUIDE.md",
            }
        )

    def handle_api(self, path: str, params: dict):
        # Model selection: 14d (default) or 60d (optional)
        requested_model = params.get("model", ["14d"])[0].lower()

        # Endpoint 1: Model metadata and selection list
        if path == "/api/models":
            self.send_json_response(
                {
                    "default_model": "14d",
                    "active_selection": "60d" if requested_model in ["60d", "multiscale"] else "14d",
                    "models": [
                        {
                            "id": "14d",
                            "name": "Sequential Temporal GRU (14-Day)",
                            "type": "tactical_forensic_triage",
                            "is_default": True,
                            "window_size": 14,
                            "loss": "Binary Focal Loss (gamma=1.5, alpha=0.75)",
                            "pr_auc": 0.6735,
                            "roc_auc": 0.9947,
                            "multiplier_over_chance": 312.0,
                            "precision_at_10": 1.0,
                            "precision_at_20": 1.0,
                            "precision_at_50": 0.94,
                            "precision_at_100": 0.81,
                            "description": "Production default model. Optimal signal-to-noise ratio for fast attack escalation triage and acute incident detection.",
                        },
                        {
                            "id": "60d",
                            "name": "Multi-Scale Hierarchical GRU (3d/14d/60d)",
                            "type": "strategic_insider_surveillance",
                            "is_default": False,
                            "window_sizes": [3, 14, 60],
                            "attention_gating": "Cross-Scale Softmax (71.0% tactical 14d, 20.6% burst 3d, 8.4% strategic 60d)",
                            "pr_auc": 0.5776,
                            "roc_auc": 0.9947,
                            "multiplier_over_chance": 267.6,
                            "precision_at_10": 0.90,
                            "precision_at_20": 0.80,
                            "precision_at_50": 0.78,
                            "precision_at_100": 0.68,
                            "description": "Optional model for strategic macro drift analysis, slow preparatory reconnaissance, and multi-tempo attack correlation.",
                        },
                    ],
                }
            )
            return

        data = get_dataset(requested_model)
        if not data:
            self.send_json_response(
                {"error": f"Failed to load dataset for model '{requested_model}'. Check backend logs."},
                status=503,
            )
            return

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
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload both 14d and 60d datasets into RAM at startup",
    )
    args = parser.parse_args()

    # Preload default dataset
    print(f"[api_server] Initializing data repositories...")
    get_dataset("14d")
    if args.preload:
        print(f"[api_server] Preloading optional 60-day strategic dataset...")
        get_dataset("60d")

    server_address = (args.host, args.port)
    httpd = http.server.HTTPServer(server_address, SOCAPIHandler)

    print("=" * 74)
    print("      CYBER-GNN // FORENSIC EXPLAINABILITY REST API FOR DESIGNERS")
    print("=" * 74)
    print(f" API Base URL:       http://{args.host}:{args.port}/")
    print(f" Default Model:      14-Day Production GRU (Tactical Triage)")
    print(f" Optional Model:     60-Day Multi-Scale GRU (?model=60d)")
    print(f" Models Endpoint:    http://{args.host}:{args.port}/api/models")
    print(f" Alerts Endpoint:    http://{args.host}:{args.port}/api/incidents")
    print(f" CORS Enabled:       Access-Control-Allow-Origin: *")
    print(" Press Ctrl+C to terminate server.")
    print("=" * 74)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[api_server] Shutting down REST API server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
