"""Serve the viewer for local use, with caching disabled so edits show up on a normal reload.

    python3 viewer/tools/serve.py [port]        # default 8123, then open http://localhost:8123
"""
import functools
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_py_bundle  # noqa: E402


class NoCache(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/py/distortkit.zip":      # the DistortKit package for the in-browser analysis
            data = build_py_bundle.build()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the terminal quiet except for errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    handler = functools.partial(NoCache, directory=ROOT)
    with http.server.ThreadingHTTPServer(("", port), handler) as srv:
        print(f"Phonon Viewer at http://localhost:{port}  (Ctrl+C to stop)")
        srv.serve_forever()
