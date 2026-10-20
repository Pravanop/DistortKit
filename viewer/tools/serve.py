"""Serve the viewer for local use, with caching disabled so edits show up on a normal reload.

    python3 viewer/tools/serve.py [port]        # default 8123, then open http://localhost:8123
"""
import functools
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class NoCache(http.server.SimpleHTTPRequestHandler):
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
