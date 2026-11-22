"""Tiny static server for local use: python tools/serve.py [port]"""
import functools
import http.server
import os
import sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
print(f"Splitter running at http://localhost:{port}")
http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
