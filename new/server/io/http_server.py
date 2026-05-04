"""Tiny static-file HTTP server for the UI.

Browsers refuse ES module imports from file:// URLs (cross-origin/CORS),
so serving ui/ over http://127.0.0.1:<port>/ is the simplest fix.
Runs in its own thread so it stays out of the asyncio loop's way.
"""
from __future__ import annotations

import http.server
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class StaticHTTPServer:
    def __init__(self, host: str, port: int, root: Path):
        self.host = host
        self.port = port
        self.root = root.resolve()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        root_str = str(self.root)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=root_str, **kw)

            def log_message(self, fmt, *args):
                # silence access logs; errors still surface via log_error
                pass

        self._server = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="http-static", daemon=True)
        self._thread.start()
        log.info("static http server listening on http://%s:%d (serving %s)",
                 self.host, self.port, self.root)

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
