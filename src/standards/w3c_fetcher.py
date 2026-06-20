from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

from .w3c_models import W3CSourceDocument


class W3CFetcher:

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def fetch_or_cache(self, source: W3CSourceDocument, refresh: bool, timeout_seconds: int, allow_network: bool) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{source.key}.html"
        if cache_path.exists() and not refresh:
            return cache_path.read_text(encoding="utf-8")
        if not allow_network:
            raise FileNotFoundError(
                f"Missing cached W3C source: {cache_path}. "
                "Run src/experiments/build_w3c_usability_spec.py --refresh to update local cache."
            )
        request = Request(source.url, headers={"User-Agent": "ui-usability-evaluation/0.1"})
        with urlopen(request, timeout=timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
        cache_path.write_text(html, encoding="utf-8")
        return html
