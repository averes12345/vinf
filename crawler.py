#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import re
import signal
import sys
import threading
import time
import typing
import urllib.parse
import urllib.robotparser

import requests


def parse_kv(s: str) -> typing.Tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Header must be KEY=VALUE")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


@dataclasses.dataclass(frozen=True)
class Config:

    # crawler settings
    seeds: frozenset[str] = frozenset({"www.nhs.uk/medicines/"})
    allowed_scopes: frozenset[str] = seeds
    user_agent: str = "FIIT-IR-Crawler/1.0 (xveresa@stuba.sk)"
    request_timeout: int = 5
    per_host_delay: float = 5
    max_depth: int = 5
    max_pages: int = 2000
    retry_max: int = 2
    retry_backoff_base: float = 15

    # io locations
    output_tsv: str = "pages.tsv"  # crawl log
    html_dir: str = "crawled_pages/"  # saved html dir
    state_dir: str = "crawled_pages/.state"  # crawled page dir
    visited_tsv: str = "crawled_pages/.state/visited.tsv"  # visited reference
    frontier_tsv: str = "crawled_pages/.state/frontier.tsv"  # que

    headers: tuple[tuple[str, str], ...] = (
        ("Accept", "text/html"),
        ("Accept-Language", "en"),
        ("From", "xveresa@stuba.sk"),
    )

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "Config":
        F = cls.__dataclass_fields__
        d = {k: F[k].default for k in F}

        p = argparse.ArgumentParser(description="Polite HTML crawler")
        p.add_argument(
            "--seed",
            dest="seeds",
            nargs="*",
            action="extend",
            default=argparse.SUPPRESS,
            help="Seed URL(s). Defaults to dataclass value.",
        )
        p.add_argument(
            "--scope",
            dest="allowed_scopes",
            nargs="*",
            action="extend",
            default=argparse.SUPPRESS,
            help="Allowed URL prefix (repeatable). Example: --scope https://www.nhs.uk/medicines/",
        )

        p.add_argument("--max-pages", type=int, default=d["max_pages"])
        p.add_argument("--max-depth", type=int, default=d["max_depth"])
        p.add_argument("--per-host-delay", type=float, default=d["per_host_delay"])
        p.add_argument(
            "--timeout", dest="request_timeout", type=int, default=d["request_timeout"]
        )
        p.add_argument("--retry-max", type=int, default=d["retry_max"])
        p.add_argument("--retry-backoff", type=float, default=d["retry_backoff_base"])
        p.add_argument("--user-agent", default=d["user_agent"])
        p.add_argument("--output", dest="output_tsv", default=d["output_tsv"])
        p.add_argument("--html-dir", dest="html_dir", default=d["html_dir"])
        p.add_argument("--state-dir", default=None)
        p.add_argument("--visited-tsv", default=None)
        p.add_argument("--frontier-tsv", default=None)

        p.add_argument(
            "--header",
            type=parse_kv,
            action="append",
            default=[],
            help="Extra header KEY=VALUE (repeat). Example: --header From=you@uni",
        )

        # print(d["seeds"])
        args = p.parse_args(argv)
        merged_headers: typing.Tuple[typing.Tuple[str, str], ...] = tuple(
            d["headers"]
        ) + tuple(args.header)
        if args.seeds is None:
            args.seeds = d["seeds"]
        if args.allowed_scopes is None:
            args.allowed_scopes = d["allowed_scopes"]
        html_dir = args.html_dir
        state_dir = args.state_dir or os.path.join(html_dir, ".state")
        visited_tsv = args.visited_tsv or os.path.join(state_dir, "visited.tsv")
        frontier_tsv = args.frontier_tsv or os.path.join(state_dir, "frontier.tsv")
        scopes = frozenset(_norm_scope(s) for s in args.allowed_scopes)
        seeds = frozenset(_norm_scope(s) for s in args.seeds)

        return cls(
            seeds=seeds,
            allowed_scopes=scopes,
            user_agent=args.user_agent,
            request_timeout=args.request_timeout,
            per_host_delay=args.per_host_delay,
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            retry_max=args.retry_max,
            retry_backoff_base=args.retry_backoff,
            output_tsv=args.output_tsv,
            html_dir=html_dir,
            state_dir=state_dir,
            visited_tsv=visited_tsv,
            frontier_tsv=frontier_tsv,
            headers=merged_headers,
        )


class Shutdown:
    def __init__(self):
        self.stop = threading.Event()
        self.signals_seen = 0

    def handler(self, signum, frame):
        self.signals_seen += 1
        if self.signals_seen == 1:
            print(f"[{signum}] graceful shutdown…", file=sys.stderr)
            self.stop.set()
        else:
            print(f"[{signum}] forcing exit now.", file=sys.stderr)
            sys.exit(1)


HREF_RE = re.compile(r"""<a\s[^>]*?href\s*=\s*(['"])(.*?)\1""", re.I)


def extract_links(html: str, base_url: str) -> typing.Iterable[str]:
    for _, href in HREF_RE.findall(html):
        href = href.strip()
        if not href:
            continue
        absu = urllib.parse.urljoin(base_url, href)
        absu, _ = urllib.parse.urldefrag(absu)
        yield absu


def _clean_url(u: str) -> str:
    # lightweight cleanup: strip whitespace and drop fragment (for pretty print)
    return urllib.parse.urldefrag(u.strip())[0]


SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]")


def pretty_filename(url: str) -> str:
    # create a single, flat, human-readable filename from the FULL normalized URL:
    s = _clean_url(url)
    s = s.replace("/", "_")
    s = SANITIZE_RE.sub("_", s)
    if not s:
        s = "_"
    return s + ".html"


def _norm_scope(s: str) -> str:
    # normalize a scope prefix to 'scheme://host/path/' (lowercased host, path ends with '/')
    if "://" not in s:
        s = "https://" + s
    u = urllib.parse.urlsplit(s)
    scheme = u.scheme or "https"
    netloc = (u.netloc or u.path).lower()
    path = u.path if u.netloc else ""
    if not path:
        path = "/"
    if not path.endswith("/"):
        path += "/"
    return urllib.parse.urlunsplit((scheme, netloc, path, "", ""))


def _url_prefix(u: str) -> str:
    # return 'scheme://host/path' for prefix comparison (no query/fragment)
    pu = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((pu.scheme, pu.netloc.lower(), pu.path, "", ""))


class Crawler:
    def __init__(self, cfg: Config, stop_event: threading.Event):
        self.cfg = cfg
        self.stop = stop_event

        os.makedirs(cfg.html_dir, exist_ok=True)
        os.makedirs(cfg.state_dir, exist_ok=True)

        self.visited: set[str] = set()
        self.frontier: collections.deque[tuple[str, int]] = collections.deque()
        self.per_host_next_time: dict[str, float] = {}
        self.robots_by_host: dict[str, urllib.robotparser.RobotFileParser | None] = {}

        if not os.path.exists(cfg.output_tsv):
            with open(cfg.output_tsv, "w", encoding="utf-8") as f:
                f.write("url\tstatus\tcontent_type\tpath\tbytes\tdepth\ttime\n")

        self._load_state_or_seed()

        self.session = requests.Session()
        self.session.headers.update({k: v for k, v in cfg.headers})
        self.session.headers["User-Agent"] = cfg.user_agent

    def _in_scope(self, url: str) -> bool:
        if not self.cfg.allowed_scopes:
            return True
        up = _url_prefix(url)
        return any(up.startswith(scope) for scope in self.cfg.allowed_scopes)

    def _save_state(self) -> None:

        with open(self.cfg.visited_tsv, "w", encoding="utf-8") as f:
            f.write("url\n")
            for u in sorted(self.visited):
                f.write(f"{u}\n")

        with open(self.cfg.frontier_tsv, "w", encoding="utf-8") as f:
            f.write("url\tdepth\n")
            for u, d in self.frontier:
                f.write(f"{u}\t{d}\n")

    def _load_state_or_seed(self) -> None:
        if os.path.exists(self.cfg.visited_tsv) and os.path.exists(
            self.cfg.frontier_tsv
        ):

            with open(self.cfg.visited_tsv, "r", encoding="utf-8") as f:
                next(f, None)  # skip header
                for line in f:
                    u = line.strip()
                    if u:
                        self.visited.add(u)

            with open(self.cfg.frontier_tsv, "r", encoding="utf-8") as f:
                next(f, None)  # skip header
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        u, d = parts[0], int(parts[1])
                        if self._in_scope(u):
                            self.frontier.append((u, d))
            print(
                f"[resume] visited={len(self.visited)} frontier={len(self.frontier)}",
                file=sys.stderr,
            )
        else:
            for s in self.cfg.seeds:
                u = _clean_url(s)
                if self._in_scope(u):
                    self.frontier.append((u, 0))
            print(f"[seed] queued {len(self.frontier)}", file=sys.stderr)

    def _allowed_by_robots(self, url: str) -> bool:
        host = urllib.parse.urlparse(url).hostname or ""
        if host not in self.robots_by_host:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = f"{urllib.parse.urlparse(url).scheme}://{host}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
            except Exception:
                self.robots_by_host[host] = None
                return False
            self.robots_by_host[host] = rp
        rp = self.robots_by_host[host]
        if rp is None:
            return False
        return rp.can_fetch(self.cfg.user_agent, url)

    def _sleep_until_allowed(self, host: str):
        now = time.time()
        t = self.per_host_next_time.get(host, 0.0)
        if t > now:
            time.sleep(t - now)
        self.per_host_next_time[host] = time.time() + self.cfg.per_host_delay

    def run(self):
        pages_fetched = 0
        last_save = time.time()

        try:
            while (
                self.frontier
                and not self.stop.is_set()
                and pages_fetched < self.cfg.max_pages
            ):
                url, depth = self.frontier.popleft()
                url = _clean_url(url)
                if url in self.visited:
                    continue

                host = urllib.parse.urlparse(url).hostname or ""
                if not self._in_scope(url):
                    continue
                if depth > self.cfg.max_depth:
                    continue
                if not self._allowed_by_robots(url):
                    continue

                self._sleep_until_allowed(host)

                status, ctype, bytes_len, saved_path, body = self._fetch(url)
                self._log_row(url, status, ctype, saved_path, bytes_len, depth)

                self.visited.add(url)
                pages_fetched += 1

                if (
                    status == 200
                    and body
                    and ctype
                    and "text/html" in (ctype or "").lower()
                ):
                    for link in extract_links(body, url):
                        n = _clean_url(link)
                        if n not in self.visited and self._in_scope(n):
                            self.frontier.append((n, depth + 1))

                # periodic save (this was needed in the beggining XD)
                if time.time() - last_save > 10:
                    self._save_state()
                    last_save = time.time()

        finally:
            self._save_state()
            print(
                f"[done] fetched={pages_fetched} visited={len(self.visited)} queued={len(self.frontier)}",
                file=sys.stderr,
            )

    def _fetch(self, url: str):
        saved_path = ""
        body = ""
        attempt = 0
        backoff = self.cfg.retry_backoff_base

        while attempt <= self.cfg.retry_max and not self.stop.is_set():
            try:
                r = self.session.get(
                    url, timeout=self.cfg.request_timeout, allow_redirects=True
                )
                status = r.status_code
                ctype = r.headers.get("Content-Type", "")
                data = r.text if "text" in (ctype or "").lower() else ""
                if status == 200 and data:
                    fname = pretty_filename(url) 
                    saved_path = os.path.join(self.cfg.html_dir, fname)
                    with open(
                        saved_path, "w", encoding=r.encoding or "utf-8", errors="ignore"
                    ) as f:
                        f.write(data)
                    body = data
                    return status, ctype, len(r.content or b""), saved_path, body
                else:
                    if 500 <= status < 600:
                        raise requests.RequestException(f"server {status}")
                    return status, ctype, len(r.content or b""), saved_path, body
            except Exception:
                attempt += 1
                if attempt > self.cfg.retry_max:
                    return 0, "", 0, saved_path, body
                time.sleep(backoff)
                backoff *= 2
        return 0, "", 0, saved_path, body

    def _log_row(
        self, url: str, status: int, ctype: str, path: str, nbytes: int, depth: int
    ):
        print(url, status, ctype, path, nbytes, depth)
        with open(self.cfg.output_tsv, "a", encoding="utf-8") as f:
            f.write(
                f"{url}\t{status}\t{ctype}\t{path}\t{nbytes}\t{depth}\t{int(time.time())}\n"
            )


if __name__ == "__main__":
    cfg = Config.from_args()
    shutdown = Shutdown()
    signal.signal(signal.SIGINT, shutdown.handler)
    signal.signal(signal.SIGTERM, shutdown.handler)

    # print(cfg)  # temporary: see what got parsed
    crawler = Crawler(cfg, stop_event=shutdown.stop)
    crawler.run()
