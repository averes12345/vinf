#!/usr/bin/env python3
import csv
import gzip
import html
import json
import os
import re
import sys
from datetime import timezone
from typing import cast

from dateutil import parser

# TODO: should look into using more metadata if necessary

# (?is) set flags (dont care about case and want to match \n)
# find the starting title tag
# match anything, which is not the closing tag and strip whitespace (non greedy just in case, but unnecessary)
RE_TITLE = re.compile(r"(?is)<title[^>]*>\s*(.*?)\s*</title>")

# NOTE: this should not be necessary for nhs data, but might well be needed for other sources

# using lookaeads ?= just in case
# find start tag
# ensure that before the end of the tag there is rel=canonical
# find the href and capture inside
# consume the rest inside the tag

RE_CANON = re.compile(
    r"""(?is)
    <link\b
    (?=[^>]*\brel\s*=\s*["']canonical["'])
    (?=[^>]*\bhref\s*=\s*["']([^"']+) ["']) 
    [^>]*> """,
    re.X,
)

# same as the previous one
RE_DESC = re.compile(
    r"""(?is)
    <meta\b
    (?=[^>]*\bname\s*=\s*["']description["'])
    (?=[^>]*\bcontent\s*=\s*["'] ([^"']*)["'])
    [^>]*> """,
    re.X,
)


RE_ART_MOD = re.compile(
    r"""(?is)
  <meta\b
    (?=[^>]*\bproperty\s*=\s*["']article:modified_time["'])
    (?=[^>]*\bcontent\s*=\s*["']([^"']+)["'])
  [^>]*>
""",
    re.X,
)


RE_ART_PUB = re.compile(
    r"""(?is)
  <meta\b
    (?=[^>]*\bproperty\s*=\s*["']article:published_time["'])
    (?=[^>]*\bcontent\s*=\s*["']([^"']+)["'])
  [^>]*>
""",
    re.X,
)

# (?is) set flags (dont care about case and want to match \n)
# find start of tag starting with anything we dont care about eg: <header class:someclass
# match anything, which is not the closing tag
# match the first closing tag if it is present
# match the closing tag for the captured starting tag we dont care about eg: <\header>
RE_HEADER_NAV_FOOTER_SCRIPTS_STYPES = re.compile(
    r"(?is)<(header|nav|footer|script|style|aside)\b[^>]*>.*?</\1>"
)

RE_NHS_SKIP_LINK = re.compile(
    r""" (?is)
  <a\b
  (?=[^.]*\bclass\s*=\s*["']nhsuk-skip-link["'])
  [^.]*>
  """,
    re.X,
)

# NEW: "Related conditions" widget:
#   <div class="beta-hub-related-links-title"> ... Related conditions ... </div>
#   <ul class="beta-hub-related-links"> ... <a>Condition</a> ... </ul>
RE_RELATED_BLOCK = re.compile(
    r"""
    (?is)
    <div\b[^>]*class=["'][^"']*beta-hub-related-links-title[^"']*["'][^>]*>
        .*?Related\s+conditions.*?
    </div>
    \s*
    <ul\b[^>]*class=["'][^"']*beta-hub-related-links[^"']*["'][^>]*>
        (.*?)
    </ul>
    """,
    re.X,
)

RE_RELATED_LINK = re.compile(r'(?is)<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>')


# colapse all whitespace except newline to a single space and unescape escaped characters
# TODO: weird unicode characters ??
def norm(s: str | None) -> str | None:
    if not s:
        return None
    s = html.unescape(s)
    s = s.strip("\n")
    s = re.sub(r"[^\S\n]+", " ", s)
    # erase space near newline
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s


def strip_tags(s: str | None) -> str:
    return re.sub(r"(?is)<[^>]+>", "", s or "").strip()


def extract_related_conditions_html(html_text: str) -> list[str]:
    """
    Look for the NHS "Related conditions" widget and return a list of
    condition names, e.g.:

      ["Chickenpox", "Cold sores", "Genital herpes", ...]
    """
    names: list[str] = []

    for m in RE_RELATED_BLOCK.finditer(html_text):
        ul_html = m.group(1)
        for m2 in RE_RELATED_LINK.finditer(ul_html):
            link_text = m2.group(2)
            name = norm(strip_tags(link_text))
            if name:
                names.append(name)

    # dedupe while preserving order
    seen = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


# PERF: opening for .html files (max 204 387 bytes so far therego big, but simple)
def open_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return f.read().decode("utf-8", "ignore")
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "ignore")


TSV_COLUMNS = ["url", "status", "content_type", "path", "bytes", "depth", "time"]


def read_pages_tsv(tsv_path: str):
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        sample = f.readline()
        f.seek(0)
        has_header = sample.lower().startswith("url\tstatus\tcontent_type\tpath")
        if has_header:
            reader = csv.DictReader(f, delimiter="\t")
        else:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=TSV_COLUMNS)
        for row in reader:
            yield row


def should_extract(row: dict) -> bool:
    status_ok = str(row.get("status", "")).strip() == "200"
    ctype = (row.get("content_type") or "").lower()
    is_html = "text/html" in ctype or "application/xhtml" in ctype or ctype == ""
    return status_ok and is_html and bool(row.get("path"))


# TODO: probably need to also parse the scripts.json, because there are things like brand name
def extract_fields(html_text: str, fetched_url: str | None = None) -> dict:
    h = RE_HEADER_NAV_FOOTER_SCRIPTS_STYPES.sub("", html_text)
    h = RE_NHS_SKIP_LINK.sub("", h)
    h = norm(h) or ""

    related_conditions = extract_related_conditions_html(h)

    m = RE_TITLE.search(h)
    # since we normalize here it should not be necessary afterwards
    title = m.group(1) if m else None

    m = RE_CANON.search(h)
    canonical = m.group(1) if m else fetched_url

    m = RE_DESC.search(h)
    description = m.group(1) if m else None

    m = RE_ART_MOD.search(h)
    date_modified = m.group(1) if m else ""
    try:
        date_modified_iso = parser.parse(date_modified)
        if date_modified_iso.tzinfo is None:  # if input has no timezone, pick one
            date_modified_iso = date_modified_iso.replace(tzinfo=timezone.utc)
        date_modified_iso = date_modified_iso.isoformat()
    except Exception:
        date_modified_iso = None

    body_text = norm(strip_tags(h))

    return {
        "data": {
            "title": title,
            "description": description,
            "body_text": body_text,
            "related_conditions": related_conditions,
        },
        "meta": {
            "canonical_url": canonical,
            "fetched_url": fetched_url,
            "date_modified": date_modified_iso,
        },
    }


def build_output_paths(out_dir: str, extractor_version: str | int):
    v = str(extractor_version)
    jsonl_path = os.path.join(out_dir, f"extracted_v{v}.jsonl")
    manifest_path = os.path.join(out_dir, f"extracted_pages_v{v}.tsv")
    return jsonl_path, manifest_path


def main(pages_tsv: str, html_root: str, out_dir: str):
    """
    version 1 = what I presented
    version 2 extracts related conditions for wikipedia page linking
    """
    extractor_version = 2
    out_jsonl, out_manifest_tsv = build_output_paths(out_dir, extractor_version)

    # ensure output directories exist
    os.makedirs(out_dir or ".", exist_ok=True)

    jsonl_f = open(out_jsonl, "w", encoding="utf-8", newline="")
    manifest_f = open(out_manifest_tsv, "w", encoding="utf-8", newline="")
    manifest_writer = csv.writer(manifest_f, delimiter="\t")

    # header for the audit manifest
    manifest_writer.writerow(
        [
            "canonical_url",
            "fetched_url",
            "source_path",
            "status",
            "title",
            "date_modified",
            "extract_status",
            "error",
        ]
    )

    count_in = count_ok = 0

    for row in read_pages_tsv(pages_tsv):
        count_in += 1
        url = row.get("url")
        path = cast(str, row.get("path"))
        status = row.get("status")
        ctype = row.get("content_type", "")

        # filter out non-HTML or failed fetches
        if not should_extract(row):
            manifest_writer.writerow(
                [
                    "",  # canonical_url (unknown on skip)
                    url,  # fetched_url
                    path,  # html_path
                    status,  # status
                    "",  # title
                    "",  # date_modified
                    "skipped",  # extract_status
                    "filter",  # error / reason
                ]
            )
            continue
        try:
            full_path = os.path.join(html_root, path)
            html_text = open_text(full_path)
            extracted = extract_fields(html_text, fetched_url=url)

            # add trace fields that help debugging later
            extracted["meta"]["source_path"] = path
            extracted["meta"]["status"] = (
                int(status) if status and status.isdigit() else status
            )

            # write one line of JSON (ndjson)
            jsonl_f.write(json.dumps(extracted, ensure_ascii=False) + "\n")

            # mark success in the manifest
            manifest_writer.writerow(
                [
                    extracted["meta"].get(
                        "canonical_url", ""
                    ),  # canonical_url (unknown on skip)
                    url,  # fetched_url
                    path,  # html_path
                    status,  # status
                    extracted["data"].get("title"),  # title
                    extracted["meta"].get("date_modified"),  # date_modified
                    "processed",  # extract_status
                    "",  # error / reason
                ]
            )
            count_ok += 1

        except Exception as e:
            # If anything fails on this row, record the error and keep going
            manifest_writer.writerow(
                [
                    "",  # canonical_url (unknown on skip)
                    url,  # fetched_url
                    path,  # html_path
                    status,  # status
                    "",  # title
                    "",  # date_modified
                    "error",  # extract_status
                    e,  # error / reason
                ]
            )

    jsonl_f.flush()
    os.fsync(jsonl_f.fileno())
    jsonl_f.close()
    manifest_f.flush()
    os.fsync(manifest_f.fileno())
    manifest_f.close()

    # quick summary to stderr
    sys.stderr.write(
        f"Processed {count_in} rows, extracted {count_ok}, wrote {out_jsonl} and {out_manifest_tsv}\n"
    )


if __name__ == "__main__":
    # expect exactly three args: pages.tsv, html_root_path , output dir
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: extractor.py pages.tsv html_root_path output dir \n")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
