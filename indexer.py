#!/usr/bin/env python3

import argparse
import json
import math
import os
# PERF: pickle is much faster, but not human readable :( (bad for showing the index XD)
# import pickle
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from zoneinfo import ZoneInfo

WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')

_INDEX_REGISTRY: Dict[str, Type["BaseInvertedIndex"]] = {}


def register_index(cls: Type["BaseInvertedIndex"]) -> Type["BaseInvertedIndex"]:
    _INDEX_REGISTRY[cls.__name__] = cls
    return cls


@dataclass
class TFPositions:
    tf: int
    positions: List[int]


@dataclass
class TermBucket:
    docs: Dict[Any, TFPositions] = field(default_factory=dict)


class BaseInvertedIndex(ABC):

    def __init__(self):
        # term -> (url, (tf, [positions]))
        self.postings: Dict[str, TermBucket] = {}
        # term -> iwf
        self.idf_cache: Dict[str, float] = {}
        # document id -> term
        self.reverse_postings: Dict[str, Set[str]] = defaultdict(set)

        # NOTE: might use for scoring
        # doc_id -> {"meta": {...}}
        self.docs: Dict[Any, Dict] = {}

        # corpus stats
        self.N: int = 0

    @abstractmethod
    def _finalize(self) -> None:
        ...

    def finalize(self) -> None:
        self._finalize()


    # NOTE: we are doing this like this and not just passing fulltext, because we know we will need to use lucine eventually and so having the text
    # setup to be extracted more properly is beneficial
    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts = (BaseInvertedIndex._flatten_text(self, v) for v in value)
            return " ".join(p for p in parts if p)
        if isinstance(value, dict):
            parts = (BaseInvertedIndex._flatten_text(self, v) for v in value.values())
            return " ".join(p for p in parts if p)
        return str(value)

    def build(self, jsonl_path: str) -> int:
        added = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Line:\t{lineno};\tError{e}")
                    continue

                data = rec.get("data")
                meta = rec.get("meta", {}) or {}

                text = self._flatten_text(data)
                if not text:
                    continue

                doc_id = meta.get("canonical_url")

                self.add_document(doc_id, text, meta)
                added += 1
        return added

    def tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]

    def _collect_hit_positions(self, doc_id: Any, terms: List[str]) -> Dict[str, List[int]]:
        hits: Dict[str, List[int]] = {}
        for t in set(terms):
            bucket = self.postings.get(t)
            if not bucket:
                continue
            tfpos = bucket.docs.get(doc_id)
            if tfpos:
                hits[t] = tfpos.positions
        return hits

    def make_snippet(
        self,
        doc_id: Any,
        query: str,
        max_tokens: int = 40,
        max_fragments: int = 1,
        highlight: bool = True,
    ) -> str:

        pairs = TOKEN_RE.findall(query or "")
        atoms = [(g1 or g2) for g1, g2 in pairs]

        # normalize atoms into the same tokens your index uses
        q_terms: List[str] = []
        for a in atoms:
            if a.upper() in ("AND", "OR", "NOT"):
                continue
            q_terms.extend(self.tokenize(a))

        if not q_terms:
            return ""

        rec = self.docs.get(doc_id, {})
        toks = rec.get("tokens")
        if not isinstance(toks, list):
            return ""
        doc_tokens: List[str] = toks

        hits_by_term = self._collect_hit_positions(doc_id, q_terms)

        # should not happen
        if not hits_by_term:
            return ""
        # weighted hits: prefer rarer terms (idf if subclass provides it)
        weighted: List[Tuple[int, float]] = []
        for t, poss in hits_by_term.items():
            try:
                idf_val = getattr(self, "_idf")(t)  # works for TF-IDF/IDF/... children
            except Exception:
                idf_val = 1.0
            w = max(idf_val, 0.0)
            for p in poss:
                weighted.append((p, w))

        if not weighted:
            frag = " ".join(doc_tokens[:max_tokens])
            return frag + (" …" if len(doc_tokens) > max_tokens else "")

        # greedy window selection
        weighted.sort(key=lambda x: (-x[1], x[0]))
        windows: List[Tuple[int, int, float]] = []
        covered: Set[int] = set()
        half = max(1, max_tokens // 2)

        for pos, w in weighted:
            if pos in covered:
                continue
            start = max(0, pos - half)
            end = min(len(doc_tokens), start + max_tokens)

            # mark coverage
            for i in range(start, end):
                covered.add(i)

            # window score = sum weights of hits inside
            score = 0.0
            for p2, w2 in weighted:
                if start <= p2 < end:
                    score += w2

            windows.append((start, end, score))
            if len(windows) >= max_fragments:
                break

        windows.sort(key=lambda z: z[0])

        # render with optional highlighting for visibility
        qset = set(q_terms)
        frags: List[str] = []
        for start, end, _ in windows:
            toks = doc_tokens[start:end]
            if highlight:
                HL = "\x1b[7;1m"   # reverse + bold
                RST = "\x1b[0m"
                rendered = [f"{HL}{t}{RST}" if t in qset else t for t in toks]
                frag = " ".join(rendered)
            else:
                frag = " ".join(toks)

            if start > 0:
                frag = "… " + frag
            if end < len(doc_tokens):
                frag = frag + " …"
            frags.append(frag)

        return "  …  ".join(frags)

    def _on_add(self, doc_id: Any, tokens: List[str]) -> None:
        return

    def _on_remove(self, doc_id: Any) -> None:
        return

    # ---------------- Ingestion ----------------
    def add_document(self, doc_id: Any, text: str, meta: Optional[Dict] = None) -> None:
        if doc_id in self.docs:
            self.remove_document(doc_id)
 
        tokens = self.tokenize(text)
        per_doc_positions: Dict[str, List[int]] = defaultdict(list)
        for i, tok in enumerate(tokens):
            per_doc_positions[tok].append(i)

        for term, pos_list in per_doc_positions.items():
            self.postings.setdefault(term, TermBucket()).docs[doc_id] = TFPositions(
                tf=len(pos_list), positions=pos_list
            )
            self.reverse_postings[doc_id].add(term)

        self.docs[doc_id] = {"meta": meta or {}, "tokens": tokens} 
        self.N += 1
        self.idf_cache.clear()
        self._on_add(doc_id, tokens)

    def remove_document(self, doc_id: Any) -> None:
        if doc_id not in self.docs:
            return
    
        # give subclasses a chance to react while all structures still exist
        self._on_remove(doc_id)
    
        terms_for_doc = self.reverse_postings.get(doc_id, set())
    
        # remove (doc_id) from each term bucket; delete empty buckets
        for term in list(terms_for_doc):
            bucket = self.postings.get(term)
            if not bucket:
                continue
            if doc_id in bucket.docs:
                del bucket.docs[doc_id]
                if not bucket.docs:
                    del self.postings[term]
    
        # remove reverse mapping for this doc
        self.reverse_postings.pop(doc_id, None)
    
        self.docs.pop(doc_id, None)
        self.N -= 1
    
        self.idf_cache.clear()

    def _term_docs(self, term_raw: str) -> Set[Any]:
        term = term_raw.lower()
        bucket = self.postings.get(term)
        return set(bucket.docs.keys()) if bucket else set()

    def _phrase_docs(self, phrase: str) -> Set[Any]:
        terms = self.tokenize(phrase)
        if not terms:
            return set()

        base_bucket = self.postings.get(terms[0])
        if not base_bucket or not base_bucket.docs:
            return set()

        result: Set[Any] = set()

        for doc_id, tfpos0 in base_bucket.docs.items():

            current = set(tfpos0.positions)
            ok = True

            for t in terms[1:]:
                bucket_t = self.postings.get(t)
                if not bucket_t:
                    ok = False
                    break
                tfpos_t = bucket_t.docs.get(doc_id)
                if tfpos_t is None:
                    ok = False
                    break

                plist = set(tfpos_t.positions)
                current = {p + 1 for p in current} & plist

                if not current:
                    ok = False
                    break
            if ok:
                result.add(doc_id)
        return result

    @abstractmethod
    def _score_terms(
        self, terms: List[str], candidates: Optional[Set[Any]]
    ) -> Dict[Any, float]: ...

    def _boolean_merge(self, tokens: List[str]) -> Set[Any]:
        current: Optional[Set[Any]] = None
        last_op = "AND"

        for tok in tokens:
            up = tok.upper()
            if up in ("AND", "OR", "NOT"):
                last_op = up
                continue

            ds = self._phrase_docs(tok)

            if current is None:
                current = ds if last_op != "NOT" else (set(self.docs.keys()) - ds)
            else:
                if last_op == "AND":
                    current &= ds
                elif last_op == "OR":
                    current |= ds
                elif last_op == "NOT":
                    current -= ds

            last_op = "AND"
        return current or set()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
        pairs = TOKEN_RE.findall(query or "")
        tokens = [(g1 or g2) for g1, g2 in pairs]
        # NOTE: should not be necessary unless the user enters something like '""  ""' or ''
        tokens = [t for t in tokens if t.strip() != ""]

        if not tokens:
            return []

        # NOTE: if implementing term search is not needed I need to do something like this
        # tokens = WORD_RE.split(query) if query else []

        candidate = self._boolean_merge(tokens)

        terms_for_scoring: List[str] = []
        for tok in tokens:
            up = tok.upper()
            if up in ("AND", "OR", "NOT"):
                continue
            norm = self.tokenize(tok)
            if norm:
                terms_for_scoring.extend(norm)

        scores = self._score_terms(terms_for_scoring, candidate)

        # NOTE: might want to help with scoring for phrase terms
        # WARNING:
        # TODO: need to finish based on score_terms implementation
        if candidate:
            scores = {d: s for d, s in scores.items() if d in candidate}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    # ---------------- Persistence (shared) ----------------
    @abstractmethod
    def _state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def _load_state_dict(self, state: Dict[str, Any]) -> None: ...

    def save(self, path: str) -> None:
        # postings -> json-serializable
        json_postings: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for term, bucket in self.postings.items():
            # bucket.docs is Dict[Any, TFPositions]
            docs_map: Dict[str, Dict[str, Any]] = {}
            for doc_id, tfpos in bucket.docs.items():
                docs_map[str(doc_id)] = {"tf": tfpos.tf, "positions": list(tfpos.positions)}
            json_postings[term] = docs_map
    
        # reverse_postings defaultdict(set) -> plain dict[str, list[str]]
        json_rev = {str(doc_id): sorted(list(terms)) for doc_id, terms in self.reverse_postings.items()}
    
        # docs: ensure keys are strings (values should already be JSON-serializable dicts)
        json_docs = {str(doc_id): doc for doc_id, doc in self.docs.items()}
    
        state = {
            "cls": self.__class__.__name__,
            "core": {
                "postings": json_postings,
                "docs": json_docs,
                "N": self.N,
                "reverse_postings": json_rev,
                "idf_cache": self.idf_cache,   # ok as-is
            },
            "extra": self._state_dict(),       # subclass-provided
        }
    
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "BaseInvertedIndex":
        with open(path, "rt", encoding="utf-8") as f:
            s = json.load(f)
    
        name = s["cls"]
        subcls = _INDEX_REGISTRY.get(name)
        if subcls is None:
            raise ValueError(f"Unknown index subclass '{name}'. Registered: {list(_INDEX_REGISTRY)}")
    
        inst: BaseInvertedIndex = subcls()
        core = s["core"]
    
        # rebuild postings
        json_postings: Dict[str, Dict[str, Dict[str, Any]]] = core["postings"]
        postings: Dict[str, TermBucket] = {}
        for term, docs_map in json_postings.items():
            tb = TermBucket()
            for doc_id, payload in docs_map.items():
                tf = int(payload["tf"])
                positions = list(payload["positions"])
                tb.docs[doc_id] = TFPositions(tf=tf, positions=positions)
            postings[term] = tb
        inst.postings = postings
    
        # docs
        inst.docs = core["docs"]
        inst.N = int(core["N"])
    
        # reverse_postings back to defaultdict(set)
        rp_src: Dict[str, List[str]] = core.get("reverse_postings", {})
        inst.reverse_postings = defaultdict(set, {doc_id: set(terms) for doc_id, terms in rp_src.items()})
    
        inst.idf_cache = core.get("idf_cache", {})
    
        # subclass extras
        inst._load_state_dict(s.get("extra", {}))
        return inst

@register_index
class ClassicTFIDFIndex(BaseInvertedIndex):
# log normalized term frequency * log inverted document frequency
    def __init__(self):
        super().__init__()

    def _finalize(self) -> None:
        N = self.N
        if N == 0:
            return
        for term, _b in self.postings.items():
            self.idf_cache[term] = self._idf(term) 

    def _idf(self, term: str) -> float:
        v = self.idf_cache.get(term)
        if v is not None:
            return v

        bucket = self.postings.get(term)
        df = len(bucket.docs) if bucket else 0
        N = self.N
        self.idf_cache[term] = math.log(N / df)
        return self.idf_cache[term]

    def _tf_weight(self, tf: int) -> float:
        return math.log(tf + 1.0)

    def _score_terms(
        self, terms: List[str], candidates: Optional[Set[Any]]
    ) -> Dict[Any, float]:
        scores: Dict[Any, float] = defaultdict(float)

        for term in terms:
            bucket = self.postings.get(term)
            # NOTE: if there is an error fix remove
            # if not bucket or not bucket.docs:
            if not bucket:
                continue

            idf = self._idf(term)
            if candidates is None:
                # if ever called without candidates, just score over the whole posting
                it = bucket.docs.keys()
            elif len(bucket.docs) <= len(candidates):
                it = ((d, tfpos) for d, tfpos in bucket.docs.items() if d in candidates)
            else:
                it = ((d, bucket.docs[d]) for d in candidates if d in bucket.docs)

            for doc_id, tfpos in it:
                scores[doc_id] += self._tf_weight(tfpos.tf) * idf

        return scores

   
    def _state_dict(self) -> Dict[str, Any]:
        return {}

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        return None




@register_index
class IDFIndex(BaseInvertedIndex):
    def __init__(self):
        super().__init__()

    def _finalize(self) -> None:
        N = self.N
        if N <= 0:
            return
        for term, _b in self.postings.items():
            self.idf_cache[term] = self._idf(term) 

    # cached IDF with lazy fill if not _finalized
    def _idf(self, term: str) -> float:
        v = self.idf_cache.get(term)
        if v is not None:
            return v

        bucket = self.postings.get(term)
        df = len(bucket.docs) if bucket else 0
        N = self.N
        v = math.log(N / df)
        self.idf_cache[term] = v
        return v

    #  term scoring limited to candidate docs
    def _score_terms(
        self, terms: List[str], candidates: Optional[Set[Any]]
    ) -> Dict[Any, float]:
        scores: Dict[Any, float] = defaultdict(float)
        if not terms:
            return scores

        for term in set(terms):
            bucket = self.postings.get(term)
            if not bucket:
                continue

            idf = self._idf(term)
            # try to iterate over candidates
            if candidates is None:
                # if ever called without candidates, just score over the whole posting
                iterator = bucket.docs.keys()
            else:
                # pick the smaller side to iterate for efficiency
                if len(bucket.docs) <= len(candidates):
                    iterator = (d for d in bucket.docs.keys() if d in candidates)
                else:
                    iterator = (d for d in candidates if d in bucket.docs)

            for doc_id in iterator:
                scores[doc_id] += idf

        return scores

    def _state_dict(self) -> Dict[str, Any]:
        return {}

    def _load_state_dict(self, state: Dict[str, Any]) -> None:
        return


class _SmartHelp(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def _print_hits(idx: "BaseInvertedIndex", hits: List[Tuple[Any, float]], time, query: str = "") -> None:
    for rank, (doc_id, score) in enumerate(hits, 1):
        doc = idx.docs.get(doc_id, {})
        # tokens = doc.get("tokens", {})
        meta = doc.get("meta", {})
        
        print(f"{rank:>2}.{doc_id}\nScore:{score:.6f}\tModified:{datetime.fromisoformat(meta['date_modified']).astimezone(ZoneInfo('Europe/Bratislava')).strftime('%Y-%m-%d %H:%M %Z')}")
        if query:
            snip = idx.make_snippet(doc_id, query, max_tokens=15, max_fragments=1, highlight=True)
            if snip:
                print(f"{snip}\n")
    print(f"# query execution time: {time * 1_000:.3f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mini-index",
        description=(
            "Build and search a simple inverted index over JSONL lines of the form:\n"
            '  {"data": {...}, "meta": {...}}\n'
        ),
        epilog=(
            "Examples:\n"
            "  # Build a new index from a JSONL file\n"
            "  mini-index --index idx.pkl --index-class ClassicTFIDFIndex build data.jsonl\n"
            "\n"
            "  # Search the index (phrases in quotes, boolean ops AND/OR/NOT)\n"
            '  mini-index --index idx.pkl search "\\"fat cat\\" OR orange" --top-k 20\n'
        ),
        formatter_class=_SmartHelp,
    )

    parser.add_argument(
        "--index",
        default="index.pkl",
        help="Path to the index file (created on build if missing).",
    )
    parser.add_argument(
        "--index-class",
        default="ClassicTFIDFIndex",
        help=f"When creating a new index, which subclass to use. Options: {list(_INDEX_REGISTRY)}",
    )

    # nice subparsers header + metavar
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        metavar="COMMAND",
        description="Valid commands are:",
    )

    # build
    p_build = subparsers.add_parser(
        "build",
        help="Build/append from a JSONL file with {'data':..., 'meta':...}.",
        description=(
            "Build (or append to) an index from a JSONL input.\n"
            "Each line must be a JSON object with 'data' (to be flattened) and 'meta' (stored as-is)."
        ),
        formatter_class=_SmartHelp,
    )
    p_build.add_argument("jsonl", help="Path to the input JSONL file.")
    p_build.set_defaults(cmd="build")

    # search
    p_search = subparsers.add_parser(
        "search",
        help="Search the index.",
        description=(
            "Search the index. Supports quoted phrases and boolean operators AND/OR/NOT.\n"
            'Example: "\\"aspirin side effects\\" OR ibuprofen"'
        ),
        formatter_class=_SmartHelp,
    )

    p_search.add_argument(
        "query",
        nargs="?",
        help="Query string. If omitted, starts an interactive prompt.",
    )
    p_search.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return."
    )

    args = parser.parse_args()
    if args.command == "search":
        ts = perf_counter()
        if not os.path.exists(args.index):
            raise SystemExit(f"Index not found: {args.index}")
        idx = BaseInvertedIndex.load(args.index)
        print(f"# build time: {(perf_counter() - ts)* 1_000:.3f} ms")

        # one-off query provided
        if args.query is not None:
            t0 = perf_counter()
            hits = idx.search(args.query, top_k=args.top_k)
            t1 = perf_counter()
            _print_hits(idx, hits, t1 - t0, args.query)
            return 0


        # interactive REPL
        try:
            while True:
                try:
                    q = input("query> ").strip()
                except EOFError:
                    print()  # newline on Ctrl-D
                    break
                if not q:
                    continue
                if q.lower() in (":q", ":quit", "quit", "exit"):
                    break
                t0 = perf_counter()
                hits = idx.search(q, top_k=args.top_k)
                t1 = perf_counter()
                _print_hits(idx, hits, t1 - t0, q)
        except KeyboardInterrupt:
            print()  # newline on Ctrl-C
        return 0
    if args.command == "build":
        ts = perf_counter()
        if os.path.exists(args.index):
            idx = BaseInvertedIndex.load(args.index)
        else:
            subcls: Optional[Type[BaseInvertedIndex]] = _INDEX_REGISTRY.get(
                args.index_class
            )
            if subcls is None:
                raise SystemExit(
                    f"Unknown index class '{args.index_class}'. Available: {list(_INDEX_REGISTRY)}"
                )
            idx = subcls()
            
        added = 0
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"[build] skip line {lineno}: JSON error: {e}", file=sys.stderr
                    )
                    continue

                data = rec.get("data")
                meta = rec.get("meta", {}) or {}
                if data is None:
                    print(
                        f"[build] skip line {lineno}: no 'data' field", file=sys.stderr
                    )
                    continue

                text = idx._flatten_text(data)
                if not text:
                    # skip truly empty after flattening
                    continue

                doc_id = meta["canonical_url"]
                idx.add_document(doc_id, text, meta)
                added += 1

        idx.finalize()
        os.makedirs(os.path.dirname(args.index) or ".", exist_ok=True)
        idx.save(args.index)
        print(f"Built {added} docs into {args.index}")
        print(f"# build time: {(perf_counter() - ts)* 1_000:.3f} ms")
        return 0

    # Should not get here
    parser.print_help()
    return 2


if __name__ == "__main__":
    main()
