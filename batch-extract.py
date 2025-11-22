#!/usr/bin/env python3
"""
IGSOA Batch Library Compiler

What it does (single run):
- Recursively scans a source repo for .md files.
- For each file:
  * Extracts nested sections/subsections + logical units + sealed boxes
  * Saves every extracted section VERBATIM
  * Writes a per-section REPORT (fingerprint + concepts + local claims)
  * Computes a per-file fingerprint (sha256 + mtime + concept/structure signatures)
- Groups processed outputs by TOPIC into root-level topic folders.
- Keeps ALL versions (even mutually exclusive ones).
- Flags:
  * Strict conceptual contradictions (mutually exclusive claims)
  * Divergent variants (nuanced disagreements / different formulations)
- Builds global indexes + cross-references:
  * 00_MASTER_INDEX.md
  * 00_TOPIC_INDEX.md
  * 00_CONTRADICTION_REPORT.md
  * 00_AXIOM_REGISTRY.md
  * 00_THEOREM_REGISTRY.md
  * 00_CONCEPT_INDEX.md
  * 00_VERSION_MAP.md
  * 00_CROSS_REFERENCE_GRAPH.md
  * library_index.json

Usage:
  python igsoa_batch_pipeline.py <source_root> <output_root> [--concepts concepts.txt]

Notes:
- Heuristic contradiction detection (conceptual + nuanced).
- You can tune TOPIC_KEYWORDS and CLAIM_PATTERNS below anytime.
"""

import re, json, sys, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict

# -----------------------
# Concepts (default list)
# -----------------------
DEFAULT_CONCEPTS = [
    "IGSOA", "Meta-Genesis", "Meta-Math", "Deviation", "δ", "Phi", "Φ", "Pi", "Π",
    "Adjacency", "Curvature", "Weitzenböck", "Spectral Triple", "Functor", "Functoriality",
    "CMB", "GW", "Echo", "Lensing", "Inflation", "Mass Tower", "δ-Higgs",
    "Causal Sphere of Influence", "DFVM", "QMM",
    "Tri-Unity", "Meta-Einstein", "NOT Axiom", "δ-curvature", "δ-harmonics"
]

def load_concepts(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_CONCEPTS
    p = Path(path)
    if not p.exists():
        print(f"[warn] concepts file not found: {p}, using defaults")
        return DEFAULT_CONCEPTS
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

# -----------------------
# Topic grouping (tunable)
# -----------------------
TOPIC_KEYWORDS = {
    "cosmology": ["CMB", "Echo", "GW", "lensing", "inflation", "matter power", "cold spot", "multipole"],
    "delta_geometry": ["δ-curvature", "adjacency", "Weitzenböck", "torsion", "spectral triple", "metric emergence"],
    "phi_harmonics": ["δ→Φ", "Φ-ladder", "harmonic", "spectrum", "eigen", "resonance"],
    "pi_projection": ["Π", "projection", "geometry", "multipole", "angular", "metric"],
    "meta_genesis": ["Meta-Genesis", "NOT Axiom", "Genesis", "Ψ₀", "Ψ₁", "monism"],
    "meta_math": ["category", "functor", "topos", "adjoint", "C*-algebra", "hyperdoctrine"],
    "forces_particles": ["δ-Higgs", "mass tower", "SU(3)", "SU(2)", "U(1)", "CKM", "PMNS", "strong force", "weak force"],
}

def infer_topic(md_rel: Path, text: str, concepts: List[str]) -> str:
    parts = md_rel.parts
    if len(parts) >= 2:
        top = parts[0].lower()
        if top in TOPIC_KEYWORDS:
            return top

    lower = text.lower()
    scores = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        s = 0
        for kw in kws:
            if kw.lower() in lower:
                s += 2
        for c in concepts:
            if c.lower() in lower and c in kws:
                s += 1
        scores[topic] = s

    best_topic, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_topic if best_score > 0 else "other"

# -----------------------
# Heading + logical unit detection
# -----------------------
HEADING_RULES: List[Tuple[int, str]] = [
    (1, r'^\s*(\d+)\.\s+(.+?)\s*$'),                   # 1. Title
    (2, r'^\s*(\d+)\.(\d+)\s+(.+?)\s*$'),              # 1.1 Title
    (3, r'^\s*(\d+)\.(\d+)\.(\d+)\s+(.+?)\s*$'),       # 1.1.1 Title
    (1, r'^\s*Section\s+(\d+)\s*[:\.]\s+(.+?)\s*$'),   # Section 4: Title
    (1, r'^\s*#\s+(.+?)\s*$'),                         # # Title
    (2, r'^\s*##\s+(.+?)\s*$'),                       # ## Title
    (3, r'^\s*###\s+(.+?)\s*$'),                      # ### Title
    (4, r'^\s*####\s+(.+?)\s*$'),                      # #### Title
    (2, r'^⟡\s*SEALED AXIOM BOX\s*—\s*(.+?)\s*⟡\s*$'),  # Sealed box title
]

LOGICAL_UNIT_RULES = [
    r'^\s*\*\*(Theorem.*?)\*\*\s*$',
    r'^\s*\*\*(Lemma.*?)\*\*\s*$',
    r'^\s*\*\*(Definition.*?)\*\*\s*$',
    r'^\s*\*\*(Corollary.*?)\*\*\s*$',
    r'^\s*\*\*(Proposition.*?)\*\*\s*$',
    r'^\s*\*\*(Axiom.*?)\*\*\s*$',
    r'^\s*(Theorem\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^\s*(Lemma\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^\s*(Definition\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^\s*(Corollary\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^\s*(Proposition\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^\s*(Axiom\s*\(?.*?\)?)\s*[:\.]?\s*$',
    r'^⟡\s*(Theorem.*?)\s*⟡$',
    r'^⟡\s*(Lemma.*?)\s*⟡$',
    r'^⟡\s*(Definition.*?)\s*⟡$',
    r'^⟡\s*(Corollary.*?)\s*⟡$',
    r'^⟡\s*(Proposition.*?)\s*⟡$',
    r'^⟡\s*(Axiom.*?)\s*⟡$',
]

COMPILED_RULES: List[Tuple[int, re.Pattern]] = [(lvl, re.compile(rx, re.MULTILINE))
                                                for lvl, rx in HEADING_RULES]
for rx in LOGICAL_UNIT_RULES:
    COMPILED_RULES.append((3, re.compile(rx, re.MULTILINE)))

TOP_HEADING_RE = re.compile(
    r'^\s*(\d+\.\s+.+|Section\s+\d+[:\.]\s+.+|#\s+.+)\s*$',
    re.MULTILINE
)
SEALED_BOX_RE = re.compile(
    r'^⟡\s*SEALED AXIOM BOX\s*—\s*(.+?)\s*⟡\s*$',
    re.MULTILINE
)

INLINE_AXIOM_RE = re.compile(r'^\s*(Axiom\s+[A-Z]?\d+.*?)[\:\.]\s*$',
                             re.MULTILINE)

THEOREM_RE = re.compile(r'^\s*(Theorem\s*(\(.+?\))?.*?)[\:\.]\s*$',
                        re.MULTILINE)
LEMMA_RE = re.compile(r'^\s*(Lemma\s*(\(.+?\))?.*?)[\:\.]\s*$',
                      re.MULTILINE)
DEF_RE = re.compile(r'^\s*(Definition\s*(\(.+?\))?.*?)[\:\.]\s*$',
                    re.MULTILINE)
COR_RE = re.compile(r'^\s*(Corollary\s*(\(.+?\))?.*?)[\:\.]\s*$',
                    re.MULTILINE)
PROP_RE = re.compile(r'^\s*(Proposition\s*(\(.+?\))?.*?)[\:\.]\s*$',
                     re.MULTILINE)

@dataclass
class Node:
    level: int
    title: str
    start: int
    end: int = -1
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None

def detect_headings(text: str) -> List[Tuple[int, int, str]]:
    hits = []
    for lvl, cregex in COMPILED_RULES:
        for m in cregex.finditer(text):
            title = m.groups()[-1].strip()
            hits.append((m.start(), lvl, title))
    hits.sort(key=lambda x: x[0])
    return hits

def build_tree(text: str) -> List[Node]:
    hits = detect_headings(text)
    if not hits:
        return []
    nodes: List[Node] = []
    stack: List[Node] = []
    for idx, lvl, title in hits:
        node = Node(level=lvl, title=title, start=idx)
        while stack and stack[-1].level >= lvl:
            stack[-1].end = idx
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            nodes.append(node)
        stack.append(node)
    end_pos = len(text)
    while stack:
        stack[-1].end = end_pos
        stack.pop()
    return nodes

PATH_LIMIT = 230  # conservative limit to avoid Windows MAX_PATH failures

def sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    name = name.strip("._") or "untitled"
    if len(name) <= max_len:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    keep = max(4, max_len - len(digest) - 1)
    return f"{name[:keep]}_{digest}"

def safe_component(base: Path, label: str, ordinal: Optional[int] = None, path_limit: int = PATH_LIMIT) -> str:
    """Create a filesystem-safe component while respecting a global path limit.

    The resulting name fits within `path_limit` once joined to `base`, avoiding
    Windows MAX_PATH errors by truncating with a stable hash when necessary.
    The `base` path is resolved to avoid under-counting when the caller passes
    a relative path.
    """
    prefix = f"{ordinal:02d}_" if ordinal is not None else ""
    base_len = len(str(base.resolve()))
    available = path_limit - base_len - 1  # leave room for separator
    available = max(1, available)
    body_limit = max(1, available - len(prefix))
    body = sanitize(label, max_len=body_limit)
    return prefix + body[:body_limit]

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_headings(text: str) -> List[str]:
    return [m.group(1).strip() for m in TOP_HEADING_RE.finditer(text)]

def extract_sealed_boxes(text: str) -> List[str]:
    return [m.group(1).strip() for m in SEALED_BOX_RE.finditer(text)]

def extract_inline_axioms(text: str) -> List[str]:
    return [m.group(1).strip() for m in INLINE_AXIOM_RE.finditer(text)]

def extract_math_units(text: str) -> List[Tuple[str, str]]:
    units = []
    for rx, kind in [
        (THEOREM_RE, "Theorem"),
        (LEMMA_RE, "Lemma"),
        (DEF_RE, "Definition"),
        (COR_RE, "Corollary"),
        (PROP_RE, "Proposition"),
    ]:
        for m in rx.finditer(text):
            units.append((kind, m.group(1).strip()))
    return units

def concept_hits(text: str, concepts: List[str]) -> List[str]:
    lower = text.lower()
    hits = []
    for c in concepts:
        if c.lower() in lower:
            hits.append(c)
    return sorted(set(hits))

# Claim extraction for contradictions (heuristic)
CLAIM_PATTERNS = [
    r'(?P<concept>δ-curvature|δ|Phi|Φ|Pi|Π|IGSOA|Meta-Genesis|Adjacency|Curvature|Weitzenböck)'
    r'\s+(?P<neg>is not|isnt|isn\'t|are not|aren\'t|has no|does not|doesn\'t|cannot|can\'t|never|without)?\s*'
    r'(?P<pred>[^\.!\n]{0,120})',
]
NEG_WORDS = {"is not","isnt","isn't","are not","aren't","has no","does not","doesn't","cannot","can't","never","without","no"}

def normalize_pred(pred: str) -> str:
    pred = pred.lower()
    pred = re.sub(r'[\[\]{}()<>,"“”\'`]', ' ', pred)
    pred = re.sub(r'\s+', ' ', pred).strip()
    return " ".join(pred.split()[:8])

def extract_claims(text: str) -> List[Dict[str, Any]]:
    claims = []
    for pat in CLAIM_PATTERNS:
        rx = re.compile(pat, re.IGNORECASE)
        for m in rx.finditer(text):
            concept = m.group("concept")
            neg = (m.group("neg") or "").lower().strip()
            pred = normalize_pred(m.group("pred") or "")
            if not pred:
                continue
            polarity = "neg" if neg in NEG_WORDS and neg else "pos"
            claims.append({
                "concept": concept,
                "predicate": pred,
                "polarity": polarity,
                "snippet": m.group(0).strip()[:180]
            })
    seen, out = set(), []
    for c in claims:
        key = (c["concept"].lower(), c["predicate"], c["polarity"])
        if key not in seen:
            seen.add(key); out.append(c)
    return out

# Divergent variants
WORD_RE = re.compile(r"[A-Za-zΑ-ωδΦΠΨ0-9_→\-]+", re.UNICODE)
def token_set(s: str) -> set:
    return set(w.lower() for w in WORD_RE.findall(s))
def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0

def file_fingerprint(md: Path, concepts: List[str], claims: List[Dict[str, Any]]) -> Dict:
    text = md.read_text(encoding="utf-8", errors="ignore")
    h = sha256_text(text)
    mtime = datetime.fromtimestamp(md.stat().st_mtime).isoformat()
    csig = concept_hits(text, concepts)
    ssig = {
        "headings": len(extract_headings(text)),
        "sealed_boxes": len(extract_sealed_boxes(text)),
        "inline_axioms": len(extract_inline_axioms(text)),
        "math_units": len(extract_math_units(text)),
        "claims": len(claims),
        "chars": len(text),
        "lines": text.count("\n") + 1,
    }
    return {"sha256": h, "mtime_iso": mtime, "concept_signature": csig, "structural_signature": ssig}

def write_node(node: Node, text: str, outdir: Path, ordinal: int, file_meta: Dict, path_limit: int) -> str:
    folder_name = safe_component(outdir, node.title, ordinal, path_limit=path_limit)
    node_dir = outdir / folder_name
    node_dir.mkdir(parents=True, exist_ok=True)

    content = text[node.start:node.end].strip()
    (node_dir / "00_SECTION.md").write_text(content, encoding="utf-8")

    sect_concepts = concept_hits(content, file_meta["concepts_all"])
    sect_claims = extract_claims(content)
    sect_fp = {
        "sha256": sha256_text(content),
        "concepts": sect_concepts,
        "claims": sect_claims,
        "parent_file": file_meta["source_rel"],
        "file_sha256": file_meta["fingerprint"]["sha256"],
        "topic": file_meta["topic"],
        "title": node.title,
        "level": node.level,
        "start": node.start,
        "end": node.end,
    }
    (node_dir / "00_REPORT.json").write_text(json.dumps(sect_fp, indent=2), encoding="utf-8")

    child_info = []
    for i, child in enumerate(node.children, 1):
        child_folder = write_node(child, text, node_dir, i, file_meta, path_limit)
        child_info.append((i, child.title, child_folder))

    index_lines = [
        f"# Index: {node.title}",
        "",
        f"**Level:** {node.level}",
        f"**Children:** {len(node.children)}",
        f"**Parent file:** {file_meta['source_rel']}",
        "",
        "---",
        "",
        "## Contents",
        "",
        "- **00_SECTION.md** — verbatim text of this section",
        "- **00_REPORT.json** — extraction report for this section",
    ]
    for i, title, folder in child_info:
        index_lines.append(f"- **{folder}/** — {title}")
    (node_dir / "00_INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    return folder_name

def extract_to_tree(md_path: Path, outdir: Path, topic: str, concepts: List[str], source_rel: str, path_limit: int):
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    nodes = build_tree(text) or [Node(level=1, title=md_path.stem, start=0, end=len(text))]

    claims = extract_claims(text)
    fingerprint = file_fingerprint(md_path, concepts, claims)

    file_meta = {"source_rel": source_rel, "topic": topic, "concepts_all": concepts, "fingerprint": fingerprint}

    outdir.mkdir(parents=True, exist_ok=True)
    top_sections = []
    for i, node in enumerate(nodes, 1):
        folder = write_node(node, text, outdir, i, file_meta, path_limit)
        top_sections.append((i, node.title, folder))

    lines = [
        "# File Root Index",
        "",
        f"**Source file:** {source_rel}",
        f"**Topic:** {topic}",
        f"**sha256:** `{fingerprint['sha256']}`",
        f"**mtime:** {fingerprint['mtime_iso']}",
        "",
        "## Top-Level Sections",
        "",
    ]
    for i, n_title, folder in top_sections:
        lines.append(f"{i}. **{n_title}** → `{folder}/`")
    (outdir / "00_FILE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "source_rel": source_rel,
        "topic": topic,
        "out_rel": str(outdir),
        "fingerprint": fingerprint,
        "concepts": fingerprint["concept_signature"],
        "claims": claims,
        "sealed_boxes": extract_sealed_boxes(text),
        "inline_axioms": extract_inline_axioms(text),
        "math_units": [{"kind": k, "title": t} for k, t in extract_math_units(text)],
        "section_titles": [n.title for n in nodes],
        "raw_text": text,  # truncated later for JSON size
    }

def analyze_contradictions(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    claim_index = defaultdict(list)
    for e in entries:
        for c in e["claims"]:
            key = (c["concept"].lower(), c["predicate"])
            claim_index[key].append((c["polarity"], e, c))

    contradictions = []
    for (concept, pred), items in claim_index.items():
        pols = {p for p, _e, _c in items}
        if "pos" in pols and "neg" in pols:
            contradictions.append({
                "concept": concept,
                "predicate": pred,
                "instances": [
                    {
                        "polarity": p,
                        "file": inst["source_rel"],
                        "topic": inst["topic"],
                        "sha256": inst["fingerprint"]["sha256"],
                        "snippet": cobj["snippet"],
                    }
                    for p, inst, cobj in items
                ]
            })

    by_title = defaultdict(list)
    for e in entries:
        for t in e["section_titles"]:
            by_title[t.lower()].append(e)

    variants = []
    for title, insts in by_title.items():
        if len(insts) < 2:
            continue
        for i in range(len(insts)):
            for j in range(i+1, len(insts)):
                a, b = insts[i], insts[j]
                sim = jaccard(token_set(a["raw_text"]), token_set(b["raw_text"]))
                if sim < 0.55:
                    variants.append({
                        "section_title": title,
                        "similarity": round(sim, 3),
                        "a_file": a["source_rel"],
                        "b_file": b["source_rel"],
                        "note": "Nuanced or model-variant disagreement (flagged, not discarded)."
                    })
    return contradictions, variants

def write_master_index(entries, outroot: Path):
    lines = [
        "# IGSOA Master Library Index",
        "",
        f"**Total source papers:** {len(entries)}",
        "",
        "---",
        "",
        "## Papers by Topic",
        ""
    ]
    topics = defaultdict(list)
    for e in entries:
        topics[e["topic"]].append(e)

    for topic in sorted(topics.keys()):
        lines.append(f"### topic_{topic}")
        for e in topics[topic]:
            lines.append(f"- **{Path(e['source_rel']).stem}**")
            lines.append(f"  - source: `{e['source_rel']}`")
            lines.append(f"  - sha256: `{e['fingerprint']['sha256'][:16]}…`")
            if e["concepts"]:
                lines.append(f"  - concepts: {', '.join(e['concepts'])}")
        lines.append("")
    (outroot / "00_MASTER_INDEX.md").write_text("\n".join(lines), encoding="utf-8")

def write_topic_index(entries, outroot: Path):
    topics = defaultdict(list)
    for e in entries:
        topics[e["topic"]].append(e)
    lines = ["# Topic Index", ""]
    for topic in sorted(topics.keys()):
        lines.append(f"## topic_{topic}")
        for e in topics[topic]:
            lines.append(f"- `{e['source_rel']}` → `{e['out_rel']}`")
        lines.append("")
    (outroot / "00_TOPIC_INDEX.md").write_text("\n".join(lines), encoding="utf-8")

def write_axiom_registry(entries, outroot: Path):
    reg = []
    for e in entries:
        for s in e["sealed_boxes"]:
            reg.append({"kind": "Sealed Axiom Box", "title": s, "file": e["source_rel"], "topic": e["topic"]})
        for a in e["inline_axioms"]:
            reg.append({"kind": "Inline Axiom", "title": a, "file": e["source_rel"], "topic": e["topic"]})
    lines = ["# Axiom Registry", "", f"**Total:** {len(reg)}", "", "---", ""]
    for i, r in enumerate(reg, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   - kind: {r['kind']}")
        lines.append(f"   - topic: {r['topic']}")
        lines.append(f"   - file: `{r['file']}`")
        lines.append("")
    (outroot / "00_AXIOM_REGISTRY.md").write_text("\n".join(lines), encoding="utf-8")
    return reg

def write_theorem_registry(entries, outroot: Path):
    reg = []
    for e in entries:
        for mu in e["math_units"]:
            reg.append({**mu, "file": e["source_rel"], "topic": e["topic"]})
    lines = ["# Theorem/Lemma/Definition Registry", "", f"**Total:** {len(reg)}", "", "---", ""]
    for i, r in enumerate(reg, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   - kind: {r['kind']}")
        lines.append(f"   - topic: {r['topic']}")
        lines.append(f"   - file: `{r['file']}`")
        lines.append("")
    (outroot / "00_THEOREM_REGISTRY.md").write_text("\n".join(lines), encoding="utf-8")
    return reg

def write_concept_index(entries, outroot: Path):
    cindex = defaultdict(list)
    for e in entries:
        for c in e["concepts"]:
            cindex[c].append(e["source_rel"])
    lines = ["# Concept Index", "", "---", ""]
    for c in sorted(cindex.keys()):
        lines.append(f"## {c}")
        for f in sorted(set(cindex[c])):
            lines.append(f"- `{f}`")
        lines.append("")
    (outroot / "00_CONCEPT_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    return cindex

def write_version_map(entries, outroot: Path):
    by_base = defaultdict(list)
    for e in entries:
        base = Path(e["source_rel"]).stem.lower()
        by_base[base].append(e)
    lines = ["# Version Map", "", "---", ""]
    for base, insts in sorted(by_base.items()):
        if len(insts) == 1:
            continue
        lines.append(f"## {base}")
        for e in insts:
            fp = e["fingerprint"]
            lines.append(f"- `{e['source_rel']}`")
            lines.append(f"  - sha256: `{fp['sha256']}`")
            lines.append(f"  - mtime: {fp['mtime_iso']}")
        lines.append("")
    (outroot / "00_VERSION_MAP.md").write_text("\n".join(lines), encoding="utf-8")

def write_cross_reference_graph(entries, outroot: Path, ax_reg, c_index):
    lines = ["# Cross-Reference Graph (Adjacency List)", "", "---", ""]
    lines.append("## Shared Concepts")
    for c, files in sorted(c_index.items()):
        if len(set(files)) >= 2:
            lines.append(f"- **{c}**:")
            for f in sorted(set(files)):
                lines.append(f"  - `{f}`")
    lines.append("")
    lines.append("## Shared Axioms (by title)")
    ax_by_title = defaultdict(list)
    for a in ax_reg:
        ax_by_title[a["title"].lower()].append(a["file"])
    for t, files in sorted(ax_by_title.items()):
        if len(set(files)) >= 2:
            lines.append(f"- **{t}**:")
            for f in sorted(set(files)):
                lines.append(f"  - `{f}`")
    (outroot / "00_CROSS_REFERENCE_GRAPH.md").write_text("\n".join(lines), encoding="utf-8")

def write_contradiction_report(contradictions, variants, outroot: Path):
    lines = ["# Contradiction Report", "", "---", ""]
    lines.append(f"## Strict Conceptual Contradictions ({len(contradictions)})")
    for i, c in enumerate(contradictions, 1):
        lines.append(f"{i}. **{c['concept']} :: {c['predicate']}**")
        for inst in c["instances"]:
            lines.append(f"   - ({inst['polarity']}) `{inst['file']}` [{inst['topic']}]")
            lines.append(f"     - snippet: {inst['snippet']}")
        lines.append("")
    lines.append(f"## Divergent Variants / Nuanced Disagreements ({len(variants)})")
    for i, v in enumerate(variants, 1):
        lines.append(f"{i}. **{v['section_title']}** (similarity {v['similarity']})")
        lines.append(f"   - A: `{v['a_file']}`")
        lines.append(f"   - B: `{v['b_file']}`")
        lines.append(f"   - note: {v['note']}")
        lines.append("")
    (outroot / "00_CONTRADICTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

def write_library_json(entries, contradictions, variants, outroot: Path):
    safe_entries = []
    for e in entries:
        ee = dict(e)
        ee["raw_text"] = ee["raw_text"][:50000]
        safe_entries.append(ee)
    payload = {
        "entries": safe_entries,
        "contradictions": contradictions,
        "variants": variants,
        "generated_at": datetime.now().isoformat(),
    }
    (outroot / "library_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="IGSOA batch library compiler")
    ap.add_argument("source_root", help="Repo/source root containing .md files")
    ap.add_argument("output_root", help="Root folder to write topic-grouped library")
    ap.add_argument("--concepts", help="Optional concepts.txt (one concept per line)")
    ap.add_argument("--path-limit", type=int, default=PATH_LIMIT,
                    help=f"Maximum path length budget for generated folders (default {PATH_LIMIT})")
    args = ap.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    concepts = load_concepts(args.concepts)
    path_limit = args.path_limit

    if not source_root.exists():
        print(f"[error] source root not found: {source_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    md_files = sorted(source_root.rglob("*.md"))
    if not md_files:
        print("[warn] no markdown files found")
        sys.exit(0)

    entries = []
    for md in md_files:
        rel = str(md.relative_to(source_root)).replace("\\", "/")
        text = md.read_text(encoding="utf-8", errors="ignore")
        topic = infer_topic(Path(rel), text, concepts)

        topic_dir_name = safe_component(output_root, f"topic_{topic}", path_limit=path_limit)
        topic_outdir = output_root / topic_dir_name
        paper_dir_name = safe_component(topic_outdir, md.stem, path_limit=path_limit)
        paper_outdir = topic_outdir / paper_dir_name

        entry = extract_to_tree(md, paper_outdir, topic, concepts, rel, path_limit)
        entries.append(entry)
        print(f"✓ processed {rel} → {topic_dir_name}/{paper_dir_name}")

    contradictions, variants = analyze_contradictions(entries)

    ax_reg = write_axiom_registry(entries, output_root)
    _th_reg = write_theorem_registry(entries, output_root)
    c_index = write_concept_index(entries, output_root)

    write_master_index(entries, output_root)
    write_topic_index(entries, output_root)
    write_version_map(entries, output_root)
    write_cross_reference_graph(entries, output_root, ax_reg, c_index)
    write_contradiction_report(contradictions, variants, output_root)
    write_library_json(entries, contradictions, variants, output_root)

    print("\n✓ Batch library build complete.")
    print(f"Output: {output_root}")

if __name__ == "__main__":
    main()
