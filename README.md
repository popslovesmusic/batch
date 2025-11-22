# IGSOA Batch Library Compiler

This repository contains `batch-extract.py`, a command-line tool that scans a source directory of Markdown papers and produces a structured library grouped by inferred topics. The script extracts sections, mathematical statements, axioms, and simple claims, then builds per-section reports and global indexes.

## Key capabilities
- Recursively processes all `.md` files under a source root.
- Infers a topic for each paper using configurable keywords and concept lists.
- Builds a tree of sections/headings and writes each section with a report.
- Detects sealed axiom boxes, inline axioms, theorems/lemmas/definitions, and simple claim statements for contradiction analysis.
- Generates topic folders, paper indexes, master/topic indexes, axiom/theorem registries, contradiction reports, and a `library_index.json` summary.

## Usage
Run the compiler against a Markdown source directory and an output directory:

```bash
python batch-extract.py <source_root> <output_root> [--concepts concepts.txt] [--path-limit 230]
```

The optional `--concepts` flag points to a file with one concept per line; otherwise, a built-in list is used. Use `--path-limit`
to adjust the maximum generated path length (defaults to 230 characters, tuned for Windows).

## Output overview
The output root is organized into `topic_<name>/` folders, each containing processed papers. Each paper folder includes per-section subdirectories with `00_SECTION.md`, `00_REPORT.json`, and an index. Root-level summary files (e.g., `00_MASTER_INDEX.md`, `00_TOPIC_INDEX.md`, `00_AXIOM_REGISTRY.md`) provide cross-references and contradiction reports.
