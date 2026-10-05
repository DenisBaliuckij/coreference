"""Summarize experiment runs: tables with bootstrap confidence intervals and text diagnostics.

    python -m pipeline.analysis.summarize RUN_DIR [RUN_DIR ...] --out OUT_DIR [--bootstrap 1000]

Per run (an --output directory of run_experiment, ideally with --save-artifacts):
- coreference: the pairing's corpus-level CorefUD scores, re-scored under every match mode
  from coref/key.conllu and coref/<resolver>.response.conllu, with a document-level bootstrap
  95% CI of CoNLL F1 (documents resampled with replacement, each resample scored as a corpus);
- graph: mean node/edge/triple F1 per resolver x backend with bootstrap 95% CIs over
  documents; paired bootstrap differences against NoResolution (CI and one-sided p = share of
  resamples with difference <= 0); "gap closed" = (R - NoResolution) / (Gold - NoResolution),
  the share of the coreference error budget a resolver recovers relative to the Gold control;
- texts (artifacts/texts): pronoun density per 1000 words, length ratio and character-level
  similarity of each resolver's text to the oracle and to the original text;
- graphs (artifacts/graphs): mean node and edge counts and the share of pronoun-labelled nodes;
- corpus: documents, tokens, mentions, entities, non-singleton entities, pronominal mentions.
Writes summary.json and summary.md to OUT_DIR.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import statistics
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..corpus.corefud_loader import parse_conllu
from ..corpus.pronouns import PRONOUNS
from ..eval.corefud_scoring import MATCH_MODES, score_corpus, scorer_available
from ..eval.graph_scoring import compute_graph_scores

WORD = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------------- bootstrap
def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0, stat=statistics.fmean) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(values)
    samples = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return samples[int(0.025 * n)], samples[min(n - 1, int(0.975 * n))]


def paired_bootstrap(a: list[float], b: list[float], n: int = 1000, seed: int = 0) -> dict:
    """Mean of a - b over paired documents, its 95% CI, and one-sided p = P(diff <= 0)."""
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(statistics.fmean([diffs[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return {"mean_diff": statistics.fmean(diffs), "ci95": [means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]],
            "p_one_sided": sum(1 for m in means if m <= 0) / n}


# ------------------------------------------------------------------------- coreference scores
def _split_docs(text: str) -> tuple[list[str], dict[str, str]]:
    """(header lines before the first '# newdoc', {doc_id: block text})."""
    header, docs, current = [], {}, None
    for line in text.splitlines():
        if line.startswith("# newdoc"):
            m = re.search(r"id\s*=\s*(\S+)", line)
            current = m.group(1) if m else f"doc{len(docs) + 1}"
            docs[current] = [line]
        elif current is None:
            header.append(line)
        else:
            docs[current].append(line)
    return header, {k: "\n".join(v) for k, v in docs.items()}


def _renamed(block: str, doc_id: str, new_id: str) -> str:
    out = []
    for line in block.splitlines():
        if line.startswith("# newdoc"):
            line = f"# newdoc id = {new_id}"
        elif line.startswith("# sent_id"):
            line = line.replace(doc_id, new_id, 1) if doc_id in line else f"{line}-{new_id}"
        out.append(line)
    return "\n".join(out)


def _join_blocks(header: list[str], blocks: list[str]) -> str:
    """Document blocks back into one CoNLL-U file: every block ends with its blank line, else
    the last sentence of one document runs into the first of the next (udapi: 'cycle')."""
    head = "\n".join(header).strip("\n")
    body = "\n\n".join(b.strip("\n") for b in blocks)
    return (head + "\n" if head else "") + body + "\n\n"


def coref_bootstrap(key_path: Path, response_path: Path, match: str, n: int, seed: int = 0,
                    workers: int = 8) -> tuple[float, float]:
    """95% CI of corpus-level CoNLL F1 over documents resampled with replacement; duplicates
    are renamed so the scorer treats them as separate documents. The resamples are drawn up
    front from one seeded generator, so the CI does not depend on ``workers`` (scorer runs are
    subprocesses, scored in parallel threads)."""
    key_header, key_docs = _split_docs(key_path.read_text(encoding="utf-8"))
    resp_header, resp_docs = _split_docs(response_path.read_text(encoding="utf-8"))
    ids = list(key_docs)
    rng = random.Random(seed)
    picks = [[ids[rng.randrange(len(ids))] for _ in ids] for _ in range(n)]

    def score(i: int) -> float:
        k_parts, r_parts = [], []
        for j, d in enumerate(picks[i]):
            new = f"{d}__b{j}"
            k_parts.append(_renamed(key_docs[d], d, new))
            r_parts.append(_renamed(resp_docs[d], d, new))
        with tempfile.TemporaryDirectory() as tmp:
            k_file, r_file = Path(tmp) / "key.conllu", Path(tmp) / "resp.conllu"
            k_file.write_text(_join_blocks(key_header, k_parts), encoding="utf-8")
            r_file.write_text(_join_blocks(resp_header, r_parts), encoding="utf-8")
            return score_corpus(k_file, r_file, match=match)["conll_f1"]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scores = sorted(pool.map(score, range(n)))
    return scores[int(0.025 * n)], scores[min(n - 1, int(0.975 * n))]


# -------------------------------------------------------------------------- text diagnostics
def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text)]


def text_diagnostics(art: Path, language: str) -> dict:
    texts_dir = art / "texts"
    if not texts_dir.is_dir():
        return {}
    pron = PRONOUNS.get(language, set())
    oracle = {p.stem: p.read_text(encoding="utf-8") for p in (texts_dir / "ORACLE").glob("*.txt")}
    original = {p.stem: p.read_text(encoding="utf-8") for p in (texts_dir / "NoResolution").glob("*.txt")}
    out = {}
    for sub in sorted(p for p in texts_dir.iterdir() if p.is_dir()):
        dens, len_ratio, sim_oracle, sim_original = [], [], [], []
        for p in sub.glob("*.txt"):
            text = p.read_text(encoding="utf-8")
            words = _words(text)
            dens.append(1000 * sum(w in pron for w in words) / max(1, len(words)))
            if p.stem in original:
                len_ratio.append(len(text) / max(1, len(original[p.stem])))
                sim_original.append(difflib.SequenceMatcher(None, text, original[p.stem], autojunk=False).ratio())
            if p.stem in oracle:
                sim_oracle.append(difflib.SequenceMatcher(None, text, oracle[p.stem], autojunk=False).ratio())
        out[sub.name] = {
            "pronouns_per_1000_words": statistics.fmean(dens) if dens else None,
            "length_ratio_to_original": statistics.fmean(len_ratio) if len_ratio else None,
            "char_similarity_to_oracle": statistics.fmean(sim_oracle) if sim_oracle else None,
            "char_similarity_to_original": statistics.fmean(sim_original) if sim_original else None,
            "documents": len(dens),
        }
    return out


def _node_labels(g: dict) -> list[str]:
    return [str(n["label"] if isinstance(n, dict) else n) for n in g.get("nodes") or []]


def graph_sizes(art: Path, language: str) -> dict:
    """Mean node/edge counts and the share of nodes whose whole label is a pronoun.

    Both backends merge nodes by label, so the node duplication rate is structurally 0 and an
    unresolved anaphor shows up instead as a node labelled "he", "it", "она"...: the pronoun
    node share is the graph-side trace of coreference left unresolved."""
    out = {}
    graphs_dir = art / "graphs"
    if not graphs_dir.is_dir():
        return out
    pron = PRONOUNS.get(language, set())
    for backend in sorted(p for p in graphs_dir.iterdir() if p.is_dir()):
        for sub in sorted(p for p in backend.iterdir() if p.is_dir()):
            nodes, edges, pron_nodes = [], [], []
            for p in sub.glob("*.json"):
                g = json.loads(p.read_text(encoding="utf-8"))
                labels = _node_labels(g)
                nodes.append(len(labels))
                edges.append(len(g.get("edges") or []))
                pron_nodes.append(sum(1 for lab in labels if lab.strip().lower() in pron))
            if nodes:
                out[f"{backend.name}/{sub.name}"] = {
                    "mean_nodes": statistics.fmean(nodes), "mean_edges": statistics.fmean(edges),
                    "pronoun_node_share": sum(pron_nodes) / max(1, sum(nodes)), "documents": len(nodes)}
    return out


def corpus_stats(key_path: Path, language: str) -> dict:
    docs = parse_conllu(key_path.read_text(encoding="utf-8"))
    pron = PRONOUNS.get(language, set())
    mentions = [m for d in docs for c in d.clusters for m in c]
    pronominal = 0
    for d in docs:
        for c in d.clusters:
            for m in c:
                words = [t.text.lower() for t in d.tokens[m.start_token: m.end_token + 1] if t.text]
                pronominal += len(words) == 1 and words[0] in pron
    return {
        "documents": len(docs),
        "tokens": sum(len([t for t in d.tokens if not t.is_empty]) for d in docs),
        "mentions": len(mentions),
        "entities": sum(len(d.clusters) for d in docs),
        "non_singleton_entities": sum(1 for d in docs for c in d.clusters if len(c) > 1),
        "pronominal_mentions": pronominal,
        "pronominal_share": pronominal / max(1, len(mentions)),
    }


# ------------------------------------------------------------------------------------- main
METRICS = (("node", "node_precision_recall_f1"), ("edge", "edge_precision_recall_f1"), ("triple", "triple_f1"))


def summarize_run(run: Path, n_boot: int) -> dict:
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    language = results["language"]
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8")) if (run / "run_config.json").exists() else {}
    summary = {"run": run.name, "language": language, "config": config, "coreference": {}, "graph": {}}
    key_path = run / "coref" / "key.conllu"
    summary["corpus"] = corpus_stats(key_path, language)

    art = run / "artifacts"
    per_doc: dict[tuple[str, str], dict[str, list[float]]] = {}
    rescored = 0
    for pairing in results["results"]:
        res, be = pairing["resolver"], pairing["graph_backend"]
        values = defaultdict(list)
        for d in pairing["documents"]:
            metrics = d["graph_metrics"]
            # graphs saved with --save-artifacts are re-scored with the current graph_scoring, so
            # a fix to a metric (e.g. Smatch, 2026-10-05) reaches old runs without new LLM calls
            o_path = art / "graphs" / be / "ORACLE" / f"{d['doc_id']}.json"
            p_path = art / "graphs" / be / res / f"{d['doc_id']}.json"
            if o_path.is_file() and p_path.is_file():
                metrics = compute_graph_scores(json.loads(o_path.read_text(encoding="utf-8")),
                                               json.loads(p_path.read_text(encoding="utf-8")), be)
                rescored += 1
            for short, key in METRICS:
                v = metrics[key]
                values[short].append(v["f1"] if isinstance(v, dict) else v)
        per_doc[(res, be)] = values
        if pairing["coreference_metrics"] and res not in summary["coreference"]:
            c = pairing["coreference_metrics"]
            entry = {"reported_match": c.get("match"), "scores": {}}
            response = run / "coref" / f"{res}.response.conllu"
            if scorer_available() and response.exists():
                for match in MATCH_MODES:
                    try:
                        s = score_corpus(key_path, response, match=match)
                    except RuntimeError as exc:  # e.g. head/partial on data without syntax
                        entry["scores"][match] = {"error": str(exc)[:200]}
                        continue
                    entry["scores"][match] = {k: s[k] for k in ("muc", "b_cubed", "ceafe", "conll_f1")}
                if n_boot:
                    entry["conll_f1_ci95"] = coref_bootstrap(key_path, response, c.get("match", "exact"), n_boot)
            summary["coreference"][res] = entry

    backends = sorted({be for _, be in per_doc})
    for be in backends:
        base = per_doc.get(("NoResolution", be))
        gold = per_doc.get(("Gold", be))
        for (res, b), values in per_doc.items():
            if b != be:
                continue
            row = {}
            for short, _ in METRICS:
                vals = values[short]
                row[short] = {"mean": statistics.fmean(vals), "ci95": bootstrap_ci(vals, n=max(n_boot, 200))}
                if base and res != "NoResolution":
                    row[short]["vs_NoResolution"] = paired_bootstrap(vals, base[short], n=max(n_boot, 200))
                if base and gold and res not in ("NoResolution", "Gold"):
                    span = statistics.fmean(gold[short]) - statistics.fmean(base[short])
                    row[short]["gap_closed"] = ((statistics.fmean(vals) - statistics.fmean(base[short])) / span
                                                if abs(span) > 1e-9 else None)
            row["documents"] = len(values["node"])
            summary["graph"][f"{res} x {be}"] = row

    summary["graph_metrics_rescored_documents"] = rescored
    summary["texts"] = text_diagnostics(art, language)
    summary["graph_sizes"] = graph_sizes(art, language)
    return summary


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def to_markdown(summaries: list[dict]) -> str:
    out = []
    for s in summaries:
        out.append(f"## {s['run']} ({s['language']})\n")
        c = s["corpus"]
        out.append(f"Corpus: {c['documents']} documents, {c['tokens']} tokens, {c['mentions']} mentions, "
                   f"{c['entities']} entities ({c['non_singleton_entities']} non-singleton), "
                   f"pronominal mentions {c['pronominal_share']:.1%}.\n")
        if s["coreference"]:
            out.append("| Resolver | Match | MUC F1 | B³ F1 | CEAFe F1 | CoNLL F1 |\n|---|---|---|---|---|---|")
            for res, e in s["coreference"].items():
                for match, sc in e["scores"].items():
                    if "error" in sc:
                        continue
                    ci = e.get("conll_f1_ci95") if match == e["reported_match"] else None
                    conll = _f(sc["conll_f1"]) + (f" [{_f(ci[0])}, {_f(ci[1])}]" if ci else "")
                    out.append(f"| {res} | {match} | {_f(sc['muc']['f1'])} | {_f(sc['b_cubed']['f1'])} | "
                               f"{_f(sc['ceafe']['f1'])} | {conll} |")
            out.append("")
        out.append("| Resolver × backend | Node F1 [95% CI] | Edge F1 [95% CI] | Triple F1 [95% CI] | "
                   "Δ edge vs NoRes [95% CI], p | Gap closed (edge) |\n|---|---|---|---|---|---|")
        for name, row in s["graph"].items():
            cells = [f"{_f(row[m]['mean'])} [{_f(row[m]['ci95'][0])}, {_f(row[m]['ci95'][1])}]"
                     for m in ("node", "edge", "triple")]
            d = row["edge"].get("vs_NoResolution")
            delta = f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}], p={d['p_one_sided']:.3f}" if d else "–"
            out.append(f"| {name} | " + " | ".join(cells) + f" | {delta} | {_f(row['edge'].get('gap_closed'), 2)} |")
        out.append("")
        if s["texts"]:
            out.append("| Text | Pronouns / 1000 words | Length ratio | Sim. to oracle | Sim. to original |\n|---|---|---|---|---|")
            for name, t in s["texts"].items():
                out.append(f"| {name} | {_f(t['pronouns_per_1000_words'], 1)} | {_f(t['length_ratio_to_original'])} | "
                           f"{_f(t['char_similarity_to_oracle'])} | {_f(t['char_similarity_to_original'])} |")
            out.append("")
        if s["graph_sizes"]:
            out.append("| Graphs | Mean nodes | Mean edges | Pronoun nodes |\n|---|---|---|---|")
            for name, g in s["graph_sizes"].items():
                out.append(f"| {name} | {_f(g['mean_nodes'], 1)} | {_f(g['mean_edges'], 1)} | "
                           f"{g['pronoun_node_share']:.1%} |")
            out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    summaries = [summarize_run(r, args.bootstrap) for r in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "summary.md").write_text(to_markdown(summaries), encoding="utf-8")
    print(to_markdown(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
