from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_results(run_id: str, language: str, per_pairing: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": per_pairing,
    }


def save_results_json(results: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_html_report(results: dict) -> str:
    rows = []
    for entry in results["results"]:
        coref = entry.get("coreference_metrics")
        graph = entry.get("graph_metrics") or {}
        conll_f1 = coref["conll_f1"] if coref else None
        node_f1 = graph.get("node_precision_recall_f1", {}).get("f1")
        edge_f1 = graph.get("edge_precision_recall_f1", {}).get("f1")
        smatch_f1 = graph.get("smatch", {}).get("f1")
        oracle_dup = graph.get("oracle_node_duplication_rate")
        predicted_dup = graph.get("predicted_node_duplication_rate")
        rows.append(
            f"<tr><td>{entry['resolver']}</td><td>{entry['graph_backend']}</td>"
            f"<td>{_fmt(conll_f1)}</td><td>{_fmt(node_f1)}</td>"
            f"<td>{_fmt(edge_f1)}</td><td>{_fmt(smatch_f1)}</td>"
            f"<td>{_fmt(oracle_dup)}</td><td>{_fmt(predicted_dup)}</td></tr>"
        )
    table_rows = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Experiment run {results['run_id']}</title></head>
<body>
<h1>Cascade experimental stand -- run {results['run_id']}</h1>
<p>Language: {results['language']} | Generated: {results['generated_at']}</p>
<table border="1" cellpadding="4" cellspacing="0">
<thead><tr><th>Resolver</th><th>Graph backend</th><th>CoNLL F1</th>
<th>Node F1</th><th>Edge F1</th><th>Smatch F1</th>
<th>Oracle node dup. rate</th><th>Predicted node dup. rate</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body></html>"""


def save_html_report(html: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
