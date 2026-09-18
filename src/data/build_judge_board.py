"""
Build Judge Board HTML for manual head-to-head evaluation.

Reads response JSONL files for two models, generates an HTML file
where the user can judge blind (Response A vs Response B, no model names).

Output: experiments/judge_board.html

Usage: python src/data/build_judge_board.py
"""
import json
import os
from pathlib import Path

RESPONSES_DIR = Path("experiments/responses")
OUTPUT = Path("experiments/judge_board.html")


def load_responses(filepath):
    responses = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            question = obj.get("question", "")
            response = obj.get("response", "")
            responses[question] = response
    return responses


def build_html(model1_name, model2_name, model1_responses, model2_responses):
    questions = sorted(set(list(model1_responses.keys()) + list(model2_responses.keys())))

    rows_html = ""
    for i, q in enumerate(questions, 1):
        r1 = model1_responses.get(q, "N/A")
        r2 = model2_responses.get(q, "N/A")
        m1_short = r1[:100] + "..." if len(r1) > 100 else r1
        m2_short = r2[:100] + "..." if len(r2) > 100 else r2

        row = (
            '<tr>'
            '<td>' + str(i) + '</td>'
            '<td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;">' + q + '</td>'
            '<td class="response" contenteditable="true">' + m1_short + '</td>'
            '<td class="response" contenteditable="true">' + m2_short + '</td>'
            '<td>'
            '<button onclick="setWinner(' + str(i) + ', \'A\')" class="btn-a">A</button> '
            '<button onclick="setWinner(' + str(i) + ', \'B\')" class="btn-b">B</button> '
            '<button onclick="setWinner(' + str(i) + ', \'tie\')" class="btn-tie">Tie</button>'
            '</td>'
            '</tr>'
        )
        rows_html += row

    total = len(questions)

    parts = [
        '<!DOCTYPE html>',
        '<html lang="en">',
        '<head>',
        '<meta charset="UTF-8">',
        '<title>Judge Board - PaulG-LM Head-to-Head</title>',
        '<style>',
        'body{font-family:Segoe UI,system-ui,sans-serif;background:#f5f5f5;padding:24px}',
        '.header{background:#1a1a2e;color:#62ff94;padding:24px;border-radius:8px;margin-bottom:24px}',
        'h1{margin:0 0 8px 0}',
        '.model-info{font-size:14px;color:#aaa;margin-top:4px}',
        'table{width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden}',
        'th,td{padding:12px 16px;text-align:left;border-bottom:1px solid #eee;font-size:13px}',
        'th{background:#fafafa;position:sticky;top:0}',
        '.response{max-height:100px;overflow-y:auto;cursor:text}',
        '.response:focus{background:#ffffcc}',
        'button{padding:4px 12px;margin:2px;border:1px solid #ccc;border-radius:4px;cursor:pointer;font-size:12px}',
        '.btn-a:hover{background:#d4edda;border-color:#28a745}',
        '.btn-b:hover{background:#f8d7da;border-color:#dc3545}',
        '.btn-tie:hover{background:#fff3cd;border-color:#ffc107}',
        '.tally{margin-top:24px;background:#1a1a2e;color:#62ff94;padding:20px;border-radius:8px}',
        'footer{margin-top:24px;color:#999;font-size:12px;text-align:center}',
        '</style>',
        '</head>',
        '<body>',
        '<div class="header">',
        '<h1>Judge Board - PaulG-LM Head-to-Head</h1>',
        '<div class="model-info">',
        '<strong>Model A:</strong> ' + model1_name + ' | ',
        '<strong>Model B:</strong> ' + model2_name + ' | ',
        '<strong>Questions:</strong> ' + str(total),
        '</div>',
        '<div style="color:#ff6666;font-size:13px;margin-top:8px">',
        'BLIND JUDGING - You do NOT see which model is PaulG-LM. Judge Response A vs Response B only.',
        '</div>',
        '</div>',
        '<table>',
        '<thead><tr><th>#</th><th>Question</th><th>Response A</th><th>Response B</th><th>Winner</th></tr></thead>',
        '<tbody>' + rows_html + '</tbody>',
        '</table>',
        '<div class="tally">',
        '<h3>Tally</h3>',
        '<div id="tally-results">Click buttons above to record your judgments.</div>',
        '</div>',
        '<footer>PaulG-LM Evaluation | Blind Judge Board | A/B responses anonymous</footer>',
        '<script>',
        'var winners = {};',
        'function setWinner(q, winner){winners[q]=winner;updateTally();}',
        'function updateTally(){',
        'var total=Object.keys(winners).length;',
        'var aWins=Object.values(winners).filter(function(v){return v==="A";}).length;',
        'var bWins=Object.values(winners).filter(function(v){return v==="B";}).length;',
        'var ties=Object.values(winners).filter(function(v){return v==="tie";}).length;',
        'var pct=total>0?Math.round((total/' + str(total) + ')*100):0;',
        'document.getElementById("tally-results").innerHTML=',
        '"<p>Judged: "+total+"/" + ' + str(total) + ' + " ("+pct+"%)</p>"+',
        '"<p>Response A wins: "+aWins+" | Response B wins: "+bWins+" | Ties: "+ties+"</p>";',
        '}',
        '</script>',
        '</body>',
        '</html>',
    ]
    html = ''.join(parts)
    return html


def main():
    print("Building Judge Board...")

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    model1_name = "PAULGLLM"
    model2_name = "STAGE 1 ONLY"

    m1_file = RESPONSES_DIR / "paulgllm_responses.jsonl"
    m2_file = RESPONSES_DIR / "stage1_only_responses.jsonl"

    if not m1_file.exists() or not m2_file.exists():
        print("ERROR: Need response files in " + str(RESPONSES_DIR) + "/")
        print("Run eval.py first to generate responses.")
        print("Expected: " + str(m1_file) + " and " + str(m2_file))
        return

    m1_responses = load_responses(str(m1_file))
    m2_responses = load_responses(str(m2_file))

    html = build_html(model1_name, model2_name, m1_responses, m2_responses)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print("Judge Board written to: " + str(OUTPUT))
    print("Open in browser to judge blind.")


if __name__ == "__main__":
    main()
