import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
import json
import hashlib
from datetime import datetime, timedelta, timezone

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


st.set_page_config(
    page_title="Autonomous Research Team Demo",
    page_icon="🧠",
    layout="wide",
)


APP_NAME = "Autonomous Research Team — Live Demo"

DEFAULT_QUESTION = (
    "How do generative AI coding assistants affect developer productivity and code quality?"
)

SAMPLE_QUESTIONS = [
    DEFAULT_QUESTION,
    "What do recent studies say about remote work effects on employee performance and well-being?",
    "How effective are carbon capture and storage technologies at reducing industrial emissions?",
    "What are the main security risks of open-source AI model deployment?",
    "How should cities balance housing density with heritage preservation?",
]

EVAL_QUESTIONS = [
    "How do generative AI coding assistants affect developer productivity and code quality?",
    "What are the strongest arguments for and against universal basic income in high-income countries?",
    "How effective are carbon capture and storage technologies at reducing industrial emissions?",
    "What do recent studies say about remote work effects on employee performance and well-being?",
    "How should cities balance housing density with heritage preservation?",
    "What are the main security risks of open-source AI model deployment?",
    "How accurate are large language models at summarizing clinical trial reports?",
    "What policy levers most effectively reduce food waste in supply chains?",
    "How does microplastic exposure affect human health according to current evidence?",
    "What are the measurable benefits of four-day workweek pilots?",
]

STAGES = ["Planner", "Searcher", "Critic", "Writer", "Final"]


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10_000_000


def truncate(text: str, n: int) -> str:
    text = str(text)
    if len(text) <= n:
        return text
    return text[: max(0, n - 3)] + "..."


def safe_ascii(text: str) -> str:
    return str(text).encode("ascii", "ignore").decode("ascii")


def safe_truncate(text: str, n: int) -> str:
    return truncate(safe_ascii(text), n)


# ---------------------------------------------------------------------
# Simulated source data
# ---------------------------------------------------------------------

def make_source_pool(question: str):
    short = truncate(question, 62)

    return [
        {
            "title": f"Peer-reviewed evidence synthesis: {short}",
            "url": "https://arxiv.org/demo/abs/2401.01492",
            "source_type": "academic",
            "published": "2024",
            "reliability": "high",
            "query_tag": "systematic review",
            "summary": (
                "A synthesis of recent studies finds meaningful effects in controlled settings, "
                "with stronger results when tasks are well-scoped and evaluation criteria are explicit."
            ),
        },
        {
            "title": f"Industry benchmark report: {short}",
            "url": "https://github.blog/demo/research-quantifying-impact",
            "source_type": "industry research",
            "published": "2023",
            "reliability": "high",
            "query_tag": "benchmark productivity",
            "summary": (
                "Benchmark-style evidence suggests that assistance tools can reduce time spent on "
                "routine implementation work, while review burden and quality controls remain important."
            ),
        },
        {
            "title": f"Practitioner survey: {short}",
            "url": "https://survey.stackoverflow.co/demo/2024/",
            "source_type": "survey",
            "published": "2024",
            "reliability": "medium",
            "query_tag": "practitioner survey",
            "summary": (
                "Survey responses show growing adoption alongside concerns about accuracy, trust, "
                "maintenance overhead, and the need for human verification."
            ),
        },
        {
            "title": f"Public-sector governance framework: {short}",
            "url": "https://www.nist.gov/demo/ai-risk-management-framework",
            "source_type": "government framework",
            "published": "2023",
            "reliability": "high",
            "query_tag": "governance risk framework",
            "summary": (
                "Governance guidance emphasizes mapping, measurement, management, and human oversight "
                "across the technology lifecycle."
            ),
        },
        {
            "title": f"Policy and innovation analysis: {short}",
            "url": "https://www.oecd.org/demo/science-technology-innovation",
            "source_type": "policy analysis",
            "published": "2024",
            "reliability": "high",
            "query_tag": "policy innovation analysis",
            "summary": (
                "Policy analysis indicates that measured benefits depend on workflow redesign, training, "
                "data quality, and complementary organizational changes."
            ),
        },
    ]


def make_extra_source_pool(question: str):
    short = truncate(question, 55)

    return [
        {
            "title": f"Targeted fact-check: {short}",
            "url": "https://www.factcheck.org/demo/targeted-review",
            "source_type": "fact-check",
            "published": "2024",
            "reliability": "high",
            "query_tag": "targeted fact-check",
            "summary": (
                "Targeted search looked for primary evidence supporting the absolute claim and found "
                "no reputable source. The claim should be removed or rewritten with uncertainty."
            ),
        },
        {
            "title": f"Replication and limitations note: {short}",
            "url": "https://www.nature.com/demo/articles/s41586-024-00002-9",
            "source_type": "academic note",
            "published": "2024",
            "reliability": "high",
            "query_tag": "replication limitations",
            "summary": (
                "A limitations note stresses that broad generalizations require replication across "
                "domains, populations, and task types before they can be treated as established."
            ),
        },
    ]


# ---------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------

def extract_section(markdown_text: str, header: str, max_lines: int = 6):
    lines = markdown_text.splitlines()
    out = []
    capture = False

    for line in lines:
        if line.startswith(header):
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture and line.strip():
            out.append(line.strip())

    return out[:max_lines]


def build_report_markdown(question, started_at, final_claims, all_claims, references, metrics):
    finding_lines = []
    for claim in final_claims:
        refs = ", ".join(str(r) for r in claim.get("refs", []))
        finding_lines.append(f"- {claim['text']} [{refs}]")

    excluded_lines = []
    for claim in all_claims:
        if claim["status"].startswith("unsupported"):
            resolution = claim.get("resolution", "No credible support found.")
            excluded_lines.append(f"- {claim['original_text']} -- {resolution}")

    ref_lines = []
    for ref in references:
        ref_lines.append(
            f"{ref['number']}. {ref['title']} -- {ref['url']} "
            f"({ref['source_type']}, {ref['published']}, reliability: {ref['reliability']})"
        )

    findings_md = "\n".join(finding_lines) if finding_lines else "- No supported findings were produced in this simulated run."
    excluded_md = "\n".join(excluded_lines) if excluded_lines else "- None."
    references_md = "\n".join(ref_lines) if ref_lines else "- No references."

    confidence = "moderate-to-high" if metrics["citation_accuracy"] >= 0.90 else "moderate"

    exec_summary = (
        f"This simulated report addresses: **{question}**. "
        f"The multi-agent workflow produced {metrics['final_findings']} supported findings from "
        f"{metrics['references_count']} cited sources. The Critic identified "
        f"{metrics['unsupported_claims_found']} unsupported or overgeneralized claims and removed "
        f"them after follow-up search. Overall confidence is {confidence}, based on source diversity, "
        f"citation checking, and explicit limitation tracking."
    )

    report = f"""# {question}

**Generated by:** Autonomous Research Team demo (Planner, Searcher, Critic, Writer)  
**Mode:** simulated data, no API key required  
**Run time:** {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Executive summary

{exec_summary}

## Key findings

{findings_md}

## Claims removed or qualified by the Critic

{excluded_md}

## Method

1. The Planner decomposed the question into bounded sub-questions.
2. The Searcher collected notes, stored URLs, and wrote short paraphrased summaries.
3. The Critic checked each claim against available sources and flagged unsupported statements.
4. Unsupported claims were sent back to the Searcher for up to {metrics['max_loops']} loops.
5. The Writer drafted a cited report using only supported or provisionally supported findings.

## Limitations

- This demo uses deterministic simulated sources so it can run without an API key or live web access.
- Citation accuracy is estimated from the simulated source graph, not from a live fact-checking service.
- Production use should connect real search APIs, document parsers, vector retrieval, and model providers.
- Human review is still recommended before using the report for high-stakes decisions.

## References

{references_md}
"""

    return report


# ---------------------------------------------------------------------
# Main simulated run builder
# ---------------------------------------------------------------------

def build_demo_run(question: str, max_loops: int = 3, budget_usd: float = 0.30):
    question = str(question).strip() or DEFAULT_QUESTION
    max_loops = int(max_loops)
    budget_usd = float(budget_usd)

    seed = stable_seed(question)
    rng = np.random.default_rng(seed)
    started_at = datetime.now(timezone.utc)

    logs = []
    clock = 0.0

    def add_log(agent, event, detail, tokens=0, level="info"):
        nonlocal clock
        clock += float(rng.uniform(0.25, 1.25))
        logs.append(
            {
                "timestamp": (started_at + timedelta(seconds=clock)).strftime("%H:%M:%S"),
                "agent": agent,
                "event": event,
                "detail": truncate(str(detail), 180),
                "tokens": int(tokens),
                "level": level,
            }
        )

    add_log("System", "run_started", question, 0)

    # -------------------------
    # Planner
    # -------------------------
    subquestions = [
        f"Clarify the scope and key terms in: {truncate(question, 80)}.",
        "Identify recent academic and empirical evidence.",
        "Collect industry, government, and standards sources.",
        "Detect overgeneralized or unsupported claims.",
        "Synthesize a cited report with limitations and references.",
    ]

    planner_tokens = int(rng.integers(420, 760))
    add_log("Planner", "plan_created", f"{len(subquestions)} sub-questions", planner_tokens)

    # -------------------------
    # Searcher
    # -------------------------
    sources = make_source_pool(question)
    notes = []
    searcher_tokens = 0

    for src in sources:
        tok = int(rng.integers(160, 320))
        searcher_tokens += tok

        note = {
            "id": f"N{len(notes) + 1}",
            "query": f"{truncate(question, 70)} {src['query_tag']}",
            "url": src["url"],
            "title": src["title"],
            "source_type": src["source_type"],
            "published": src["published"],
            "reliability": src["reliability"],
            "summary": src["summary"],
        }
        notes.append(note)

        add_log("Searcher", "page_fetched", src["url"], tok // 3)
        add_log("Searcher", "note_stored", truncate(src["summary"], 90), tok)

    source_urls = [s["url"] for s in sources]
    topic = truncate(question, 72)

    claims = [
        {
            "id": "C1",
            "text": (
                f"Recent evidence related to '{topic}' shows measurable effects, but outcomes vary "
                "by setting, task type, and implementation quality."
            ),
            "support_urls": [source_urls[0], source_urls[1]],
        },
        {
            "id": "C2",
            "text": (
                "Governance, human oversight, and evaluation design materially influence whether "
                "benefits are realized safely."
            ),
            "support_urls": [source_urls[3]],
        },
        {
            "id": "C3",
            "text": f"The question '{topic}' has one universally agreed best practice.",
            "support_urls": [],
        },
        {
            "id": "C4",
            "text": (
                "Leading studies recommend combining quantitative metrics with qualitative "
                "stakeholder feedback."
            ),
            "support_urls": [source_urls[1], source_urls[2]],
        },
        {
            "id": "C5",
            "text": (
                "Evidence quality varies by domain, dataset, and replication status, so conclusions "
                "should carry explicit uncertainty."
            ),
            "support_urls": [source_urls[0], source_urls[4]],
        },
        {
            "id": "C6",
            "text": "No human review is required once an automated system produces an output.",
            "support_urls": [],
        },
    ]

    # -------------------------
    # Critic
    # -------------------------
    critique_log = []
    critic_tokens = 0

    for claim in claims:
        claim["original_text"] = claim["text"]
        tok = int(rng.integers(70, 160))
        critic_tokens += tok

        if not claim["support_urls"]:
            claim["status"] = "unsupported"
            critique_log.append(
                {
                    "loop": 0,
                    "claim_id": claim["id"],
                    "finding": "No supporting source found.",
                    "action": "send_to_searcher",
                }
            )
            add_log("Critic", "unsupported_claim_flagged", claim["id"], tok, "warning")

        elif len(claim["support_urls"]) == 1:
            claim["status"] = "weakly_supported"
            claim["text"] = claim["text"] + " Single-source finding; treat as provisional."
            critique_log.append(
                {
                    "loop": 0,
                    "claim_id": claim["id"],
                    "finding": "Only one supporting source found.",
                    "action": "add_caveat",
                }
            )
            add_log("Critic", "weak_support_flagged", claim["id"], tok, "warning")

        else:
            claim["status"] = "supported"
            critique_log.append(
                {
                    "loop": 0,
                    "claim_id": claim["id"],
                    "finding": "Supported by multiple sources.",
                    "action": "accept",
                }
            )
            add_log("Critic", "claim_accepted", claim["id"], tok)

    unsupported_claims = [c for c in claims if c["status"] == "unsupported"]
    unsupported_found = len(unsupported_claims)
    loops_used = 0
    extra_pool = make_extra_source_pool(question)

    for claim in unsupported_claims:
        if loops_used >= max_loops:
            claim["status"] = "unsupported_unresolved"
            claim["resolution"] = "Loop budget reached before follow-up search; claim excluded from final report."
            critique_log.append(
                {
                    "loop": loops_used,
                    "claim_id": claim["id"],
                    "finding": "Loop budget reached.",
                    "action": "exclude",
                }
            )
            add_log("Critic", "loop_budget_reached", claim["id"], 0, "warning")
            continue

        loops_used += 1
        extra = extra_pool[(loops_used - 1) % len(extra_pool)]
        tok = int(rng.integers(180, 340))
        searcher_tokens += tok

        note = {
            "id": f"N{len(notes) + 1}",
            "query": f"verify {claim['id']} {truncate(claim['original_text'], 60)}",
            "url": extra["url"],
            "title": extra["title"],
            "source_type": extra["source_type"],
            "published": extra["published"],
            "reliability": extra["reliability"],
            "summary": extra["summary"],
        }
        notes.append(note)

        add_log("Searcher", "followup_search", extra["url"], tok // 3)
        add_log("Searcher", "note_stored", truncate(extra["summary"], 90), tok)

        claim["status"] = "unsupported_removed"
        claim["resolution"] = "Follow-up search found no credible support; claim removed from final findings."

        critique_log.append(
            {
                "loop": loops_used,
                "claim_id": claim["id"],
                "finding": "Unsupported absolute claim after targeted search.",
                "action": "removed_after_followup",
            }
        )
        add_log("Critic", "unsupported_claim_removed", claim["id"], int(rng.integers(70, 150)))

    weak_count = sum(1 for c in claims if c["status"] == "weakly_supported")

    # -------------------------
    # Writer
    # -------------------------
    final_claims = [c for c in claims if c["status"] in ("supported", "weakly_supported")]

    cited_urls = []
    for claim in final_claims:
        for url in claim["support_urls"]:
            if url not in cited_urls:
                cited_urls.append(url)

    notes_by_url = {n["url"]: n for n in notes}
    references = []
    url_to_ref = {}

    for i, url in enumerate(cited_urls, start=1):
        note = notes_by_url.get(url)
        if note:
            ref = {
                "number": i,
                "title": note["title"],
                "url": note["url"],
                "source_type": note["source_type"],
                "published": note["published"],
                "reliability": note["reliability"],
            }
            references.append(ref)
            url_to_ref[url] = i

    for claim in final_claims:
        claim["refs"] = [url_to_ref[url] for url in claim["support_urls"] if url in url_to_ref]

    writer_tokens = int(rng.integers(850, 1500))
    add_log("Writer", "draft_started", "Composing executive summary, findings, limitations, references", writer_tokens // 3)

    total_tokens = planner_tokens + searcher_tokens + critic_tokens + writer_tokens
    estimated_cost = round(total_tokens / 1000.0 * 0.0085, 4)
    budget_status = "within_budget" if estimated_cost <= budget_usd else "over_budget"

    citation_accuracy = float(
        np.clip(
            0.99 - 0.035 * unsupported_found - 0.02 * weak_count + float(rng.normal(0.0, 0.008)),
            0.70,
            0.995,
        )
    )

    metrics = {
        "citation_accuracy": round(citation_accuracy, 3),
        "unsupported_claims_found": unsupported_found,
        "weak_claims_found": weak_count,
        "loops_used": loops_used,
        "max_loops": max_loops,
        "total_tokens": total_tokens,
        "planner_tokens": planner_tokens,
        "searcher_tokens": searcher_tokens,
        "critic_tokens": critic_tokens,
        "writer_tokens": writer_tokens,
        "estimated_cost_usd": estimated_cost,
        "budget_usd": budget_usd,
        "budget_status": budget_status,
        "final_findings": len(final_claims),
        "references_count": len(references),
    }

    report_markdown = build_report_markdown(
        question=question,
        started_at=started_at,
        final_claims=final_claims,
        all_claims=claims,
        references=references,
        metrics=metrics,
    )

    add_log("Writer", "draft_completed", f"{len(final_claims)} findings, {len(references)} references", writer_tokens)
    add_log("System", "run_completed", f"Cost ${estimated_cost:.4f}", 0)

    return {
        "question": question,
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "subquestions": subquestions,
        "notes": notes,
        "claims": claims,
        "critique_log": critique_log,
        "logs": logs,
        "references": references,
        "report_markdown": report_markdown,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------

VIDEO_W, VIDEO_H, FPS = 1280, 720, 15
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (24, 26, 32)
HEADER_BG = (31, 35, 45)
PANEL = (38, 43, 55)
PANEL_LIGHT = (49, 56, 70)
BORDER = (72, 82, 102)
TEXT = (236, 239, 246)
MUTED = (154, 162, 178)
DARK = (18, 18, 22)
BLUE = (255, 170, 90)
GREEN = (100, 210, 110)
YELLOW = (90, 200, 255)
RED = (75, 75, 235)


def new_frame():
    img = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    img[:] = BG
    return img


def put_wrapped(
    img,
    text,
    org,
    font_scale=0.55,
    color=TEXT,
    thickness=1,
    max_width=1100,
    line_gap=8,
    max_lines=12,
):
    text = safe_ascii(text)
    if not text:
        return org[1]

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, FONT, font_scale, thickness)
        if tw <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = safe_truncate(lines[-1], max(10, len(lines[-1]) - 3)) + "..."

    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
        (_, th), _ = cv2.getTextSize(line, FONT, font_scale, thickness)
        y += th + line_gap

    return y


def scene_frame(title, kicker):
    img = new_frame()
    cv2.rectangle(img, (0, 0), (VIDEO_W, 112), HEADER_BG, -1)
    cv2.putText(img, safe_ascii(kicker), (45, 42), FONT, 0.52, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, safe_truncate(title, 72), (45, 82), FONT, 0.88, TEXT, 2, cv2.LINE_AA)
    cv2.line(img, (45, 106), (VIDEO_W - 45, 106), BORDER, 1)
    return img


def draw_timeline(img, progress):
    progress = max(0.0, min(1.0, progress))
    y = VIDEO_H - 42

    cv2.putText(
        img,
        f"Demo progress {int(progress * 100)}%",
        (45, y - 12),
        FONT,
        0.45,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(img, (45, y), (VIDEO_W - 45, y + 10), PANEL_LIGHT, -1)
    filled_w = int((VIDEO_W - 90) * progress)
    cv2.rectangle(img, (45, y), (45 + filled_w, y + 10), BLUE, -1)


def scene_intro(run):
    img = scene_frame("Autonomous Research Team", "DEMO • NO API KEY REQUIRED")

    put_wrapped(img, run["question"], (50, 185), 0.78, TEXT, 2, 1150, 12, 4)
    put_wrapped(
        img,
        (
            "Four cooperating agents turn a research question into a cited, fact-checked report. "
            "This run uses deterministic sample data, critique loops, budget tracking, and structured logs."
        ),
        (50, 330),
        0.56,
        MUTED,
        1,
        1120,
        10,
        4,
    )

    x = 50
    for agent in ["Planner", "Searcher", "Critic", "Writer"]:
        (tw, _), _ = cv2.getTextSize(agent, FONT, 0.62, 2)
        cv2.rectangle(img, (x, 500), (x + tw + 44, 560), PANEL, -1)
        cv2.putText(img, agent, (x + 22, 540), FONT, 0.62, BLUE, 2, cv2.LINE_AA)
        x += tw + 70

    put_wrapped(
        img,
        (
            "Generated MP4 walkthrough starts automatically. Use the controls to inspect each agent, "
            "export artifacts, or run a new simulated question."
        ),
        (50, 610),
        0.50,
        MUTED,
        1,
        1150,
        8,
        2,
    )

    return img


def scene_planner(run):
    img = scene_frame("Planner breaks the question into sub-questions", "STEP 1 • PLAN")

    y = 155
    for i, sq in enumerate(run["subquestions"][:5], start=1):
        cv2.rectangle(img, (50, y), (VIDEO_W - 50, y + 78), PANEL, -1)
        cv2.circle(img, (86, y + 39), 18, BLUE, -1)
        cv2.putText(img, str(i), (79, y + 47), FONT, 0.62, DARK, 2, cv2.LINE_AA)
        put_wrapped(img, sq, (122, y + 32), 0.56, TEXT, 1, 980, 8, 2)
        y += 92

    return img


def scene_searcher(run):
    img = scene_frame("Searcher collects notes with URLs and summaries", "STEP 2 • SEARCH")

    y = 150
    for note in run["notes"][:3]:
        cv2.rectangle(img, (50, y), (VIDEO_W - 50, y + 140), PANEL, -1)
        cv2.putText(img, safe_truncate(note["title"], 62), (75, y + 34), FONT, 0.58, TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, safe_truncate(note["url"], 78), (75, y + 64), FONT, 0.46, GREEN, 1, cv2.LINE_AA)
        put_wrapped(img, note["summary"], (75, y + 94), 0.48, MUTED, 1, 1080, 8, 2)
        y += 160

    return img


def scene_critic(run):
    img = scene_frame("Critic checks claims against sources", "STEP 3 • CRITIQUE")

    y = 150
    for claim in run["claims"][:4]:
        status = claim["status"]

        if status.startswith("unsupported"):
            color = RED
            label = "UNSUPPORTED"
        elif status == "weakly_supported":
            color = YELLOW
            label = "WEAK"
        else:
            color = GREEN
            label = "SUPPORTED"

        cv2.rectangle(img, (50, y), (770, y + 92), PANEL, -1)
        cv2.putText(img, f"{claim['id']} {label}", (70, y + 30), FONT, 0.52, color, 2, cv2.LINE_AA)
        put_wrapped(img, claim["original_text"], (70, y + 56), 0.46, TEXT, 1, 650, 8, 2)
        y += 108

    cv2.rectangle(img, (800, 150), (VIDEO_W - 50, 560), PANEL, -1)
    cv2.putText(img, "Critic loop", (825, 200), FONT, 0.70, TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, f"Max loops: {run['metrics']['max_loops']}", (825, 245), FONT, 0.55, MUTED, 1, cv2.LINE_AA)
    cv2.putText(img, f"Loops used: {run['metrics']['loops_used']}", (825, 285), FONT, 0.55, BLUE, 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Unsupported found: {run['metrics']['unsupported_claims_found']}",
        (825, 325),
        FONT,
        0.55,
        RED,
        2,
        cv2.LINE_AA,
    )
    put_wrapped(
        img,
        (
            "Unsupported claims are sent back to Searcher. If no credible source is found, "
            "the claim is removed before writing."
        ),
        (825, 375),
        0.50,
        MUTED,
        1,
        360,
        10,
        4,
    )

    return img


def scene_writer(run):
    img = scene_frame("Writer drafts a cited report", "STEP 4 • WRITE")

    cv2.rectangle(img, (50, 150), (820, 620), PANEL, -1)
    cv2.putText(img, "Executive summary", (80, 195), FONT, 0.65, TEXT, 2, cv2.LINE_AA)

    summary_lines = extract_section(run["report_markdown"], "## Executive summary", 5)
    y = 235
    for line in summary_lines:
        y = put_wrapped(img, line, (80, y), 0.50, TEXT, 1, 680, 8, 3) + 8

    cv2.rectangle(img, (850, 150), (VIDEO_W - 50, 620), PANEL, -1)
    cv2.putText(img, "References", (880, 195), FONT, 0.65, TEXT, 2, cv2.LINE_AA)

    y = 235
    for ref in run["references"][:5]:
        put_wrapped(img, f"[{ref['number']}] {ref['title']}", (880, y), 0.48, TEXT, 1, 330, 6, 2)
        put_wrapped(img, ref["url"], (880, y + 25), 0.40, GREEN, 1, 330, 6, 1)
        y += 72

    return img


def scene_metrics(run):
    img = scene_frame("Run metrics and evaluation snapshot", "STEP 5 • MEASURE")

    m = run["metrics"]
    tiles = [
        ("Citation accuracy", f"{m['citation_accuracy'] * 100:.1f}%"),
        ("Unsupported claims", str(m["unsupported_claims_found"])),
        ("Loops used", f"{m['loops_used']} / {m['max_loops']}"),
        ("Total tokens", f"{m['total_tokens']:,}"),
        ("Estimated cost", f"${m['estimated_cost_usd']:.4f}"),
        ("Budget", f"${m['budget_usd']:.2f}"),
    ]

    for idx, (label, value) in enumerate(tiles):
        col = idx % 3
        row = idx // 3
        x = 50 + col * 395
        y = 165 + row * 195

        cv2.rectangle(img, (x, y), (x + 360, y + 150), PANEL, -1)
        cv2.putText(img, label, (x + 25, y + 45), FONT, 0.55, MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, value, (x + 25, y + 100), FONT, 0.90, BLUE, 2, cv2.LINE_AA)

    put_wrapped(
        img,
        (
            "Next production step: connect real Claude API calls, live web search, page fetching, "
            "Chroma retrieval, and LangGraph or CrewAI orchestration."
        ),
        (50, 590),
        0.55,
        MUTED,
        1,
        1150,
        8,
        2,
    )

    return img


def generate_video_bytes(run):
    scenes = [
        scene_intro(run),
        scene_planner(run),
        scene_searcher(run),
        scene_critic(run),
        scene_writer(run),
        scene_metrics(run),
    ]

    durations = [2.0, 2.3, 2.3, 2.5, 2.5, 2.0]
    frame_counts = [max(1, int(d * FPS)) for d in durations]
    total_frames = sum(frame_counts)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "demo.mp4")

        writer = None
        use_imageio = False

        if imageio is not None:
            try:
                writer = imageio.get_writer(
                    out_path,
                    fps=FPS,
                    codec="libx264",
                    quality=7,
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
                use_imageio = True
            except Exception:
                writer = None

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, FPS, (VIDEO_W, VIDEO_H))
            if not writer.isOpened():
                raise RuntimeError("Could not open a video writer. Install imageio-ffmpeg or opencv-python-headless.")

        frame_idx = 0
        for scene, n_frames in zip(scenes, frame_counts):
            for _ in range(n_frames):
                frame = scene.copy()
                draw_timeline(frame, frame_idx / max(1, total_frames - 1))

                if use_imageio:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    writer.write(frame)

                frame_idx += 1

        writer.close()

        with open(out_path, "rb") as fh:
            return fh.read()


# ---------------------------------------------------------------------
# Evaluation table
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_evaluation_table():
    rows = []

    for i, q in enumerate(EVAL_QUESTIONS, start=1):
        run = build_demo_run(q, max_loops=3, budget_usd=0.30)
        m = run["metrics"]

        rows.append(
            {
                "id": f"Q{i}",
                "question": truncate(q, 80),
                "citation_accuracy": m["citation_accuracy"],
                "unsupported_claims": m["unsupported_claims_found"],
                "loops": m["loops_used"],
                "cost_usd": m["estimated_cost_usd"],
                "tokens": m["total_tokens"],
            }
        )

    df = pd.DataFrame(rows)
    df["citation_accuracy_pct"] = (df["citation_accuracy"] * 100).round(1)
    return df


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------

def render_stage(stage, run):
    m = run["metrics"]

    if stage == "Planner":
        st.markdown("#### Planner output")
        st.write("The Planner decomposes the research question into bounded sub-questions.")
        for i, sq in enumerate(run["subquestions"], start=1):
            st.markdown(f"**{i}.** {sq}")
        st.caption(f"Simulated planner tokens: {m['planner_tokens']:,}")

    elif stage == "Searcher":
        st.markdown("#### Searcher notes")
        st.write("Each note stores a URL, source type, reliability label, and short paraphrased summary.")
        df = pd.DataFrame(run["notes"])
        cols = ["id", "query", "title", "url", "source_type", "published", "reliability", "summary"]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.caption(f"Stored notes: {len(df)} | Simulated searcher tokens: {m['searcher_tokens']:,}")

    elif stage == "Critic":
        st.markdown("#### Claim checking")
        claim_rows = []
        for c in run["claims"]:
            claim_rows.append(
                {
                    "ID": c["id"],
                    "Status": c["status"],
                    "Claim": c["original_text"],
                    "Supporting URLs": ", ".join(c["support_urls"]),
                    "Resolution": c.get("resolution", ""),
                }
            )

        st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Critique log")
        st.dataframe(pd.DataFrame(run["critique_log"]), use_container_width=True, hide_index=True)
        st.caption(
            f"Loops used: {m['loops_used']} / {m['max_loops']} | "
            f"Unsupported found: {m['unsupported_claims_found']} | "
            f"Weak claims: {m['weak_claims_found']}"
        )

    elif stage == "Writer":
        st.markdown("#### Writer draft")
        st.markdown(run["report_markdown"])

    else:
        st.markdown("#### Final run summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final findings", m["final_findings"])
        c2.metric("References", m["references_count"])
        c3.metric("Citation accuracy", f"{m['citation_accuracy'] * 100:.1f}%")
        c4.metric("Estimated cost", f"${m['estimated_cost_usd']:.4f}")

        st.markdown("#### References")
        if run["references"]:
            ref_df = pd.DataFrame(run["references"])
            st.dataframe(
                ref_df[["number", "title", "url", "source_type", "published", "reliability"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No references were produced.")


def main():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.question = DEFAULT_QUESTION
        st.session_state.max_loops = 3
        st.session_state.budget = 0.30
        st.session_state.mode = "Sample questions"
        st.session_state.stage = "Final"

        with st.spinner("Loading realistic demo data and generating MP4 walkthrough..."):
            st.session_state.run = build_demo_run(
                st.session_state.question,
                st.session_state.max_loops,
                st.session_state.budget,
            )
            try:
                st.session_state.video = generate_video_bytes(st.session_state.run)
                st.session_state.video_error = None
            except Exception as exc:
                st.session_state.video = None
                st.session_state.video_error = str(exc)

    st.sidebar.title("Control panel")

    mode = st.sidebar.radio(
        "Question source",
        ["Sample questions", "Custom question"],
        key="mode",
    )

    if mode == "Sample questions":
        idx = (
            SAMPLE_QUESTIONS.index(st.session_state.question)
            if st.session_state.question in SAMPLE_QUESTIONS
            else 0
        )
        selected_q = st.sidebar.selectbox("Sample research question", SAMPLE_QUESTIONS, index=idx)
    else:
        selected_q = st.sidebar.text_input("Custom research question", value=st.session_state.question)

    max_loops = st.sidebar.slider(
        "Maximum Critic loops",
        min_value=1,
        max_value=5,
        value=int(st.session_state.max_loops),
    )

    budget = st.sidebar.number_input(
        "Cost budget per run (USD)",
        min_value=0.05,
        max_value=2.00,
        value=float(st.session_state.budget),
        step=0.05,
        format="%.2f",
    )

    st.sidebar.caption(
        "No API key required. This app uses deterministic simulated research data and renders an MP4 demo locally."
    )

    if st.sidebar.button("Run simulation", type="primary"):
        new_q = (selected_q or "").strip() or DEFAULT_QUESTION
        st.session_state.question = new_q
        st.session_state.max_loops = int(max_loops)
        st.session_state.budget = float(budget)

        with st.spinner("Planning, searching, critiquing, writing, and rendering demo video..."):
            st.session_state.run = build_demo_run(
                st.session_state.question,
                st.session_state.max_loops,
                st.session_state.budget,
            )
            try:
                st.session_state.video = generate_video_bytes(st.session_state.run)
                st.session_state.video_error = None
                st.sidebar.success("Simulation ready.")
            except Exception as exc:
                st.session_state.video = None
                st.session_state.video_error = str(exc)
                st.sidebar.warning("Simulation ready, but video generation failed.")

    st.title("🧠 Autonomous Research Team — Live Demo")
    st.caption(
        "Planner → Searcher → Critic → Writer | simulated multi-agent workflow with citations, "
        "critique loops, budget tracking, structured logs, and an auto-generated MP4 demo."
    )

    run = st.session_state.run
    video = st.session_state.video

    tab_demo, tab_report, tab_logs, tab_eval, tab_about = st.tabs(
        ["Live demo", "Report", "Agent logs", "10-question eval", "About"]
    )

    with tab_demo:
        col_video, col_metrics = st.columns([2.2, 1.0])

        with col_video:
            st.subheader("Generated MP4 walkthrough")
            if video:
                st.video(
                    video,
                    format="video/mp4",
                    loop=True,
                    autoplay=True,
                    muted=True,
                )
                st.download_button(
                    "Download MP4",
                    data=video,
                    file_name="autonomous_research_team_demo.mp4",
                    mime="video/mp4",
                )
            else:
                st.error("Video generation failed. The interactive dashboard below still works.")
                if st.session_state.get("video_error"):
                    st.code(st.session_state.video_error)

        with col_metrics:
            m = run["metrics"]
            st.metric("Citation accuracy", f"{m['citation_accuracy'] * 100:.1f}%")
            st.metric("Unsupported claims found", m["unsupported_claims_found"])
            st.metric("Critic loops used", f"{m['loops_used']} / {m['max_loops']}")
            st.metric("Estimated cost", f"${m['estimated_cost_usd']:.4f}")
            st.metric("Budget", f"${m['budget_usd']:.2f}")
            st.metric("Total tokens", f"{m['total_tokens']:,}")

        st.markdown("---")
        st.subheader("Interactive agent walkthrough")
        stage = st.radio("Stage", STAGES, horizontal=True, key="stage")
        render_stage(stage, run)

        st.markdown("---")
        st.subheader("Export artifacts")

        c1, c2, c3, c4 = st.columns(4)

        c1.download_button(
            "report.md",
            data=run["report_markdown"].encode("utf-8"),
            file_name="research_report.md",
            mime="text/markdown",
        )

        c2.download_button(
            "notes.csv",
            data=pd.DataFrame(run["notes"]).to_csv(index=False).encode("utf-8"),
            file_name="notes.csv",
            mime="text/csv",
        )

        c3.download_button(
            "critique.csv",
            data=pd.DataFrame(run["critique_log"]).to_csv(index=False).encode("utf-8"),
            file_name="critique_log.csv",
            mime="text/csv",
        )

        c4.download_button(
            "logs.json",
            data=json.dumps(run["logs"], indent=2).encode("utf-8"),
            file_name="agent_logs.json",
            mime="application/json",
        )

    with tab_report:
        st.subheader("Final cited report")
        st.markdown(run["report_markdown"])

    with tab_logs:
        st.subheader("Structured agent logs")
        log_df = pd.DataFrame(run["logs"])
        agents = sorted(log_df["agent"].unique().tolist())
        selected_agents = st.multiselect("Filter by agent", agents, default=agents)
        filtered = log_df[log_df["agent"].isin(selected_agents)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_eval:
        st.subheader("10-question evaluation snapshot")
        with st.spinner("Building evaluation table..."):
            eval_df = get_evaluation_table()

        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average citation accuracy", f"{eval_df['citation_accuracy_pct'].mean():.1f}%")
        c2.metric("Total unsupported claims", int(eval_df["unsupported_claims"].sum()))
        c3.metric("Average cost per run", f"${eval_df['cost_usd'].mean():.4f}")

        st.markdown("#### Citation accuracy by question")
        st.bar_chart(eval_df.set_index("id")["citation_accuracy_pct"])

        st.markdown("#### Unsupported claims by question")
        st.bar_chart(eval_df.set_index("id")["unsupported_claims"])

    with tab_about:
        st.markdown(
            """
            ### What this demo shows

            This app simulates a four-agent research team:

            1. **Planner** breaks the question into sub-questions.
            2. **Searcher** collects notes with URLs and paraphrased summaries.
            3. **Critic** checks claims and sends unsupported claims back for follow-up search.
            4. **Writer** drafts a cited Markdown report with references and limitations.

            ### Why no API key is needed

            All agent behavior is deterministic and simulated. That makes the project easy to demo,
            test, and extend before connecting real services.

            ### Suggested production upgrade path

            - Replace simulated search with a real web search API.
            - Replace simulated summaries with Claude API calls.
            - Store fetched pages and notes in Chroma.
            - Use LangGraph or CrewAI for orchestration.
            - Add persistent run logging and cost metering.
            """
        )


if __name__ == "__main__":
    main()