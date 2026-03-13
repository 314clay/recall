#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "textual>=1.0.0",
#     "psycopg2-binary",
#     "google-genai",
#     "rich",
# ]
# ///
"""
recall — Claude Code Session History TUI

A navigable, searchable session browser with semantic search,
background summary generation, and project grouping.

Usage:
    recall              # Launch TUI (default: last 7 days)
    recall --brief      # Plain-text morning recap to stdout
    recall --days 14    # Look back further
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from rich.console import Console
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
)

# ─────────────────────────────────────────────────────────────────────
# Project mapping (ported from claude-sessions)
# ─────────────────────────────────────────────────────────────────────

# Project rules: map cwd patterns to short project names.
# Customize by creating ~/.config/recall/projects.json with entries like:
#   { "rules": [{"pattern": "my-app", "name": "app"}],
#     "colors": {"app": "#ff5555"} }
PROJECT_RULES: list[tuple[re.Pattern, str]] = []
PROJECT_COLORS: dict[str, str] = {}

def _load_project_config() -> None:
    """Load project rules from ~/.config/recall/projects.json if it exists."""
    config_path = os.path.expanduser("~/.config/recall/projects.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path) as f:
            config = json.load(f)
        for rule in config.get("rules", []):
            PROJECT_RULES.append(
                (re.compile(rule["pattern"], re.I), rule["name"])
            )
        PROJECT_COLORS.update(config.get("colors", {}))
    except (json.JSONDecodeError, KeyError, OSError):
        pass

_load_project_config()

FALLBACK_COLORS = [
    "#4488ff", "#aa77ff", "#ff8800", "#77cc33",
    "#ff4477", "#00cccc", "#ffcc00", "#ff7799",
]

_color_idx: dict[str, int] = {}


def derive_project(cwd: str) -> str:
    if not cwd:
        return "?"
    home = os.path.expanduser("~")
    path = cwd.replace(home, "~")
    for pattern, name in PROJECT_RULES:
        if pattern.search(path):
            return name
    parts = [p for p in path.rstrip("/").split("/") if p and p != "~"]
    if parts:
        return parts[-1][:16]
    return "?"


def get_project_color(project: str) -> str:
    if project in PROJECT_COLORS:
        return PROJECT_COLORS[project]
    if project not in _color_idx:
        _color_idx[project] = len(_color_idx) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[_color_idx[project]]


# ─────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────

def clean_goal(content: str, max_len: int = 80) -> str:
    if not content:
        return ""
    text = content.strip()
    if len(text) <= 1:
        return ""
    if text.startswith("<task-notification") or text.startswith("<task-output"):
        return ""
    if re.match(r"^/\S+\s*$", text):
        return ""
    text = re.sub(
        r"<(task-notification|task-output|system-reminder|output-file)[^>]*>.*?</\1>",
        " ", text, flags=re.S,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^/\S+\s*", "", text.strip())
    text = re.sub(r"^Resume:\s*", "", text.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def format_duration(delta: timedelta | None) -> str:
    if not delta:
        return "—"
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining_mins = minutes % 60
    if remaining_mins:
        return f"{hours}h{remaining_mins:02d}m"
    return f"{hours}h"


def format_model(model: str | None) -> str:
    if not model:
        return ""
    m = model.lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "haiku" in m:
        return "haiku"
    return model[:12]


def day_label(dt: datetime) -> str:
    today = datetime.now(timezone.utc).date()
    d = dt.date()
    if d == today:
        return f"TODAY ({d.strftime('%a %b %d')})"
    elif d == today - timedelta(days=1):
        return f"YESTERDAY ({d.strftime('%a %b %d')})"
    else:
        days_ago = (today - d).days
        return f"{days_ago}d AGO ({d.strftime('%a %b %d')})"


# ─────────────────────────────────────────────────────────────────────
# Session data model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    session_id: str
    project: str
    cwd: str
    goal: str
    summary: str
    user_requests: str
    completed_work: str
    topics: list[str]
    user_msg_count: int
    total_msg_count: int
    started: datetime
    last_active: datetime
    duration: timedelta | None
    model: str
    top_tools: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Database layer
# ─────────────────────────────────────────────────────────────────────

def get_connection():
    """Connect to the Claude sessions database.

    Configure via environment variables:
        RECALL_DB_HOST (default: localhost)
        RECALL_DB_PORT (default: 5433)
        RECALL_DB_NAME (default: connectingservices)
        RECALL_DB_USER (default: $USER)
        RECALL_DB_PASSWORD (default: empty)
    Or set RECALL_DATABASE_URL for a full connection string.
    """
    dsn = os.environ.get("RECALL_DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.environ.get("RECALL_DB_HOST", "localhost"),
        port=int(os.environ.get("RECALL_DB_PORT", "5433")),
        dbname=os.environ.get("RECALL_DB_NAME", "connectingservices"),
        user=os.environ.get("RECALL_DB_USER", os.environ.get("USER", "")),
        password=os.environ.get("RECALL_DB_PASSWORD", ""),
    )


def fetch_sessions(days: int = 7) -> list[SessionData]:
    """Fetch recent sessions with summaries and metadata."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    s.session_id,
                    s.cwd,
                    COUNT(*) FILTER (WHERE m.role = 'user') AS user_msg_count,
                    COUNT(*) AS total_msg_count,
                    MIN(m.timestamp) AS started,
                    MAX(m.timestamp) AS last_active,
                    MAX(m.timestamp) - MIN(m.timestamp) AS duration,
                    (SELECT model FROM claude_sessions.messages
                     WHERE session_id = s.session_id AND role = 'assistant' AND model IS NOT NULL
                     ORDER BY sequence_num DESC LIMIT 1) AS model,
                    ss.summary,
                    ss.user_requests,
                    ss.completed_work,
                    ss.topics,
                    fm.summary AS fm_summary
                FROM claude_sessions.messages m
                JOIN claude_sessions.sessions s ON s.session_id = m.session_id
                LEFT JOIN claude_sessions.session_summaries ss ON s.session_id = ss.session_id
                LEFT JOIN claude_sessions.first_messages fm ON s.session_id = fm.session_id
                WHERE m.timestamp >= NOW() - make_interval(days => %(days)s)
                  AND s.parent_session_id IS NULL
                GROUP BY s.session_id, ss.summary, ss.user_requests, ss.completed_work,
                         ss.topics, fm.summary
                HAVING COUNT(*) FILTER (WHERE m.role = 'user') >= 2
                ORDER BY MAX(m.timestamp) DESC;
            """, {"days": days})
            rows = cur.fetchall()

            # Fetch goals from first user messages
            if rows:
                session_ids = [r["session_id"] for r in rows]
                cur.execute("""
                    SELECT session_id, sequence_num, content
                    FROM claude_sessions.messages
                    WHERE session_id = ANY(%(ids)s)
                      AND role = 'user'
                      AND sequence_num <= 5
                    ORDER BY session_id, sequence_num ASC;
                """, {"ids": session_ids})
                msg_rows = cur.fetchall()

                # Fetch top tools per session
                cur.execute(r"""
                    SELECT session_id,
                        (regexp_matches(content, '"name":\s*"([^"]+)"', 'g'))[1] AS tool_name
                    FROM claude_sessions.messages
                    WHERE session_id = ANY(%(ids)s)
                      AND role = 'assistant'
                      AND content LIKE '%%"type": "tool_use"%%'
                    ORDER BY session_id;
                """, {"ids": session_ids})
                tool_rows = cur.fetchall()
            else:
                msg_rows = []
                tool_rows = []
    finally:
        conn.close()

    # Build goal lookup
    session_msgs: dict[str, list[str]] = defaultdict(list)
    for mr in msg_rows:
        session_msgs[mr["session_id"]].append(mr["content"])

    # Build tool counts per session
    session_tools: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tr in tool_rows:
        if tr["tool_name"]:
            session_tools[tr["session_id"]][tr["tool_name"]] += 1

    results = []
    for row in rows:
        sid = row["session_id"]

        # Pick goal from first non-noise user message
        goal = ""
        for msg_content in session_msgs.get(sid, []):
            candidate = clean_goal(msg_content)
            if candidate:
                goal = candidate
                break
        if not goal:
            goal = "(no readable goal)"

        # Use session_summaries if available, else first_messages summary
        summary = row["summary"] or row["fm_summary"] or ""
        user_requests = row["user_requests"] or ""
        completed_work = row["completed_work"] or ""
        topics = row["topics"] if isinstance(row["topics"], list) else []

        # Top 3 tools by usage
        tool_counts = session_tools.get(sid, {})
        # Filter out internal tools
        skip_tools = {"Read", "Write", "Glob", "Grep"}
        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])
        top_tools = [t[0] for t in top_tools[:5]]

        results.append(SessionData(
            session_id=sid,
            project=derive_project(row["cwd"] or ""),
            cwd=row["cwd"] or "",
            goal=goal,
            summary=summary,
            user_requests=user_requests,
            completed_work=completed_work,
            topics=topics,
            user_msg_count=row["user_msg_count"],
            total_msg_count=row["total_msg_count"],
            started=row["started"],
            last_active=row["last_active"],
            duration=row["duration"],
            model=format_model(row.get("model")),
            top_tools=top_tools,
        ))

    return results


def fetch_session_messages(session_id: str) -> list[dict]:
    """Fetch all messages for a session."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, sequence_num, timestamp, model
                FROM claude_sessions.messages
                WHERE session_id = %(sid)s
                ORDER BY sequence_num ASC;
            """, {"sid": session_id})
            return cur.fetchall()
    finally:
        conn.close()


def fetch_sessions_needing_summaries(days: int = 7, limit: int = 20) -> list[dict]:
    """Find sessions that have no summary yet."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT s.session_id
                FROM claude_sessions.sessions s
                JOIN claude_sessions.messages m ON s.session_id = m.session_id
                LEFT JOIN claude_sessions.session_summaries ss ON s.session_id = ss.session_id
                WHERE m.timestamp >= NOW() - make_interval(days => %(days)s)
                  AND s.parent_session_id IS NULL
                  AND ss.session_id IS NULL
                GROUP BY s.session_id
                HAVING COUNT(*) FILTER (WHERE m.role = 'user') >= 2
                ORDER BY MAX(m.timestamp) DESC
                LIMIT %(limit)s;
            """, {"days": days, "limit": limit})
            return cur.fetchall()
    finally:
        conn.close()


def fetch_sessions_needing_embeddings(days: int = 7, limit: int = 50) -> list[dict]:
    """Find sessions with summaries but no embeddings."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ss.session_id, ss.summary, ss.user_requests, ss.topics
                FROM claude_sessions.session_summaries ss
                JOIN claude_sessions.sessions s ON s.session_id = ss.session_id
                LEFT JOIN claude_sessions.session_embeddings se ON ss.session_id = se.session_id
                WHERE s.start_time >= NOW() - make_interval(days => %(days)s)
                  AND s.parent_session_id IS NULL
                  AND se.session_id IS NULL
                  AND ss.summary IS NOT NULL
                LIMIT %(limit)s;
            """, {"days": days, "limit": limit})
            return cur.fetchall()
    finally:
        conn.close()


def save_summary(session_id: str, summary_data: dict) -> None:
    """Insert or update a session summary."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO claude_sessions.session_summaries
                    (session_id, summary, user_requests, completed_work, topics, model, generated_at)
                VALUES (%(sid)s, %(summary)s, %(user_requests)s, %(completed_work)s,
                        %(topics)s::jsonb, %(model)s, NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    user_requests = EXCLUDED.user_requests,
                    completed_work = EXCLUDED.completed_work,
                    topics = EXCLUDED.topics,
                    model = EXCLUDED.model,
                    generated_at = NOW();
            """, {
                "sid": session_id,
                "summary": summary_data.get("summary", ""),
                "user_requests": summary_data.get("user_requests", ""),
                "completed_work": summary_data.get("completed_work", ""),
                "topics": json.dumps(summary_data.get("topics", [])),
                "model": "gemini-2.5-flash",
            })
        conn.commit()
    finally:
        conn.close()


def save_embedding(session_id: str, embedding: list[float], content_hash: str) -> None:
    """Insert or update a session embedding."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            vec_str = "[" + ",".join(str(f) for f in embedding) + "]"
            cur.execute("""
                INSERT INTO claude_sessions.session_embeddings
                    (session_id, embedding, content_hash, model)
                VALUES (%(sid)s, %(emb)s::vector, %(hash)s, 'text-embedding-004')
                ON CONFLICT (session_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    content_hash = EXCLUDED.content_hash;
            """, {"sid": session_id, "emb": vec_str, "hash": content_hash})
        conn.commit()
    finally:
        conn.close()


def search_embeddings(query_embedding: list[float], limit: int = 20) -> list[dict]:
    """Search session embeddings by cosine similarity."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            vec_str = "[" + ",".join(str(f) for f in query_embedding) + "]"
            cur.execute("""
                SELECT
                    se.session_id,
                    1 - (se.embedding <=> %(vec)s::vector) AS similarity,
                    ss.summary,
                    ss.user_requests,
                    ss.topics,
                    ss.detected_project,
                    s.cwd,
                    s.start_time,
                    (SELECT COUNT(*) FROM claude_sessions.messages
                     WHERE session_id = s.session_id AND role = 'user') AS user_msg_count
                FROM claude_sessions.session_embeddings se
                JOIN claude_sessions.sessions s ON se.session_id = s.session_id
                LEFT JOIN claude_sessions.session_summaries ss ON se.session_id = ss.session_id
                WHERE s.parent_session_id IS NULL
                ORDER BY se.embedding <=> %(vec)s::vector
                LIMIT %(limit)s;
            """, {"vec": vec_str, "limit": limit})
            return cur.fetchall()
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# LLM helpers (Gemini)
# ─────────────────────────────────────────────────────────────────────

def _get_google_api_key() -> str:
    """Get Google AI API key from GOOGLE_API_KEY env var or macOS Keychain.

    For Keychain, set the service name via RECALL_KEYCHAIN_GOOGLE env var
    (default: google-ai-api-key).
    """
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        return key
    keychain_service = os.environ.get("RECALL_KEYCHAIN_GOOGLE", "google-ai-api-key")
    try:
        key = subprocess.check_output(
            ["security", "find-generic-password", "-s", keychain_service, "-w"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return key


SUMMARY_PROMPT = """Analyze this Claude Code conversation and provide a structured summary.

CONVERSATION:
{conversation}

Respond in exactly this JSON format (no markdown, just raw JSON):
{{
    "summary": "One paragraph overview of what happened in this session",
    "user_requests": "Bullet list of what the user asked for (use \\n for newlines)",
    "completed_work": "Bullet list of what was actually accomplished (use \\n for newlines)",
    "topics": ["topic1", "topic2", "topic3"]
}}

TOPIC GUIDELINES:
- Extract 2-5 topic tags that categorize this conversation
- Topics should be lowercase, hyphenated if multi-word
- Be concise. Focus on the key points."""


def generate_summary(session_id: str) -> dict | None:
    """Generate a summary for a session using Gemini Flash."""
    from google import genai

    messages = fetch_session_messages(session_id)
    if not messages:
        return None

    conversation_parts = []
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "CLAUDE"
        content = msg["content"] or ""
        if len(content) > 2000:
            content = content[:2000] + "..."
        # Strip XML noise
        content = re.sub(
            r"<(system-reminder|task-notification|task-output)[^>]*>.*?</\1>",
            "", content, flags=re.S,
        )
        content = re.sub(r"<[^>]+>", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        if content:
            conversation_parts.append(f"{role}: {content}")

    conversation_text = "\n\n".join(conversation_parts)
    if len(conversation_text) > 50000:
        conversation_text = conversation_text[:50000] + "\n[TRUNCATED]"

    prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

    api_key = _get_google_api_key()
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = resp.text.strip()

    # Extract JSON
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

    try:
        result = json.loads(text)
        return {
            "summary": result.get("summary", ""),
            "user_requests": result.get("user_requests", ""),
            "completed_work": result.get("completed_work", ""),
            "topics": result.get("topics", []),
        }
    except json.JSONDecodeError:
        # Try to find JSON in the text
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                return {
                    "summary": result.get("summary", ""),
                    "user_requests": result.get("user_requests", ""),
                    "completed_work": result.get("completed_work", ""),
                    "topics": result.get("topics", []),
                }
            except json.JSONDecodeError:
                pass
    return None


def embed_text(text: str) -> list[float] | None:
    """Embed text using Google text-embedding-004."""
    from google import genai

    api_key = _get_google_api_key()
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return result.embeddings[0].values


# ─────────────────────────────────────────────────────────────────────
# Brief mode (plain text morning recap)
# ─────────────────────────────────────────────────────────────────────

def print_brief(days: int = 1) -> None:
    """Print a plain-text morning recap to stdout."""
    console = Console()
    sessions = fetch_sessions(days)
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    # Group by day
    by_day: dict[str, list[SessionData]] = defaultdict(list)
    for s in sessions:
        key = s.started.strftime("%Y-%m-%d")
        by_day[key].append(s)

    # Collect unique projects
    projects = set(s.project for s in sessions)

    for day_key in sorted(by_day.keys(), reverse=True):
        day_sessions = by_day[day_key]
        dt = day_sessions[0].started
        label = day_label(dt)
        day_projects = set(s.project for s in day_sessions)
        console.print(f"\n[bold]{label}[/bold] — {len(day_sessions)} sessions — {len(day_projects)} projects")
        console.print("─" * 50)

        for s in day_sessions:
            time_str = s.started.strftime("%H:%M")
            proj_color = get_project_color(s.project)
            summary_line = s.summary[:70] if s.summary else s.goal
            tools_str = ",".join(s.top_tools[:3]) if s.top_tools else ""
            console.print(
                f"  {time_str}  [{proj_color}]{s.project:12s}[/]  {summary_line}"
            )
            console.print(
                f"         {s.user_msg_count:>3} msgs  {s.model:8s}  {format_duration(s.duration):>6s}  {tools_str}",
                style="dim",
            )


# ─────────────────────────────────────────────────────────────────────
# TUI Widgets
# ─────────────────────────────────────────────────────────────────────

class SessionRow(Static):
    """A single session row in the timeline, expandable."""

    expanded: reactive[bool] = reactive(False)

    def __init__(self, session: SessionData, **kwargs):
        super().__init__(**kwargs)
        self.session = session
        self.can_focus = True

    def render(self) -> str:
        s = self.session
        time_str = s.started.strftime("%H:%M")
        proj = s.project
        dur = format_duration(s.duration)
        tools = ",".join(s.top_tools[:3]) if s.top_tools else ""
        arrow = "▶" if self.expanded else "▸"
        summary_line = s.summary[:60] if s.summary else s.goal[:60]

        lines = [
            f"{arrow} {time_str}  [{get_project_color(proj)}]{proj:12s}[/]  {summary_line}",
            f"           {s.user_msg_count:>3} msgs  {s.model:8s}  {dur:>6s}  {tools}",
        ]

        if self.expanded:
            lines.append("")
            if s.summary:
                lines.append(f"  [bold]Summary:[/] {s.summary}")
            if s.user_requests:
                lines.append(f"  [bold]Requests:[/] {s.user_requests[:200]}")
            if s.completed_work:
                lines.append(f"  [bold]Completed:[/] {s.completed_work[:200]}")
            if s.topics:
                tags = " ".join(f"[{t}]" for t in s.topics)
                lines.append(f"  [bold]Topics:[/] {tags}")
            lines.append(f"  [dim]\\[r] Resume  \\[m] Messages  \\[c] Copy ID[/]")
            lines.append("")

        return "\n".join(lines)

    def toggle_expanded(self) -> None:
        self.expanded = not self.expanded


class DayGroup(Static):
    """A day header in the timeline."""

    def __init__(self, label: str, count: int, projects: int, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.count = count
        self.projects = projects

    def render(self) -> str:
        return (
            f"[bold]{self.label}[/bold] — {self.count} sessions — {self.projects} projects\n"
            f"{'─' * 60}"
        )


class SearchResultRow(Static):
    """A search result row."""

    def __init__(self, result: dict, **kwargs):
        super().__init__(**kwargs)
        self.result = result
        self.can_focus = True

    def render(self) -> str:
        r = self.result
        sim = r.get("similarity", 0)
        date = r.get("start_time")
        date_str = date.strftime("%b %d") if date else "?"
        proj = derive_project(r.get("cwd", ""))
        summary = (r.get("summary") or "")[:70]
        proj_color = get_project_color(proj)
        return (
            f"[bold]{sim:.2f}[/]  {date_str}  [{proj_color}]{proj:12s}[/]  {summary}"
        )


class ProjectGroup(Static):
    """A project in the projects view."""

    def __init__(self, project: str, sessions: list[SessionData], **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.sessions = sessions
        self.can_focus = True

    def render(self) -> str:
        p = self.project
        color = get_project_color(p)
        count = len(self.sessions)

        # Activity sparkline (last 7 days)
        today = datetime.now(timezone.utc).date()
        day_counts = [0] * 7
        for s in self.sessions:
            day_idx = (today - s.started.date()).days
            if 0 <= day_idx < 7:
                day_counts[6 - day_idx] += 1
        spark_chars = " ▁▂▃▄▅▆▇█"
        max_c = max(day_counts) if any(day_counts) else 1
        sparkline = "".join(
            spark_chars[min(int(c / max_c * 8), 8)] if c > 0 else spark_chars[0]
            for c in day_counts
        )

        # Topic cloud
        all_topics: dict[str, int] = defaultdict(int)
        for s in self.sessions:
            for t in s.topics:
                all_topics[t] += 1
        top_topics = sorted(all_topics.items(), key=lambda x: -x[1])[:5]
        topic_str = " ".join(f"[{t}]" for t, _ in top_topics) if top_topics else ""

        # Recent sessions
        recent = self.sessions[:3]
        recent_lines = []
        for s in recent:
            time_str = s.started.strftime("%b %d %H:%M")
            summary = (s.summary or s.goal)[:50]
            recent_lines.append(f"    {time_str}  {summary}")

        lines = [
            f"[{color} bold]{p}[/]  {count} sessions  {sparkline}",
        ]
        if topic_str:
            lines.append(f"  {topic_str}")
        lines.extend(recent_lines)
        lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Main TUI App
# ─────────────────────────────────────────────────────────────────────

class RecallApp(App):
    """Claude Code Session History TUI."""

    CSS = """
    Screen {
        background: $surface;
    }

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 0 1;
    }

    #timeline-scroll {
        height: 1fr;
    }

    #search-scroll {
        height: 1fr;
    }

    #projects-scroll {
        height: 1fr;
    }

    SessionRow {
        padding: 0 1;
        margin: 0;
    }

    SessionRow:focus {
        background: $accent 20%;
    }

    SessionRow:hover {
        background: $accent 10%;
    }

    DayGroup {
        padding: 1 1 0 1;
        color: $text;
    }

    SearchResultRow {
        padding: 0 1;
    }

    SearchResultRow:focus {
        background: $accent 20%;
    }

    ProjectGroup {
        padding: 0 1;
    }

    ProjectGroup:focus {
        background: $accent 20%;
    }

    #search-input {
        dock: top;
        margin: 1;
    }

    #status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $primary-background;
        color: $text-muted;
    }

    .dim {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "focus_search", "Search", key_display="/"),
        Binding("r", "resume_session", "Resume", show=False),
        Binding("c", "copy_session_id", "Copy ID", show=False),
        Binding("m", "show_messages", "Messages", show=False),
        Binding("enter", "toggle_expand", "Expand", show=False),
        Binding("1", "jump_day(1)", "1d", show=False),
        Binding("2", "jump_day(2)", "2d", show=False),
        Binding("3", "jump_day(3)", "3d", show=False),
        Binding("4", "jump_day(4)", "4d", show=False),
        Binding("5", "jump_day(5)", "5d", show=False),
        Binding("6", "jump_day(6)", "6d", show=False),
        Binding("7", "jump_day(7)", "7d", show=False),
    ]

    TITLE = "recall"

    def __init__(self, days: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.days = days
        self.sessions: list[SessionData] = []
        self.search_results: list[dict] = []
        self._summary_worker_count = 0
        self._embedding_worker_count = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent("Timeline", "Search", "Projects"):
            with TabPane("Timeline", id="timeline-tab"):
                yield VerticalScroll(id="timeline-scroll")
            with TabPane("Search", id="search-tab"):
                yield Input(placeholder="Search sessions semantically...", id="search-input")
                yield VerticalScroll(id="search-scroll")
            with TabPane("Projects", id="projects-tab"):
                yield VerticalScroll(id="projects-scroll")
        yield Static("Loading...", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.load_sessions()

    @work(thread=True)
    def load_sessions(self) -> None:
        """Load sessions from DB in a worker thread."""
        try:
            self.sessions = fetch_sessions(self.days)
            self.call_from_thread(self._render_timeline)
            self.call_from_thread(self._render_projects)
            self.call_from_thread(
                self._update_status,
                f"{len(self.sessions)} sessions loaded"
            )
        except Exception as e:
            self.call_from_thread(self._update_status, f"Error: {e}")
            return

        # Start background workers
        self.call_from_thread(self._start_background_workers)

    def _start_background_workers(self) -> None:
        self.backfill_summaries()
        self.backfill_embeddings()

    @work(thread=True)
    def backfill_summaries(self) -> None:
        """Background worker: generate missing summaries."""
        try:
            needed = fetch_sessions_needing_summaries(self.days, limit=10)
        except Exception:
            return

        if not needed:
            return

        self.call_from_thread(
            self._update_status,
            f"Generating {len(needed)} summaries..."
        )

        for i, row in enumerate(needed):
            sid = row["session_id"]
            try:
                summary_data = generate_summary(sid)
                if summary_data:
                    save_summary(sid, summary_data)
                    self._summary_worker_count += 1
                    # Update the in-memory session if it exists
                    for s in self.sessions:
                        if s.session_id == sid:
                            s.summary = summary_data.get("summary", "")
                            s.user_requests = summary_data.get("user_requests", "")
                            s.completed_work = summary_data.get("completed_work", "")
                            s.topics = summary_data.get("topics", [])
                            break
                    self.call_from_thread(self._render_timeline)
                    self.call_from_thread(
                        self._update_status,
                        f"Generated {self._summary_worker_count} summaries ({i+1}/{len(needed)})"
                    )
            except Exception:
                continue

        self.call_from_thread(
            self._update_status,
            f"Done: {self._summary_worker_count} summaries generated"
        )

    @work(thread=True)
    def backfill_embeddings(self) -> None:
        """Background worker: generate missing embeddings."""
        try:
            needed = fetch_sessions_needing_embeddings(self.days, limit=50)
        except Exception:
            return

        if not needed:
            return

        for row in needed:
            sid = row["session_id"]
            try:
                # Build content to embed
                parts = [row.get("summary") or ""]
                if row.get("user_requests"):
                    parts.append(row["user_requests"])
                topics = row.get("topics")
                if isinstance(topics, list):
                    parts.append(" ".join(topics))
                elif isinstance(topics, str):
                    parts.append(topics)

                content = " ".join(p for p in parts if p)
                if not content.strip():
                    continue

                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                embedding = embed_text(content)
                if embedding:
                    save_embedding(sid, embedding, content_hash)
                    self._embedding_worker_count += 1
            except Exception:
                continue

        if self._embedding_worker_count:
            self.call_from_thread(
                self._update_status,
                f"Embedded {self._embedding_worker_count} sessions"
            )

    def _render_timeline(self) -> None:
        """Render the timeline view."""
        try:
            scroll = self.query_one("#timeline-scroll", VerticalScroll)
        except NoMatches:
            return
        scroll.remove_children()

        # Group by day
        by_day: dict[str, list[SessionData]] = defaultdict(list)
        for s in self.sessions:
            key = s.started.strftime("%Y-%m-%d")
            by_day[key].append(s)

        for day_key in sorted(by_day.keys(), reverse=True):
            day_sessions = by_day[day_key]
            dt = day_sessions[0].started
            label = day_label(dt)
            projects = len(set(s.project for s in day_sessions))

            scroll.mount(DayGroup(label, len(day_sessions), projects))
            for s in day_sessions:
                scroll.mount(SessionRow(s))

    def _render_projects(self) -> None:
        """Render the projects view."""
        try:
            scroll = self.query_one("#projects-scroll", VerticalScroll)
        except NoMatches:
            return
        scroll.remove_children()

        by_project: dict[str, list[SessionData]] = defaultdict(list)
        for s in self.sessions:
            by_project[s.project].append(s)

        # Sort by session count descending
        for proj in sorted(by_project.keys(), key=lambda p: -len(by_project[p])):
            scroll.mount(ProjectGroup(proj, by_project[proj]))

    def _render_search_results(self) -> None:
        """Render search results."""
        try:
            scroll = self.query_one("#search-scroll", VerticalScroll)
        except NoMatches:
            return
        scroll.remove_children()

        if not self.search_results:
            scroll.mount(Static("No results. Type a query and press Enter.", classes="dim"))
            return

        for r in self.search_results:
            scroll.mount(SearchResultRow(r))

    def _update_status(self, text: str) -> None:
        try:
            bar = self.query_one("#status-bar", Static)
            bar.update(text)
        except NoMatches:
            pass

    # ── Input handling ──

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input."""
        if event.input.id == "search-input":
            query = event.value.strip()
            if query:
                self.run_search(query)

    @work(thread=True)
    def run_search(self, query: str) -> None:
        """Run semantic search."""
        self.call_from_thread(self._update_status, f"Searching: {query}...")
        try:
            query_embedding = embed_text(query)
            if not query_embedding:
                self.call_from_thread(self._update_status, "Error: Could not embed query (API key missing?)")
                return

            results = search_embeddings(query_embedding, limit=20)
            self.search_results = results
            self.call_from_thread(self._render_search_results)
            self.call_from_thread(
                self._update_status,
                f"Found {len(results)} results for: {query}"
            )
        except Exception as e:
            self.call_from_thread(self._update_status, f"Search error: {e}")

    # ── Actions ──

    def _get_focused_session(self) -> SessionData | None:
        """Get the session from the currently focused widget."""
        focused = self.focused
        if isinstance(focused, SessionRow):
            return focused.session
        if isinstance(focused, SearchResultRow):
            # Convert search result to minimal session data for resume
            r = focused.result
            return SessionData(
                session_id=r["session_id"],
                project=derive_project(r.get("cwd", "")),
                cwd=r.get("cwd", ""),
                goal="",
                summary=r.get("summary", ""),
                user_requests="",
                completed_work="",
                topics=[],
                user_msg_count=r.get("user_msg_count", 0),
                total_msg_count=0,
                started=r.get("start_time", datetime.now(timezone.utc)),
                last_active=datetime.now(timezone.utc),
                duration=None,
                model="",
            )
        return None

    def action_toggle_expand(self) -> None:
        focused = self.focused
        if isinstance(focused, SessionRow):
            focused.toggle_expanded()

    def action_resume_session(self) -> None:
        session = self._get_focused_session()
        if session:
            self.exit()
            os.execvp("claude", ["claude", "--resume", session.session_id])

    def action_copy_session_id(self) -> None:
        session = self._get_focused_session()
        if session:
            try:
                subprocess.run(
                    ["pbcopy"],
                    input=session.session_id.encode(),
                    check=True,
                )
                self._update_status(f"Copied: {session.session_id}")
            except Exception:
                self._update_status(f"ID: {session.session_id}")

    def action_show_messages(self) -> None:
        session = self._get_focused_session()
        if session:
            self.push_screen(MessageScreen(session.session_id))

    def action_focus_search(self) -> None:
        try:
            tabs = self.query_one(TabbedContent)
            tabs.active = "search-tab"
            inp = self.query_one("#search-input", Input)
            inp.focus()
        except NoMatches:
            pass

    def action_jump_day(self, n: int) -> None:
        """Scroll to N days ago in the timeline."""
        target_date = (datetime.now(timezone.utc) - timedelta(days=n - 1)).strftime("%Y-%m-%d")
        try:
            scroll = self.query_one("#timeline-scroll", VerticalScroll)
            for child in scroll.children:
                if isinstance(child, DayGroup) and target_date in child.label:
                    child.scroll_visible()
                    break
        except NoMatches:
            pass


class MessageScreen(Screen):
    """Screen showing full message transcript."""

    BINDINGS = [
        Binding("q", "go_back", "Back"),
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(
            Static(f"Loading messages for {self.session_id}...", id="messages-content"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.load_messages()

    @work(thread=True)
    def load_messages(self) -> None:
        messages = fetch_session_messages(self.session_id)
        lines = [f"[bold]Session {self.session_id}[/bold] — {len(messages)} messages\n"]

        for msg in messages:
            role = msg["role"].upper()
            ts = msg["timestamp"].strftime("%H:%M:%S") if msg.get("timestamp") else ""
            content = msg.get("content", "") or ""
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "…"
            # Strip XML
            content = re.sub(r"<[^>]+>", "", content)
            content = re.sub(r"\s+", " ", content).strip()

            if role == "USER":
                lines.append(f"[bold cyan]{ts} USER:[/] {content}\n")
            else:
                lines.append(f"[dim]{ts} {role}:[/] {content}\n")

        text = "\n".join(lines)
        self.call_from_thread(self._set_content, text)

    def _set_content(self, text: str) -> None:
        try:
            widget = self.query_one("#messages-content", Static)
            widget.update(text)
        except NoMatches:
            pass

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="recall — Claude Code Session History TUI")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--brief", action="store_true", help="Plain-text morning recap to stdout")
    args = parser.parse_args()

    if args.brief:
        print_brief(args.days)
    else:
        app = RecallApp(days=args.days)
        app.run()


if __name__ == "__main__":
    main()
