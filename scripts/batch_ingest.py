"""
scripts/batch_ingest.py
------------------------
CLI for batch flashcard generation using the Anthropic Messages Batch API.

Run with --help for usage:
    python -m scripts.batch_ingest --help
    python -m scripts.batch_ingest submit --help
    python -m scripts.batch_ingest status --help
    python -m scripts.batch_ingest list --help
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import click

from core.batch_client import BatchClient, CollectResult
from core.database import BatchJob, SessionLocal, Subject

# ---------------------------------------------------------------------------
# Question type registry
# ---------------------------------------------------------------------------

_ALL_QTYPES = [
    "active_recall",
    "fill_blank",
    "short_answer",
    "long_answer",
    "numerical",
    "scenario",
]

_VALID_QTYPES = set(_ALL_QTYPES)

# Tolerant aliases: "numericals", "recall", "fill" etc. all accepted
_QTYPE_ALIASES: dict[str, str] = {
    "numericals":    "numerical",
    "scenarios":     "scenario",
    "recall":        "active_recall",
    "active":        "active_recall",
    "fill":          "fill_blank",
    "fill_in_blank": "fill_blank",
    "short":         "short_answer",
    "long":          "long_answer",
}

_QTYPE_HELP = (
    "\b\n"
    "Supported values:\n"
    "  active_recall  core concepts and definitions\n"
    "  fill_blank     fill-in-the-blank sentence\n"
    "  short_answer   2-3 sentence synthesis\n"
    "  long_answer    multi-part structured answer\n"
    "  numerical      novel math / derivation problems\n"
    "  scenario       what-if / case study\n\n"
    "Aliases accepted (e.g. 'numericals' -> 'numerical'):\n"
    "  recall, active, fill, fill_in_blank, short, long,\n"
    "  numericals, scenarios\n\n"
    "Default: all six types."
)


def _normalize_qtypes(raw: list[str]) -> list[str]:
    """Validate and alias-resolve a list of raw qtype strings. Exits on unknown."""
    result: list[str] = []
    unknown: list[str] = []
    for q in raw:
        q = q.strip().lower()
        if q in _VALID_QTYPES:
            result.append(q)
        elif q in _QTYPE_ALIASES:
            resolved = _QTYPE_ALIASES[q]
            click.echo(f"  Note: '{q}' resolved to '{resolved}'")
            result.append(resolved)
        else:
            unknown.append(q)
    if unknown:
        raise click.ClickException(
            f"Unknown question type(s): {unknown}\n"
            f"Valid types:   {_ALL_QTYPES}\n"
            f"Aliases known: {sorted(_QTYPE_ALIASES)}"
        )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Config file not found: {config_path}")
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_subject(db, name: str, create: bool) -> int:
    """Return subject id matched case-insensitively. Create if create=True."""
    match = (
        db.query(Subject)
        .filter(Subject.name.ilike(name), Subject.is_archived.is_(False))
        .first()
    )
    if match:
        return match.id
    if create:
        subj = Subject(name=name)
        db.add(subj)
        db.commit()
        db.refresh(subj)
        click.echo(f"Created subject: {subj.name!r} (id={subj.id})")
        return subj.id
    available = [s.name for s in db.query(Subject).filter(Subject.is_archived.is_(False)).all()]
    raise click.ClickException(
        f"Subject {name!r} not found.\n"
        f"Available: {available}\n"
        "Use --create-subject to create it automatically."
    )


def _fmt_eta(eta_seconds: Optional[float]) -> str:
    if eta_seconds is None:
        return "ETA unknown"
    minutes = int(eta_seconds // 60)
    secs = int(eta_seconds % 60)
    return f"~{minutes}m {secs}s remaining (est.)"


def _print_result(result: CollectResult) -> None:
    if result.status == "completed":
        click.echo(
            f"  COMPLETE — {result.flashcards_created} card(s) created, "
            f"{result.flashcards_rejected} rejected"
        )
    elif result.status == "in_progress":
        pct = (
            f"{result.completed_requests}/{result.total_requests}"
            if result.total_requests
            else "?"
        )
        click.echo(f"  IN PROGRESS — {pct} requests done | {_fmt_eta(result.eta_seconds)}")
    elif result.status == "pending":
        click.echo("  PENDING — not yet started by Anthropic")
    else:
        click.echo(f"  {result.status.upper()}{' — ' + result.error if result.error else ''}")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

_GROUP_HELP = """\
Offline batch flashcard generation using the Anthropic Messages Batch API.

\b
HOW IT WORKS
  1. 'submit' indexes your PDFs (if not already in the library), builds all
     flashcard generation requests, submits them as a single Anthropic batch,
     and exits immediately. The process does NOT stay running.

  2. 'status' checks whether the batch has finished. When complete it downloads
     results, runs the critic, and saves cards to the database. Safe to run
     multiple times — already-processed requests are skipped.

  3. 'list' shows every job ever submitted so you can track history.

\b
QUICK START
  python -m scripts.batch_ingest submit \\
      --folder ./documents \\
      --subject "Engineering Statistics" \\
      --create-subject

  python -m scripts.batch_ingest status

\b
COMMANDS
  submit  Index PDFs, submit batch job, exit (options: --folder, --subject,
            --qtypes, --config, --create-subject)
  status  Poll Anthropic; collect cards when done (option: --job-id)
  list    Show all jobs with status and card counts (options: --filter,
            --last)

Use '<command> --help' for full option details.
"""


@click.group(help=_GROUP_HELP)
def cli():
    pass


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

_SUBMIT_HELP = """\
Index PDFs if needed, submit a batch generation job, then exit immediately.

\b
INDEXING PHASE
  Each PDF is checked against the library by content hash. If not present it
  runs the full LangGraph indexing pipeline (topic extraction, chunking,
  Qdrant embedding). You will see LLM routing logs like:
    [model_hop] tier='fast' -> anthropic/claude-sonnet-4-6
  This is normal — the pipeline selects models automatically. Wait for the
  '✓ Indexed' confirmation before the process moves on.

\b
SUBMISSION PHASE
  Once all PDFs are indexed, one batch request is built per chunk per question
  type and submitted to Anthropic. The job ID and Anthropic batch ID are
  printed — save these. Then the process exits. Use 'status' to collect
  results when ready.

\b
CONFIG FILE
  All options can be provided in a YAML file instead of CLI flags:

    folder: ./documents
    subject: Engineering Statistics
    create_subject: true
    question_types:
      - active_recall
      - numerical

\b
EXAMPLES
  # All six question types (default), create subject if missing
  python -m scripts.batch_ingest submit \\
      --folder ./documents --subject "Physics" --create-subject

  # Only two question types
  python -m scripts.batch_ingest submit \\
      --folder ./docs --subject "Physics" --qtypes active_recall,numerical

  # Via config file
  python -m scripts.batch_ingest submit --config batch_config.yaml
"""


@cli.command(help=_SUBMIT_HELP)
@click.option(
    "--folder", "-f",
    default=None,
    metavar="PATH",
    help="Folder containing PDF files to process. All *.pdf files in this folder are included.",
)
@click.option(
    "--subject", "-s",
    default=None,
    metavar="NAME",
    help=(
        "Subject name to link generated flashcards to. "
        "Must already exist unless --create-subject is set. "
        "Matched case-insensitively."
    ),
)
@click.option(
    "--qtypes",
    default=None,
    metavar="TYPE[,TYPE...]",
    help=_QTYPE_HELP,
)
@click.option(
    "--config", "-c",
    "config_file",
    default=None,
    metavar="FILE",
    help=(
        "Path to a YAML config file. Accepted keys: folder, subject, "
        "question_types (list), create_subject (bool). "
        "CLI flags override config file values."
    ),
)
@click.option(
    "--create-subject",
    is_flag=True,
    default=False,
    help="Create the subject if it does not already exist in the database.",
)
def submit(folder, subject, qtypes, config_file, create_subject):
    cfg = _load_yaml_config(config_file)

    folder = folder or cfg.get("folder")
    subject = subject or cfg.get("subject")
    create_subject = create_subject or cfg.get("create_subject", False)

    # Resolve question types (CLI wins over config file)
    if qtypes:
        raw_list = [q.strip() for q in qtypes.split(",") if q.strip()]
    elif cfg.get("question_types"):
        raw = cfg["question_types"]
        raw_list = raw if isinstance(raw, list) else [q.strip() for q in str(raw).split(",")]
    else:
        raw_list = None

    question_types = _normalize_qtypes(raw_list) if raw_list else list(_ALL_QTYPES)

    if not folder:
        raise click.ClickException("--folder is required (or set 'folder' in config file).")
    if not subject:
        raise click.ClickException("--subject is required (or set 'subject' in config file).")

    folder_path = Path(folder)
    if not folder_path.exists():
        raise click.ClickException(f"Folder not found: {folder}")

    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        raise click.ClickException(f"No PDF files found in {folder}")

    click.echo(f"Folder:         {folder_path.resolve()}")
    click.echo(f"PDFs found:     {len(pdf_files)}")
    click.echo(f"Subject:        {subject!r}")
    click.echo(f"Question types: {question_types}")

    with SessionLocal() as db:
        subject_id = _resolve_subject(db, subject, create_subject)

    client = BatchClient()

    # Step 1 — index each PDF
    doc_ids: list[str] = []
    total_steps = len(pdf_files) + 1
    for i, pdf in enumerate(pdf_files, 1):
        click.echo(f"\n[Step {i}/{total_steps}] Indexing {pdf.name}…")
        click.echo("  LLM routing logs below are normal — please wait.")
        doc_id = client.index_pdf_if_needed(pdf, subject_id)
        click.echo(f"  ✓ Done: {pdf.name} (doc_id={doc_id[:8]}…)")
        doc_ids.append(doc_id)

    # Step 2 — build and submit
    job_id = str(uuid.uuid4())
    click.echo(f"\n[Step {total_steps}/{total_steps}] Building batch requests…")
    requests = client.build_requests(job_id, doc_ids, subject_id, question_types)

    if not requests:
        click.echo("Nothing to submit — all subtopics already have cards for this subject.")
        return

    with SessionLocal() as db:
        db.add(
            BatchJob(
                id=job_id,
                subject_id=subject_id,
                status="indexing",
                doc_ids=json.dumps(doc_ids),
                question_types=json.dumps(question_types),
                request_count=len(requests),
            )
        )
        db.commit()

    click.echo(f"  Submitting {len(requests)} request(s) to Anthropic…")
    anthropic_batch_id = client.submit(job_id, requests)

    click.echo("")
    click.echo("=" * 64)
    click.echo(f"  Job ID           {job_id}")
    click.echo(f"  Anthropic batch  {anthropic_batch_id}")
    click.echo(f"  Requests         {len(requests)}")
    click.echo(f"  ETA              < 1 hour typical (24 h max)")
    click.echo("=" * 64)
    click.echo("Run:  python -m scripts.batch_ingest status")
    click.echo("      to collect results when complete.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

_STATUS_HELP = """\
Check batch job status and collect results when complete.

\b
BEHAVIOUR
  Without --job-id: checks all jobs in 'submitted', 'collecting', or
  'indexing' (interrupted) state.

  With --job-id: checks that specific job regardless of status.

  When a batch has finished processing, results are downloaded automatically,
  the critic runs on each card, and flashcards are saved to the database.
  Cards appear in Mentor Review (or are auto-approved if AUTO_ACCEPT_CONTENT
  is set). Safe to run multiple times — already-collected requests are skipped.

\b
JOB STATES
  indexing    Interrupted before submission. Re-run 'submit' to retry.
  submitted   Waiting for Anthropic to process. Run 'status' to poll.
  collecting  Downloading results (brief transient state).
  completed   All results collected and saved to the database.
  failed      Submission or collection error (see 'list' for details).

\b
EXAMPLES
  python -m scripts.batch_ingest status
  python -m scripts.batch_ingest status --job-id a1b2c3d4-...
"""


@cli.command(help=_STATUS_HELP)
@click.option(
    "--job-id",
    default=None,
    metavar="UUID",
    help=(
        "UUID of the specific job to check. "
        "If omitted, all active jobs (indexing / submitted / collecting) are checked."
    ),
)
def status(job_id):
    with SessionLocal() as db:
        if job_id:
            jobs = db.query(BatchJob).filter(BatchJob.id == job_id).all()
        else:
            jobs = (
                db.query(BatchJob)
                .filter(BatchJob.status.in_(["indexing", "submitted", "collecting"]))
                .order_by(BatchJob.created_at.desc())
                .all()
            )
        jobs_snapshot = [
            {
                "id": j.id,
                "anthropic_batch_id": j.anthropic_batch_id,
                "status": j.status,
                "request_count": j.request_count,
                "question_types": j.question_types,
            }
            for j in jobs
        ]

    if not jobs_snapshot:
        click.echo("No active batch jobs. Run 'list' to see all jobs.")
        return

    client = BatchClient()
    for job in jobs_snapshot:
        qtypes = ", ".join(json.loads(job["question_types"] or "[]"))
        click.echo(
            f"\nJob {job['id'][:8]}…  "
            f"[{job['status']}]  "
            f"{job['request_count']} request(s)  "
            f"types: {qtypes}"
        )
        click.echo(f"  Anthropic batch: {job['anthropic_batch_id'] or '—'}")
        if job["status"] == "indexing":
            click.echo("  INTERRUPTED — indexing did not complete. Re-run 'submit' to retry.")
        else:
            result = client.collect(job["id"])
            _print_result(result)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

_LIST_HELP = """\
List batch jobs with their status and card counts.

\b
STATUS VALUES
  indexing    Interrupted before submission.
  submitted   Waiting for Anthropic.
  collecting  Downloading results.
  completed   Done — cards saved to database.
  failed      Error occurred.

\b
EXAMPLES
  python -m scripts.batch_ingest list
  python -m scripts.batch_ingest list --filter submitted
  python -m scripts.batch_ingest list --last 5
"""

_STATUS_CHOICES = click.Choice(
    ["indexing", "submitted", "collecting", "completed", "failed"],
    case_sensitive=False,
)


@cli.command(name="list", help=_LIST_HELP)
@click.option(
    "--filter", "status_filter",
    default=None,
    metavar="STATUS",
    type=_STATUS_CHOICES,
    help=(
        "Show only jobs with this status. "
        "One of: indexing, submitted, collecting, completed, failed."
    ),
)
@click.option(
    "--last",
    default=None,
    metavar="N",
    type=int,
    help="Show only the N most recent jobs.",
)
def list_jobs(status_filter, last):
    with SessionLocal() as db:
        q = db.query(BatchJob).order_by(BatchJob.created_at.desc())
        if status_filter:
            q = q.filter(BatchJob.status == status_filter.lower())
        if last:
            q = q.limit(last)
        jobs = q.all()

    if not jobs:
        msg = "No batch jobs found."
        if status_filter:
            msg = f"No jobs with status '{status_filter}'."
        click.echo(msg)
        return

    click.echo(
        f"\n{'JOB ID':38} {'STATUS':12} {'REQS':6} {'CARDS':6}  {'CREATED'}"
    )
    click.echo("-" * 88)
    for j in jobs:
        created = j.created_at.strftime("%Y-%m-%d %H:%M") if j.created_at else ""
        click.echo(
            f"{str(j.id):38} {j.status:12} {j.request_count or 0:6} "
            f"{j.completed_count or 0:6}  {created}"
        )


if __name__ == "__main__":
    cli()
