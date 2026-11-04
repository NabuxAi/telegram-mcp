"""Background worker that processes queued channel-import jobs.

Long channel imports run in a separate thread so the bot's polling loop stays
responsive. Each job tracks progress and a resume cursor in the database, so an
import that is interrupted (crash/restart) is requeued and continues where it
left off instead of starting over.

The worker is deliberately decoupled from Telegram: it is given a ``runner``
callable that performs the actual import. This keeps the queue/state machine
fully unit-testable with a fake runner.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from .db import Repository

logger = logging.getLogger(__name__)

# runner(job, on_progress, is_cancelled) -> None; raises on failure.
#   on_progress(processed: int, total: int | None, cursor: int | None)
#   is_cancelled() -> bool
Runner = Callable[[dict, Callable[[int, "int | None", "int | None"], None], Callable[[], bool]], None]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobWorker(threading.Thread):
    def __init__(self, repository: Repository, runner: Runner, *, poll_interval: float = 2.0) -> None:
        super().__init__(daemon=True, name="job-worker")
        self.repository = repository
        self.runner = runner
        self.poll_interval = poll_interval
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        requeued = self.repository.requeue_running_jobs(updated_at=_now())
        if requeued:
            logger.info("Requeued %s interrupted job(s) on startup", requeued)
        while not self._stop.is_set():
            try:
                processed = self.process_one()
            except Exception:
                logger.exception("Job worker loop error")
                processed = False
            if not processed:
                self._stop.wait(self.poll_interval)

    def process_one(self) -> bool:
        """Claim and run a single queued job. Returns True if one was handled."""
        job = self.repository.claim_next_queued_job(updated_at=_now())
        if not job:
            return False

        job_id = job["id"]

        def on_progress(processed: int, total: int | None, cursor: int | None) -> None:
            self.repository.update_job_progress(
                job_id=job_id, processed=processed, total=total, cursor=cursor, updated_at=_now()
            )

        def is_cancelled() -> bool:
            return self.repository.is_cancel_requested(job_id=job_id)

        try:
            self.runner(job, on_progress, is_cancelled)
        except Exception as exc:
            logger.exception("Import job %s failed", job_id)
            self.repository.finish_job(job_id=job_id, status="failed", error=str(exc), updated_at=_now())
            return True

        status = "cancelled" if is_cancelled() else "done"
        self.repository.finish_job(job_id=job_id, status=status, updated_at=_now())
        return True
