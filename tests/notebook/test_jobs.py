from telegram_notebook.db import Repository
from telegram_notebook.jobs import JobWorker


def _repo(tmp_path) -> Repository:
    repo = Repository(tmp_path / "store.db")
    repo.init()
    return repo


# --- job repository ---

def test_job_lifecycle_fields(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="https://t.me/x", limit=None)
    job = repo.get_job(job_id=jid)
    assert job["status"] == "queued" and job["processed"] == 0 and job["cursor"] == 0

    # owner scoping
    assert repo.get_job(job_id=jid, owner_id=2) is None
    assert len(repo.list_jobs(owner_id=1)) == 1
    assert repo.list_jobs(owner_id=2) == []


def test_claim_is_atomic_and_ordered(tmp_path):
    repo = _repo(tmp_path)
    a = repo.create_job(owner_id=1, channel_url="a", limit=None)
    b = repo.create_job(owner_id=1, channel_url="b", limit=None)

    first = repo.claim_next_queued_job()
    assert first["id"] == a and first["status"] == "running"
    second = repo.claim_next_queued_job()
    assert second["id"] == b
    assert repo.claim_next_queued_job() is None  # nothing left queued


def test_progress_cancel_and_requeue(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="x", limit=None)
    repo.claim_next_queued_job()

    repo.update_job_progress(job_id=jid, processed=5, total=10, cursor=123)
    job = repo.get_job(job_id=jid)
    assert (job["processed"], job["total"], job["cursor"]) == (5, 10, 123)

    assert repo.is_cancel_requested(job_id=jid) is False
    assert repo.request_job_cancel(owner_id=1, job_id=jid) is True
    assert repo.is_cancel_requested(job_id=jid) is True
    # cannot cancel another owner's job
    assert repo.request_job_cancel(owner_id=2, job_id=jid) is False

    # a crash leaves it 'running'; requeue resets it
    repo.finish_job(job_id=jid, status="running")
    assert repo.requeue_running_jobs() == 1
    assert repo.get_job(job_id=jid)["status"] == "queued"


# --- worker state machine (fake runner, no Telegram) ---

def test_worker_runs_job_to_done_with_progress(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="x", limit=None)

    def runner(job, on_progress, is_cancelled):
        assert job["id"] == jid
        on_progress(3, 3, 999)  # simulate work

    worker = JobWorker(repo, runner)
    assert worker.process_one() is True
    job = repo.get_job(job_id=jid)
    assert job["status"] == "done"
    assert (job["processed"], job["cursor"]) == (3, 999)
    # no more work
    assert worker.process_one() is False


def test_worker_marks_failed_on_exception(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="x", limit=None)

    def runner(job, on_progress, is_cancelled):
        raise RuntimeError("boom")

    assert JobWorker(repo, runner).process_one() is True
    job = repo.get_job(job_id=jid)
    assert job["status"] == "failed" and "boom" in job["error"]


def test_worker_marks_cancelled(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="x", limit=None)
    repo.request_job_cancel(owner_id=1, job_id=jid)

    observed = {}

    def runner(job, on_progress, is_cancelled):
        observed["cancelled"] = is_cancelled()  # runner sees the request and stops early

    assert JobWorker(repo, runner).process_one() is True
    assert observed["cancelled"] is True
    assert repo.get_job(job_id=jid)["status"] == "cancelled"


def test_worker_resume_cursor_is_passed_to_runner(tmp_path):
    repo = _repo(tmp_path)
    jid = repo.create_job(owner_id=1, channel_url="x", limit=None)
    repo.claim_next_queued_job()
    repo.update_job_progress(job_id=jid, processed=2, total=10, cursor=42)
    # simulate interruption + restart
    repo.requeue_running_jobs()

    seen = {}

    def runner(job, on_progress, is_cancelled):
        seen["cursor"] = job["cursor"]

    JobWorker(repo, runner).process_one()
    assert seen["cursor"] == 42  # resumes from where it stopped
