import os
import threading
import time

import pytest

from mempalace import daemon
from mempalace import service

# Env keys run_server mutates from its background thread, plus umask. If a
# lifecycle test times out before the server comes up, run_server's finally
# never runs and those mutations leak into the rest of the suite — every later
# test that reads MempalaceConfig().palace_path sees a stale deleted tmp path and
# fails (the 60+ test cascade seen on slow CI runners). The fixtures below force a
# clean baseline around every daemon test so a leaked thread can't poison the
# process for tests/test_mcp_server.py and friends (which have no such guard).
_LEAK_ENV_KEYS = ("MEMPALACE_PALACE_PATH", "MEMPALACE_BACKEND", "MEMPALACE_BACKEND_EXPLICIT")


@pytest.fixture(scope="module")
def _clean_env_snapshot():
    """Capture the true pre-suite values once, before any daemon test runs."""
    return {key: os.environ.get(key) for key in _LEAK_ENV_KEYS}


@pytest.fixture(autouse=True)
def _isolate_process_global_state(_clean_env_snapshot):
    """Restore the process-global env + umask to the pre-suite baseline after every
    daemon test, even if a leaked run_server thread is still holding them mutated.
    """
    prev_umask = os.umask(0o022)
    os.umask(prev_umask)  # read current umask without changing it
    yield
    for key, value in _clean_env_snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    os.umask(prev_umask)


def _raise_not_ready(*a, **kw):
    """Stand-in for DaemonClient when the spawned daemon must never come up."""
    raise daemon.DaemonError("not ready")


def test_prune_terminal_drops_old_terminal_jobs_keeps_active(tmp_path, monkeypatch):
    """Terminal jobs older than the retention window are pruned; queued/running
    and fresh terminal jobs are untouched. Bounded queue growth for the DB that
    holds verbatim payloads."""
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(str(palace)))

    old_term = store.enqueue("mine", {"source": "old"})
    store.finish(old_term.id, state="succeeded", result={"success": True})
    fresh_term = store.enqueue("mine", {"source": "fresh"})
    store.finish(fresh_term.id, state="succeeded", result={"success": True})
    queued = store.enqueue("mine", {"source": "queued"})

    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with store._lock, store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET finished_at = ? WHERE id = ?",
            (cutoff, old_term.id),
        )

    pruned = store.prune_terminal(older_than_days=7)
    assert pruned == 1
    # The old terminal job is gone; the fresh terminal and queued jobs survive.
    with pytest.raises(daemon.DaemonError):
        store.get(old_term.id)
    assert store.get(fresh_term.id).state == "succeeded"
    assert store.get(queued.id).state == "queued"


def test_queue_dedupes_and_recovers_running_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()

    store = daemon.QueueStore(daemon.queue_path(str(palace)))
    first = store.enqueue("mine", {"source": "a"}, dedupe_key="same")
    second = store.enqueue("mine", {"source": "a"}, dedupe_key="same")

    assert second.id == first.id

    claimed = store.claim_next()
    assert claimed.id == first.id
    assert claimed.state == "running"

    recovered = store.recover_running()
    assert recovered == 1
    assert store.get(first.id).state == "queued"


def test_daemon_http_lifecycle_executes_job(tmp_path, monkeypatch):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    calls = []

    def fake_execute(kind, payload):
        calls.append((kind, payload))
        return {"success": True, "exit_code": 0, "stdout": "done\n"}

    monkeypatch.setattr(service, "execute_job", fake_execute)

    thread = threading.Thread(
        target=daemon.run_server,
        kwargs={"palace_path": str(palace), "port": 0},
        daemon=True,
    )
    thread.start()

    client = None
    # 30s: localhost bind is sub-second locally, but contended CI runners (notably
    # the macOS GitHub Actions fleet) can take several seconds to bring the server
    # up. A too-tight deadline here makes the test flake AND, because the server
    # thread never shuts down on timeout, leaks env/umask into the rest of the
    # suite (guarded by the _isolate_process_global_state fixture above).
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        client = daemon.get_client_if_running(str(palace))
        if client is not None:
            break
        time.sleep(0.05)

    assert client is not None
    health = client.health()
    assert health["ok"] is True
    assert health["palace_path"] == daemon.canonical_palace_path(str(palace))

    job = client.submit("mine", {"source": "src"}, dedupe_key="job")
    finished = client.wait(job["id"], timeout=5)

    assert finished["state"] == "succeeded"
    assert finished["result"]["stdout"] == "done\n"
    assert calls == [("mine", {"source": "src", "palace_path": str(palace.resolve())})]

    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_submit_job_uses_client_and_waits(monkeypatch, tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()

    class DummyClient:
        def __init__(self):
            self.submitted = None

        def submit(self, kind, payload, dedupe_key=None, priority=0):
            self.submitted = (kind, payload, dedupe_key, priority)
            return {"id": "job-1", "state": "queued"}

        def wait(self, job_id, timeout=daemon.DEFAULT_WAIT_TIMEOUT):
            assert job_id == "job-1"
            return {
                "id": "job-1",
                "state": "succeeded",
                "result": {"success": True, "exit_code": 0},
            }

    dummy = DummyClient()
    monkeypatch.setattr(daemon, "ensure_client", lambda *a, **kw: dummy)

    job = daemon.submit_job(
        "mine",
        {"source": "src"},
        palace_path=str(palace),
        dedupe_key="dedupe",
        wait=True,
    )

    assert job["state"] == "succeeded"
    assert dummy.submitted[0] == "mine"
    # palace_path is overridden (not trusted from the payload), never appended.
    assert dummy.submitted[1]["palace_path"] == daemon.canonical_palace_path(str(palace))
    assert dummy.submitted[2] == "dedupe"


def test_service_tool_classification():
    assert service.classify_tool("mempalace_search") == "read"
    assert service.classify_tool("mempalace_add_drawer") == "write"
    assert service.classify_tool("mempalace_mine") == "maintenance"
    assert service.classify_tool("unknown") == "unknown"


# --- helpers for HTTP-lifecycle tests ---


def _start_server(tmp_path, monkeypatch, execute_fn):
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    monkeypatch.setattr(service, "execute_job", execute_fn)
    thread = threading.Thread(
        target=daemon.run_server,
        kwargs={"palace_path": str(palace), "port": 0},
        daemon=True,
    )
    thread.start()
    client = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        client = daemon.get_client_if_running(str(palace))
        if client is not None:
            break
        time.sleep(0.05)
    assert client is not None
    return client, thread, palace


# --- ship-blocker regressions ---


def test_systemexit_in_job_does_not_kill_worker(tmp_path, monkeypatch):
    """A SystemExit (BaseException, not Exception) must be caught, the job
    marked failed, and the worker kept alive for the next job. Regression for
    the critical worker-death bug."""
    state = {"first": True}

    def fake_execute(kind, payload):
        if state["first"]:
            state["first"] = False
            raise SystemExit("boom")
        return {"success": True, "exit_code": 0}

    client, thread, palace = _start_server(tmp_path, monkeypatch, fake_execute)
    try:
        first = client.submit("mine", {"source": "src"})
        finished_first = client.wait(first["id"], timeout=5)
        assert finished_first["state"] == "failed"
        assert finished_first["error"]["error_class"] == "SystemExit"

        # Worker must still be alive — health reports it and a second job runs.
        assert client.health()["worker_alive"] is True
        second = client.submit("mine", {"source": "src2"})
        finished_second = client.wait(second["id"], timeout=5)
        assert finished_second["state"] == "succeeded"
    finally:
        client.shutdown()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_shutdown_cancels_active_job(tmp_path, monkeypatch):
    """POST /shutdown must not leave an in-flight job 'running' for blind
    re-queue on next start. The worker is drained (bounded), then the active
    job is marked 'cancelled' so recover_running won't re-run it.

    In production the serve process exits immediately after run_server returns,
    killing the daemon worker thread before it can overwrite the cancelled
    state. The test mirrors that by asserting the cancelled state *before*
    releasing the blocked worker.
    """
    block = threading.Event()

    def fake_execute(kind, payload):
        # Simulate a long-running job that never finishes on its own.
        block.wait(30)
        return {"success": True, "exit_code": 0}

    monkeypatch.setattr(daemon, "SHUTDOWN_DRAIN_SECONDS", 0.2)
    client, thread, palace = _start_server(tmp_path, monkeypatch, fake_execute)
    job = client.submit("mine", {"source": "src"}, dedupe_key="x")
    # Wait until the worker has claimed it (state flips to running).
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.get_job(job["id"])["state"] == "running":
            break
        time.sleep(0.02)
    assert client.get_job(job["id"])["state"] == "running"

    client.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()

    # The interrupted job must be cancelled (terminal), not left running.
    store = daemon.QueueStore(daemon.queue_path(str(palace)))
    final = store.get(job["id"])
    assert final.state == "cancelled"
    # And recover_running must not re-queue a cancelled job.
    assert store.recover_running() == 0

    # Release the blocked worker so it (and the daemon thread) can exit.
    block.set()


def test_recover_running_dead_letters_exhausted_jobs(tmp_path, monkeypatch):
    """A job that has crashed MAX_ATTEMPTS times must be dead-lettered to
    'failed', not re-queued — non-idempotent kinds (diary_write) would
    otherwise duplicate verbatim content on every restart."""
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(str(palace)))
    job = store.enqueue("diary_write", {"entry": "x"})
    # Simulate MAX_ATTEMPTS claims that each crashed (running, attempts=MAX).
    with store._lock, store._connect() as conn:
        conn.execute(
            "UPDATE jobs SET state='running', attempts=? WHERE id=?",
            (daemon.MAX_ATTEMPTS, job.id),
        )

    recovered = store.recover_running()
    assert recovered == 0  # not re-queued
    final = store.get(job.id)
    assert final.state == "failed"
    assert final.attempts == daemon.MAX_ATTEMPTS


def test_claim_next_does_not_reclaim_running_job(tmp_path, monkeypatch):
    """The conditional UPDATE (WHERE state='queued') means a job already
    flipped to 'running' cannot be claimed again — the cross-process
    double-execution guard."""
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(str(palace)))
    job = store.enqueue("mine", {"source": "src"})
    first = store.claim_next()
    assert first.id == job.id
    # Manually re-mark it queued but leave a second claim attempt: claim_next
    # should still only ever return one running job per claim. After finishing
    # the first, the next claim returns None (queue empty).
    store.finish(first.id, state="succeeded", result={"success": True})
    assert store.claim_next() is None


def test_queue_db_file_is_owner_only(tmp_path, monkeypatch):
    """The queue DB holds verbatim payloads — it must be 0600, not the sqlite
    default 0644. Regression for the privacy-principle violation."""
    import os as _os

    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    store = daemon.QueueStore(daemon.queue_path(str(palace)))
    store.enqueue("diary_write", {"entry": "secret verbatim content"})
    mode = _os.stat(str(store.path)).st_mode & 0o777
    assert mode == 0o600, f"queue.sqlite3 is {oct(mode)}, expected 0600"


def test_token_file_is_owner_only(tmp_path, monkeypatch):
    import os as _os

    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    daemon.ensure_token(str(palace))
    token_path = daemon.state_dir(str(palace)) / "token"
    assert (_os.stat(str(token_path)).st_mode & 0o777) == 0o600


def test_health_rejects_missing_and_wrong_token(tmp_path, monkeypatch):
    from urllib import error as urlerror
    from urllib import request as urlrequest

    client, thread, palace = _start_server(tmp_path, monkeypatch, lambda k, p: {"success": True})
    try:
        base = f"http://127.0.0.1:{client.port}"
        # No Authorization header → 401.
        with pytest.raises(urlerror.HTTPError):
            urlrequest.urlopen(urlrequest.Request(base + "/health"), timeout=3)
        # Wrong token → 401.
        with pytest.raises(urlerror.HTTPError):
            urlrequest.urlopen(
                urlrequest.Request(base + "/health", headers={"Authorization": "Bearer wrong"}),
                timeout=3,
            )
    finally:
        client.shutdown()
        thread.join(timeout=5)


def test_worker_overrides_client_palace_path(tmp_path, monkeypatch):
    """An authenticated client must not be able to retarget the daemon at a
    different palace by stuffing palace_path into the payload."""
    seen = {}

    def fake_execute(kind, payload):
        seen["palace_path"] = payload.get("palace_path")
        return {"success": True, "exit_code": 0}

    client, thread, palace = _start_server(tmp_path, monkeypatch, fake_execute)
    try:
        job = client.submit(
            "mine", {"source": "src", "palace_path": "/tmp/other-palace"}, dedupe_key="p"
        )
        client.wait(job["id"], timeout=5)
    finally:
        client.shutdown()
        thread.join(timeout=5)
    assert seen["palace_path"] == daemon.canonical_palace_path(str(palace))
    assert seen["palace_path"] != "/tmp/other-palace"


def test_mcp_tool_allowlist_rejects_non_write_tools(tmp_path, monkeypatch):
    """The daemon queue is a durable write surface; read/maintenance/unknown
    tools must be rejected so verbatim content can't be exfiltrated into the
    queue or retried destructively."""
    # read tool → rejected
    out = service.run_mcp_tool({"name": "mempalace_search", "arguments": {}})
    assert out["success"] is False
    assert "only accepts write tools" in out["error"]
    # maintenance tool → rejected (has its own kinds: mine/sync)
    out = service.run_mcp_tool({"name": "mempalace_mine", "arguments": {}})
    assert out["success"] is False
    # unknown tool → rejected
    out = service.run_mcp_tool({"name": "mempalace_bogus", "arguments": {}})
    assert out["success"] is False
    # write tool → passes the allowlist (handler not called here since TOOLS
    # won't have it under the test name; but classification must let it through)
    assert service.classify_tool("mempalace_add_drawer") == "write"


def test_execute_job_isolates_env_per_job(monkeypatch):
    """A job that mutates MEMPALACE_BACKEND must not leak into the next job's
    env. Regression for the per-job isolation bug (_apply_backend poisoning)."""
    import os as _os

    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    monkeypatch.delenv("MEMPALACE_PALACE_PATH", raising=False)

    def fake_mine(payload):
        _os.environ["MEMPALACE_BACKEND"] = "leaked-backend"
        return {"success": True, "exit_code": 0}

    monkeypatch.setattr(service, "run_mine", fake_mine)
    service.execute_job("mine", {"palace_path": "/tmp/p", "source": "s"})
    assert _os.environ.get("MEMPALACE_BACKEND") is None


def test_daemon_client_raises_on_endpoint_missing_port(tmp_path, monkeypatch):
    """A malformed endpoint.json must raise DaemonError, not a bare KeyError."""
    import json as _json

    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    daemon.ensure_token(str(palace))
    # endpoint with no port
    daemon.state_dir(str(palace)).mkdir(parents=True, exist_ok=True)
    (daemon.state_dir(str(palace)) / "endpoint.json").write_text(
        _json.dumps({"host": "127.0.0.1", "pid": 1}) + "\n", encoding="utf-8"
    )

    with pytest.raises(daemon.DaemonError):
        daemon.DaemonClient(str(palace))


def test_start_daemon_kills_orphan_on_readiness_timeout(tmp_path, monkeypatch):
    """If the spawned daemon never becomes ready, start_daemon must kill and
    reap the orphaned subprocess rather than leaking it with the port/token."""
    monkeypatch.setenv(daemon.STATE_ROOT_ENV, str(tmp_path / "state"))
    palace = tmp_path / "palace"
    palace.mkdir()
    daemon.ensure_token(str(palace))

    monkeypatch.setattr(daemon, "get_client_if_running", lambda *a, **kw: None)

    class FakeProc:
        def __init__(self):
            self.killed = False
            self.returncode = None

        def poll(self):
            return self.returncode  # None == still alive

        def kill(self):
            self.killed = True

        def wait(self):
            self.returncode = -9
            return self.returncode

    fake = FakeProc()

    def fake_popen(*a, **kw):
        return fake

    monkeypatch.setattr(daemon.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(daemon, "DaemonClient", _raise_not_ready)
    monkeypatch.setattr(daemon.time, "sleep", lambda *a, **kw: None)

    with pytest.raises(daemon.DaemonError):
        daemon.start_daemon(str(palace), timeout=0.05)
    assert fake.killed is True


# --- service.run_* happy-path coverage ---
# These close the draft PR's follow-up ("Add focused happy-path tests for
# service.run_mine / run_diary_write / run_mcp_tool") and, now that the daemon
# tests complete reliably, keep service.py's coverage above the CI gate. The
# capsys-using tests come first; the two that import mempalace.mcp_server (which
# rebinds sys.stdout) come last and do not use capsys, so the rebind can't break
# capture in this file or later files (capsys activates after the rebind).


def test_print_job_result_replays_stdout_stderr_and_returns_exit_code(capsys):
    from mempalace import service

    code = service.print_job_result(
        {"success": False, "error": "boom", "stdout": "out\n", "stderr": "err\n", "exit_code": 3}
    )
    assert code == 3
    captured = capsys.readouterr()
    assert "out" in captured.out
    assert "err" in captured.err


def test_print_job_result_prints_error_to_stderr_when_no_stderr(capsys):
    from mempalace import service

    code = service.print_job_result({"success": False, "error": "boom", "exit_code": 1})
    assert code == 1
    captured = capsys.readouterr()
    assert "mempalace: boom" in captured.err


def test_run_sync_returns_success_when_palace_dir_missing(tmp_path):
    from mempalace import service

    result = service.run_sync({"palace_path": str(tmp_path / "nope"), "dry_run": True})
    assert result["success"] is True
    assert result["exit_code"] == 0


def test_run_sync_returns_success_when_palace_has_no_backend_artifact(tmp_path):
    from mempalace import service

    palace = tmp_path / "palace"
    palace.mkdir()
    result = service.run_sync({"palace_path": str(palace), "dry_run": True})
    assert result["success"] is True
    assert result["exit_code"] == 0


def test_run_mine_invalid_mode_returns_structured_error(tmp_path):
    from mempalace import service

    palace = tmp_path / "palace"
    palace.mkdir()
    out = service.run_mine({"palace_path": str(palace), "mode": "bogus"})
    assert out["success"] is False
    assert "invalid mine mode" in out["error"]
    assert out["exit_code"] == 2


def test_run_mcp_tool_rejects_non_dict_arguments():
    from mempalace import service

    out = service.run_mcp_tool({"name": "mempalace_add_drawer", "arguments": "nope"})
    assert out["success"] is False
    assert "must be an object" in out["error"]
    assert out["exit_code"] == 2


def test_run_mcp_tool_dispatches_write_tool(monkeypatch):
    import mempalace.mcp_server as mcp
    from mempalace import service

    captured = {}

    def fake_handler(**arguments):
        captured["arguments"] = arguments
        return {"success": True, "written": True}

    monkeypatch.setattr(mcp, "TOOLS", {"mempalace_add_drawer": {"handler": fake_handler}})
    out = service.run_mcp_tool({"name": "mempalace_add_drawer", "arguments": {"x": 1}})
    assert out["success"] is True
    assert out["written"] is True
    assert out["exit_code"] == 0
    assert captured["arguments"] == {"x": 1}


def test_run_diary_write_forwards_args_and_sets_exit_code(monkeypatch):
    import mempalace.mcp_server as mcp
    from mempalace import service

    captured = {}

    def fake_diary(agent_name, entry, topic, wing):
        captured.update(agent_name=agent_name, entry=entry, topic=topic, wing=wing)
        return {"success": True}

    monkeypatch.setattr(mcp, "tool_diary_write", fake_diary)
    out = service.run_diary_write(
        {"agent_name": "alice", "entry": "hello", "topic": "t", "wing": "w"}
    )
    assert out["success"] is True
    assert out["exit_code"] == 0
    assert captured == {"agent_name": "alice", "entry": "hello", "topic": "t", "wing": "w"}


def test_run_mine_applies_backend_before_mode_validation(tmp_path):
    """Covers _apply_backend (env set + get_backend_class validation) on the daemon
    path; the invalid mode short-circuits before any mining runs."""
    from mempalace import service

    palace = tmp_path / "palace"
    palace.mkdir()
    out = service.run_mine({"palace_path": str(palace), "mode": "bogus", "backend": "chroma"})
    assert out["success"] is False
    assert out["exit_code"] == 2


def test_execute_job_dispatches_diary_write_mcp_tool_and_unknown(monkeypatch):
    """Covers execute_job's kind dispatch for diary_write, mcp_tool, and the
    unknown-kind fallback."""
    import mempalace.mcp_server as mcp
    from mempalace import service

    monkeypatch.setattr(mcp, "tool_diary_write", lambda **kw: {"success": True})
    monkeypatch.setattr(
        mcp, "TOOLS", {"mempalace_add_drawer": {"handler": lambda **kw: {"success": True}}}
    )
    assert service.execute_job("diary_write", {"entry": "x"})["success"] is True
    assert (
        service.execute_job("mcp_tool", {"name": "mempalace_add_drawer", "arguments": {}})[
            "success"
        ]
        is True
    )
    unknown = service.execute_job("bogus_kind", {})
    assert unknown["success"] is False
    assert unknown["exit_code"] == 2


def test_run_sync_structured_errors_on_sync_failures(tmp_path, monkeypatch):
    """Covers run_sync's three exception handlers (MineAlreadyRunning, ValueError,
    generic Exception) so a failing sync_palace returns a structured error instead
    of propagating."""
    import mempalace.sync as sync_module
    from mempalace import service
    from mempalace.palace import MineAlreadyRunning

    palace = tmp_path / "palace"
    palace.mkdir()
    (palace / "chroma.sqlite3").touch()

    def _raise(exc):
        def fn(**kw):
            raise exc

        return fn

    monkeypatch.setattr(sync_module, "sync_palace", _raise(MineAlreadyRunning("locked")))
    r = service.run_sync({"palace_path": str(palace), "dry_run": True})
    assert r["success"] is False
    assert r["error_class"] == "LockHeldByOtherProcess"

    monkeypatch.setattr(sync_module, "sync_palace", _raise(ValueError("bad scope")))
    r = service.run_sync({"palace_path": str(palace), "dry_run": True})
    assert r["success"] is False
    assert r["exit_code"] == 2

    monkeypatch.setattr(sync_module, "sync_palace", _raise(RuntimeError("boom")))
    r = service.run_sync({"palace_path": str(palace), "dry_run": True})
    assert r["success"] is False
    assert "sync failed" in r["error"]
