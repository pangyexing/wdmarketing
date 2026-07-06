"""Dependency-free progress monitoring for Stage-1 / Stage-2 pipelines.

Two layers:
- StageProgress — coarse numbered pipeline steps with per-step + cumulative timing
- track() / ProgressCounter — fine-grained loop progress with percent + ETA

Everything logs through the standard ``wdm`` logger so progress lines land in
the same stderr stream (and format) as the rest of the pipeline logs.
"""
import logging
import time

logger = logging.getLogger(__name__)


def fmt_duration(seconds):
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "{0:.1f}s".format(seconds)
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return "{0}m{1:02d}s".format(minutes, sec)
    hours, minutes = divmod(minutes, 60)
    return "{0}h{1:02d}m".format(hours, minutes)


def _log_progress(label, done, total, t0, extra=None):
    elapsed = time.time() - t0
    suffix = " | {0}".format(extra) if extra else ""
    if total:
        eta = elapsed / done * (total - done)
        logger.info("%s: %d/%d (%.0f%%) elapsed %s, ETA %s%s",
                    label, done, total, 100.0 * done / total,
                    fmt_duration(elapsed), fmt_duration(eta), suffix)
    else:
        logger.info("%s: %d done, elapsed %s%s",
                    label, done, fmt_duration(elapsed), suffix)


class ProgressCounter:
    """Manual counter for loops where a generator wrapper doesn't fit
    (e.g. nested block-pair loops). Call tick() once per unit of work.
    """

    def __init__(self, label, total=None, every=1):
        self.label = label
        self.total = int(total) if total else None
        self.every = max(1, int(every))
        self.done = 0
        self.t0 = time.time()

    def tick(self, extra=None):
        self.done += 1
        if self.done % self.every == 0 or self.done == self.total:
            _log_progress(self.label, self.done, self.total, self.t0, extra)


def track(iterable, total=None, label="progress", every=1):
    """Yield from iterable, logging 'label: i/n (pct%) elapsed, ETA' as items
    are consumed. With total=None falls back to a plain running count.
    """
    counter = ProgressCounter(label, total=total, every=every)
    for item in iterable:
        yield item
        counter.tick()


class StageProgress:
    """Coarse step tracker for a multi-step stage.

    Usage:
        prog = StageProgress("Stage 1", total=8)
        with prog.step("IV / WOE"):
            ...
        prog.finish()
    """

    def __init__(self, stage_name, total):
        self.stage_name = stage_name
        self.total = int(total)
        self.done = 0
        self.t0 = time.time()

    def step(self, name):
        return _Step(self, name)

    def finish(self):
        logger.info("[%s] all %d steps done in %s",
                    self.stage_name, self.total,
                    fmt_duration(time.time() - self.t0))


class _Step:
    def __init__(self, prog, name):
        self.prog = prog
        self.name = name
        self.t_start = None

    def __enter__(self):
        self.prog.done += 1
        self.t_start = time.time()
        logger.info("[%s %d/%d] %s ...",
                    self.prog.stage_name, self.prog.done, self.prog.total,
                    self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "FAILED" if exc_type else "done"
        logger.info("[%s %d/%d] %s %s in %s (total elapsed %s)",
                    self.prog.stage_name, self.prog.done, self.prog.total,
                    self.name, status,
                    fmt_duration(time.time() - self.t_start),
                    fmt_duration(time.time() - self.prog.t0))
        return False
