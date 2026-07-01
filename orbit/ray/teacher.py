"""OPD teacher serving.

Mirrors RolloutManager's Manager -> RolloutServer -> ServerGroup hierarchy (see
orbit/ray/rollout.py), but as its own Ray actor scoped to the frozen teacher model(s) that
score the student's already-sampled rollout tokens for on-policy distillation (see
compute_teacher_log_probs). Kept as a separate actor -- rather than folded into
RolloutManager as just another entry in its ``servers`` dict -- because a TeacherGroup that
owns N independent RolloutServer entries (one per teacher name) is the natural home for
future multi-teacher support: adding a second teacher is just another dict entry here, with
no change to RolloutManager or the student rollout path.

Teachers never generate real rollouts (compute_teacher_log_probs forces max_new_tokens=0 to
score already-sampled tokens instead) and never receive weight updates -- every
RolloutServer started here has update_weights=False.
"""

import logging

import ray

from orbit.utils.health_monitor import RolloutHealthMonitor
from orbit.utils.http_utils import init_http_client
from orbit.utils.logging_utils import configure_logger

from .rollout import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_teacher_offset,
    _resolve_sglang_config,
    _start_models_on_pg,
    _teacher_model_names,
)

logger = logging.getLogger(__name__)


def start_teacher_servers(args, pg) -> dict[str, RolloutServer]:
    """Start SGLang engines for the OPD teacher model(s) declared in --sglang-config.

    Returns a dict mapping teacher model name -> ``RolloutServer``. Currently always a
    single entry (args.teacher_model_name), but multi-teacher only needs
    _teacher_model_names() to return more names -- this function already handles N models.
    """
    config = _resolve_sglang_config(args)
    teacher_names = _teacher_model_names(args)
    teacher_models = [m for m in config.models if m.name in teacher_names]
    assert teacher_models, (
        f"advantage_estimator=on_policy_distillation but no --sglang-config model matches "
        f"--teacher-model-name={args.teacher_model_name!r}."
    )
    return _start_models_on_pg(
        args, teacher_models, pg, _compute_teacher_offset(args), _compute_megatron_num_gpus(args)
    )


@ray.remote
class TeacherGroup:
    """Owns the frozen teacher model(s) served via SGLang for on-policy distillation."""

    def __init__(self, args, pg):
        configure_logger()
        self.args = args
        self.pg = pg
        init_http_client(args)
        self.servers: dict[str, RolloutServer] = start_teacher_servers(args, pg)

        self._health_monitors = []
        if args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)

    def dispose(self):
        for monitor in self._health_monitors:
            monitor.stop()

    def health_monitoring_pause(self) -> None:
        for monitor in self._health_monitors:
            monitor.pause()

    def health_monitoring_resume(self) -> None:
        for monitor in self._health_monitors:
            monitor.resume()

    def score(self, samples):
        """Score `samples` against every teacher, setting sample.teacher_log_probs in place.

        Runs inside this actor (not RolloutManager) since it needs this actor's own
        args.sglang_model_routers, populated when this actor started its own servers.
        Combining multiple teachers' scores is future multi-teacher work -- for now
        self.servers always has exactly one entry.
        """
        from orbit.rollout.generate_utils.generate_endpoint_utils import compute_teacher_log_probs
        from orbit.utils.async_utils import run

        self.health_monitoring_resume()
        for model_name in self.servers:
            run(compute_teacher_log_probs(self.args, model_name, samples))
        return samples

    def offload(self, tags: list[str] | None = None):
        self.health_monitoring_pause()
        if tags is not None:
            handles = [
                engine.release_memory_occupation.remote(tags=tags)
                for srv in self.servers.values()
                for engine in srv.engines
                if engine is not None
            ]
            return ray.get(handles) if handles else []
        for srv in self.servers.values():
            srv.offload()

    def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            srv.onload(tags)

    def onload_weights(self):
        for srv in self.servers.values():
            srv.onload_weights()

    def onload_kv(self):
        for srv in self.servers.values():
            srv.onload_kv()
