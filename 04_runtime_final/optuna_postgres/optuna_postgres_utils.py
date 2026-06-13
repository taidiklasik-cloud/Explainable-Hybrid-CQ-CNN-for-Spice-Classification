"""optuna_postgres_utils.py
Utilities for Optuna with local/shared PostgreSQL storage.

Role split used in this project:
- Optuna orchestrator (laptop): creates trials with ask/suggest and inserts tasks into cqcnn_orchestration.
- Worker(s): claim tasks from local PostgreSQL, load trial params from Optuna PostgreSQL, train, then tell result.
- cqcnn_orchestration: orchestration log, heartbeat, checkpoint metadata, hijacking/resume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
from optuna.trial import TrialState


@dataclass(frozen=True)
class OptunaPgConfig:
    storage_url: str
    seed: int = 42
    use_tpe_multivariate: bool = True
    constant_liar: bool = True


def create_or_load_study(
    *,
    study_name: str,
    storage_url: str,
    direction: str,
    seed: int = 42,
    constant_liar: bool = True,
) -> optuna.Study:
    """Create/load a study in PostgreSQL RDB storage.

    direction must be "minimize" or "maximize".
    constant_liar=True is helpful for parallel/distributed TPE so running trials are considered.
    """
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"Invalid Optuna direction: {direction}")

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        group=True,
        constant_liar=constant_liar,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )


def suggest_from_search_space(trial: optuna.trial.Trial, search_space: dict[str, Any] | None) -> dict[str, Any]:
    """Convert JSON search-space spec into Optuna trial.suggest_* calls."""
    params: dict[str, Any] = {}
    for name, spec in (search_space or {}).items():
        ptype = spec.get("type")
        if ptype == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "categorical":
            choices = list(spec.get("choices", []))
            if not choices:
                raise ValueError(f"Categorical search space for {name} has no choices.")
            params[name] = trial.suggest_categorical(name, choices)
        else:
            raise ValueError(f"Unknown search-space type for {name}: {ptype}")
    return params


def get_frozen_trial_by_number(study: optuna.Study, trial_number: int) -> optuna.trial.FrozenTrial:
    for t in study.get_trials(deepcopy=False):
        if t.number == int(trial_number):
            return t
    raise KeyError(f"Trial number not found in study {study.study_name}: {trial_number}")


def get_trial_params(study: optuna.Study, trial_number: int) -> dict[str, Any]:
    """Return authoritative sampled parameters from Optuna PostgreSQL."""
    trial = get_frozen_trial_by_number(study, trial_number)
    return dict(trial.params)


def tell_trial_result(study: optuna.Study, trial_number: int, value: float) -> None:
    """Tell result to Optuna exactly once.

    If the trial is already complete, this function becomes a no-op. This helps recovery
    when a worker updates the orchestration DB but a retry accidentally calls tell again.
    """
    frozen = get_frozen_trial_by_number(study, trial_number)
    if frozen.state == TrialState.COMPLETE:
        return
    if frozen.state not in {TrialState.RUNNING, TrialState.WAITING}:
        raise RuntimeError(f"Cannot tell trial {trial_number}; current state is {frozen.state}.")
    study.tell(frozen, float(value))

def delete_study_if_exists(
    *,
    study_name: str,
    storage_url: str,
) -> bool:
    """Delete one Optuna study if it exists.

    Returns:
        True  = study ditemukan dan berhasil dihapus.
        False = study tidak ditemukan, tidak ada yang dihapus.
    """
    try:
        optuna.load_study(
            study_name=study_name,
            storage=storage_url,
        )
    except KeyError:
        return False

    optuna.delete_study(
        study_name=study_name,
        storage=storage_url,
    )
    return True


def list_studies(
    *,
    storage_url: str,
) -> list[str]:
    """List all Optuna study names in PostgreSQL storage."""
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    return [s.study_name for s in summaries]


def delete_test_studies(
    *,
    storage_url: str,
    prefixes: tuple[str, ...] = ("test_",),
    exact_names: tuple[str, ...] = (),
    dry_run: bool = True,
) -> list[str]:
    """Delete test studies only.

    Safety design:
    - By default only studies starting with 'test_' are targeted.
    - dry_run=True means only preview, not delete.
    - Set dry_run=False to actually delete.

    Returns:
        List of targeted/deleted study names.
    """
    summaries = optuna.get_all_study_summaries(storage=storage_url)

    targets: list[str] = []
    for s in summaries:
        name = s.study_name
        is_exact_match = name in exact_names
        is_prefix_match = any(name.startswith(prefix) for prefix in prefixes)

        if is_exact_match or is_prefix_match:
            targets.append(name)

    if dry_run:
        return targets

    for name in targets:
        optuna.delete_study(
            study_name=name,
            storage=storage_url,
        )

    return targets
