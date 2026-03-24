"""Observation parsing and runtime history management utilities."""

from __future__ import annotations

import inspect
from typing import Iterable

import numpy as np


ObservationValue = np.ndarray | dict[str, np.ndarray]


class SimpleObservationManager:
    """Builds observation groups from a lightweight declarative config.

    The manager resolves configured observation terms against methods on the
    provided environment object, optionally applies clipping and scaling, and
    manages fixed-length history buffers for terms or whole groups.
    """

    _GROUP_SKIP_KEYS = {
        "enable_corruption",
        "concatenate_terms",
        "history_length",
        "flatten_history_dim",
        "concatenate_dim",
    }

    def __init__(self, cfg: object, env: object) -> None:
        """Initializes the observation manager.

        Args:
            cfg: Observation configuration container. It can be a plain object
                or a dictionary keyed by group name.
            env: Runtime object whose ``_obs_*`` methods produce raw
                observation terms.
        """
        self.cfg = cfg
        self.env = env
        self._group_terms = {}
        self._group_cfg = {}
        self._history = {}
        self._group_concat = {}
        self._group_concat_dim = {}
        self._prepare()

    def _iter_cfg_items(self, cfg_obj: object) -> Iterable[tuple[str, object]]:
        """Iterates over config items while preserving declaration order."""
        if inspect.isclass(cfg_obj):
            return []
        if isinstance(cfg_obj, dict):
            return cfg_obj.items()
        items = cfg_obj.__dict__.items()
        if len(items) > 0:
            return items
        items = []
        for k, _ in cfg_obj.__class__.__dict__.items():
            if k.startswith("_"):
                continue
            v = getattr(cfg_obj, k)
            if inspect.isclass(v):
                continue
            items.append((k, v))
        return items

    def _prepare(self) -> None:
        """Resolves all configured groups and terms into executable metadata."""
        for group_name, group_cfg in self._iter_cfg_items(self.cfg):
            if group_cfg is None or not hasattr(group_cfg, "concatenate_terms"):
                continue
            self._group_cfg[group_name] = group_cfg
            self._group_terms[group_name] = []
            self._history[group_name] = {}
            self._group_concat[group_name] = bool(group_cfg.concatenate_terms)
            concat_dim = getattr(group_cfg, "concatenate_dim", -1)
            self._group_concat_dim[group_name] = (
                concat_dim + 1 if concat_dim >= 0 else concat_dim
            )

            group_history = getattr(group_cfg, "history_length", None)
            group_flatten = getattr(group_cfg, "flatten_history_dim", True)

            for term_name, term_cfg in self._iter_cfg_items(group_cfg):
                if term_name in self._GROUP_SKIP_KEYS or term_name.startswith("_"):
                    continue
                if term_cfg is None:
                    continue
                term_func = None
                if isinstance(term_cfg, str):
                    term_func = term_cfg
                    term_history = 0
                    term_flatten = True
                    term_params = {}
                    term_clip = None
                    term_scale = None
                else:
                    term_func = getattr(term_cfg, "func", None)
                    if term_func is None:
                        env_func_name = f"_obs_{term_name}"
                        if hasattr(self.env, env_func_name):
                            term_func = env_func_name
                        else:
                            continue
                    term_history = getattr(term_cfg, "history_length", 0)
                    term_flatten = getattr(term_cfg, "flatten_history_dim", True)
                    term_params = getattr(term_cfg, "params", {})
                    term_clip = getattr(term_cfg, "clip", None)
                    term_scale = getattr(term_cfg, "scale", None)
                if group_history is not None:
                    term_history = group_history
                    term_flatten = group_flatten
                if isinstance(term_func, str):
                    if not hasattr(self.env, term_func):
                        raise AttributeError(
                            f"Env does not have observation function '{term_func}' for term '{term_name}'"
                        )
                    term_func = getattr(self.env, term_func)
                self._group_terms[group_name].append(
                    {
                        "name": term_name,
                        "func": term_func,
                        "params": term_params,
                        "clip": term_clip,
                        "scale": term_scale,
                        "history_length": int(term_history),
                        "flatten_history_dim": bool(term_flatten),
                    }
                )
                self._history[group_name][term_name] = {
                    "buffer": None,
                }
            term_names = [t["name"] for t in self._group_terms[group_name]]
            print(
                f"[SimpleObservationManager] group='{group_name}', terms={term_names}"
            )

    def _to_numpy(self, obs: object) -> np.ndarray:
        """Converts a supported observation value to ``numpy.ndarray``."""
        if isinstance(obs, np.ndarray):
            return obs
        # Torch tensors are supported implicitly to keep the manager free of a
        # hard torch dependency.
        if hasattr(obs, "detach") and hasattr(obs, "cpu") and hasattr(obs, "numpy"):
            return obs.detach().cpu().numpy()
        return np.asarray(obs)

    def compute_group(
        self,
        group_name: str,
        update_history: bool = True,
    ) -> ObservationValue:
        """Computes one configured observation group.

        Args:
            group_name: Group key defined in the active observation config.
            update_history: Whether term history buffers should advance.

        Returns:
            Either a concatenated numpy array or a dictionary of named term
            arrays, depending on the group configuration.

        Raises:
            ValueError: If the requested group does not exist.
        """
        if group_name not in self._group_terms:
            raise ValueError(f"Unknown observation group: {group_name}")
        group_obs = []
        for term in self._group_terms[group_name]:
            func = term["func"]
            params = term["params"]
            if hasattr(func, "__self__") and func.__self__ is not None:
                obs = func(**params)
            else:
                obs = func(self.env, **params)
            obs = self._to_numpy(obs)
            if obs.ndim == 1:
                obs = np.expand_dims(obs, axis=0)

            clip = term["clip"]
            if clip is not None:
                obs = np.clip(obs, clip[0], clip[1])
            scale = term["scale"]
            if scale is not None:
                scale_t = scale
                if not isinstance(scale, np.ndarray):
                    scale_t = np.asarray(scale, dtype=obs.dtype)
                obs = obs * scale_t

            if term["history_length"] > 0:
                hist = self._history[group_name][term["name"]]
                # Warm-start the history buffer with the first observation so
                # downstream code can always assume a full history window.
                if hist["buffer"] is None:
                    hist["buffer"] = [obs.copy() for _ in range(term["history_length"])]
                elif update_history:
                    hist["buffer"].pop(0)
                    hist["buffer"].append(obs.copy())
                hist_tensor = np.stack(hist["buffer"], axis=1)
                if term["flatten_history_dim"]:
                    obs = hist_tensor.reshape(hist_tensor.shape[0], -1)
                else:
                    obs = hist_tensor

            group_obs.append(obs)

        if self._group_concat[group_name]:
            return np.concatenate(group_obs, axis=self._group_concat_dim[group_name])
        return {
            term["name"]: obs
            for term, obs in zip(self._group_terms[group_name], group_obs)
        }


class TermCfg:
    """Configuration for one observation term.

    Args:
        func: Optional callable or environment method name. When omitted, the
            manager falls back to ``env._obs_<term_name>``.
        params: Keyword arguments forwarded to the term callable.
        clip: Optional ``(min, max)`` tuple applied elementwise.
        scale: Optional scalar or vector scale applied after clipping.
        history_length: Number of steps kept for this term when no group-level
            override is active.
        flatten_history_dim: Whether the temporal dimension should be flattened
            into the feature dimension.
    """

    def __init__(
        self,
        func=None,
        params=None,
        clip=None,
        scale=None,
        history_length=0,
        flatten_history_dim=True,
    ):
        self.func = func
        self.params = params or {}
        self.clip = clip
        self.scale = scale
        self.history_length = history_length
        self.flatten_history_dim = flatten_history_dim


class GroupCfg:
    """Base class for one observation group configuration.

    Group-level history settings override term-level history settings because
    the final policy input usually expects a consistent temporal layout across
    the whole group.
    """

    concatenate_terms = True
    concatenate_dim = -1
    history_length = None
    flatten_history_dim = True
    enable_corruption = False
