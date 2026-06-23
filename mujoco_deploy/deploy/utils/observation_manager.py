import inspect
import numpy as np

class SimpleObservationManager:
    """简化版 ObservationManager：从 cfg 解析观测组/项，支持 history/clip/scale/拼接。"""

    _GROUP_SKIP_KEYS = {
        "enable_corruption",
        "concatenate_terms",
        "history_length",
        "flatten_history_dim",
        "concatenate_dim",
    }

    def __init__(self, cfg, env):
        # cfg: ObservationGroupCfg 容器（可能是 dict 或对象）
        # env: 传给观测函数的环境对象
        self.cfg = cfg
        self.env = env
        # group_name -> [term_cfg_dict, ...]
        self._group_terms = {}
        # group_name -> group_cfg
        self._group_cfg = {}
        # group_name -> term_name -> history buffer
        self._history = {}
        # group_name -> 是否拼接
        self._group_concat = {}
        # group_name -> 拼接维度（已考虑 batch 维）
        self._group_concat_dim = {}
        self._prepare()

    def _iter_cfg_items(self, cfg_obj):
        # 只解析实例，跳过类对象；优先保持定义顺序
        if inspect.isclass(cfg_obj):
            return []
        if isinstance(cfg_obj, dict):
            return cfg_obj.items()
        items = cfg_obj.__dict__.items()
        if len(items) > 0:
            return items
        # __dict__ 为空时，从类的 __dict__ 读取（保留定义顺序），并过滤无关项
        items = []
        for k, _ in cfg_obj.__class__.__dict__.items():
            if k.startswith("_"):
                continue
            v = getattr(cfg_obj, k)
            if inspect.isclass(v):
                continue
            items.append((k, v))
        return items

    def _prepare(self):
        # 解析 cfg 中的所有 ObservationGroup
        for group_name, group_cfg in self._iter_cfg_items(self.cfg):
            if group_cfg is None or not hasattr(group_cfg, "concatenate_terms"):
                continue
            self._group_cfg[group_name] = group_cfg
            self._group_terms[group_name] = []
            self._history[group_name] = {}
            # 记录拼接策略
            self._group_concat[group_name] = bool(group_cfg.concatenate_terms)
            concat_dim = getattr(group_cfg, "concatenate_dim", -1)
            self._group_concat_dim[group_name] = concat_dim + 1 if concat_dim >= 0 else concat_dim

            # 组级 history/flatten，可覆盖 term 级配置
            group_history = getattr(group_cfg, "history_length", None)
            group_flatten = getattr(group_cfg, "flatten_history_dim", True)

            for term_name, term_cfg in self._iter_cfg_items(group_cfg):
                if term_name in self._GROUP_SKIP_KEYS or term_name.startswith("_"):
                    continue
                if term_cfg is None:
                    continue
                # 解析 term func（支持字符串 / None / env._obs_{term_name}）
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
                    # 读取 term 级配置（保留 term_cfg 中的 history/flatten）
                    term_history = getattr(term_cfg, "history_length", 0)
                    term_flatten = getattr(term_cfg, "flatten_history_dim", True)
                    term_params = getattr(term_cfg, "params", {})
                    term_clip = getattr(term_cfg, "clip", None)
                    term_scale = getattr(term_cfg, "scale", None)
                # 组级配置优先
                if group_history is not None:
                    term_history = group_history
                    term_flatten = group_flatten
                # 支持 func 为字符串：从 env 中解析 _obs_* 方法
                if isinstance(term_func, str):
                    if not hasattr(self.env, term_func):
                        raise AttributeError(
                            f"Env does not have observation function '{term_func}' for term '{term_name}'"
                        )
                    term_func = getattr(self.env, term_func)
                # 记录 term 信息
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
                # 先占位，buffer 在首次 compute 时初始化
                self._history[group_name][term_name] = {
                    "buffer": None,
                }
            # 调试输出：打印当前解析到的 group/term
            term_names = [t["name"] for t in self._group_terms[group_name]]
            print(f"[SimpleObservationManager] group='{group_name}', terms={term_names}")

    def _to_numpy(self, obs):
        # 统一为 numpy.ndarray，便于后续处理（支持 numpy / list / 标量 / torch.Tensor）
        if isinstance(obs, np.ndarray):
            return obs
        # Support torch.Tensor without importing torch
        if hasattr(obs, "detach") and hasattr(obs, "cpu") and hasattr(obs, "numpy"):
            return obs.detach().cpu().numpy()
        return np.asarray(obs)

    def compute_group(self, group_name, update_history=True):
        # 计算指定 group 的观测
        if group_name not in self._group_terms:
            raise ValueError(f"Unknown observation group: {group_name}")
        group_obs = []
        for term in self._group_terms[group_name]:
            # 1) 调用观测函数
            func = term["func"]
            params = term["params"]
            if hasattr(func, "__self__") and func.__self__ is not None:
                obs = func(**params)
            else:
                obs = func(self.env, **params)
            obs = self._to_numpy(obs)
            if obs.ndim == 1:
                obs = np.expand_dims(obs, axis=0)

            # 2) clip
            clip = term["clip"]
            if clip is not None:
                obs = np.clip(obs, clip[0], clip[1])
            # 3) scale（支持标量或向量）
            scale = term["scale"]
            if scale is not None:
                scale_t = scale
                if not isinstance(scale, np.ndarray):
                    scale_t = np.asarray(scale, dtype=obs.dtype)
                obs = obs * scale_t

            # 4) history（按时间维堆叠，旧 -> 新）
            if term["history_length"] > 0:
                hist = self._history[group_name][term["name"]]
                # 首次使用：用第一帧填满整个历史
                if hist["buffer"] is None:
                    hist["buffer"] = [obs.copy() for _ in range(term["history_length"])]
                # 后续调用：仅在 update_history=True 时滚动更新
                elif update_history:
                    hist["buffer"].pop(0)
                    hist["buffer"].append(obs.copy())
                hist_tensor = np.stack(hist["buffer"], axis=1)
                if term["flatten_history_dim"]:
                    obs = hist_tensor.reshape(hist_tensor.shape[0], -1)
                else:
                    obs = hist_tensor

            # 记录本 term 输出
            group_obs.append(obs)

        # 5) 按组配置决定拼接或返回 dict
        if self._group_concat[group_name]:
            return np.concatenate(group_obs, axis=self._group_concat_dim[group_name])
        return {term["name"]: obs for term, obs in zip(self._group_terms[group_name], group_obs)}

class TermCfg:
    """观测项配置：func 可省略，默认解析为 env._obs_{term_name}。"""

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
    """观测组配置：group 级 history 会覆盖 term 级。"""

    concatenate_terms = True
    concatenate_dim = -1
    history_length = None
    flatten_history_dim = True
    enable_corruption = False
