import numpy as np
import torch
from observation_manager import SimpleObservationManager
# ==== 伪造环境 ====
class DummyEnv:
    def __init__(self):
        self.t = 0

# ==== 伪造观测函数 ====
def obs_a(env, **kwargs):
    env.t += 1
    return torch.tensor([[float(env.t), 10.0, -10.0]])  # shape (1,3)

def obs_b(env, **kwargs):
    return torch.tensor([[1.0, 2.0]])  # shape (1,2)

# ==== 伪造 cfg ====
class TermCfg:
    def __init__(self, func, params=None, clip=None, scale=None, history_length=0, flatten_history_dim=True):
        self.func = func
        self.params = params or {}
        self.clip = clip
        self.scale = scale
        self.history_length = history_length
        self.flatten_history_dim = flatten_history_dim

class GroupCfg:
    concatenate_terms = True
    concatenate_dim = -1
    history_length = 4
    flatten_history_dim = True
    enable_corruption = False

    # 两个 term
    a = TermCfg(obs_a, clip=(-5, 5), scale=2.0)
    b = TermCfg(obs_b, scale=[1.0, 10.0])

class ObsCfg:
    policy = GroupCfg()
    policy_2 = GroupCfg()

if __name__ == "__main__":
    # ==== 测试 ====
    env = DummyEnv()
    mgr = SimpleObservationManager(ObsCfg(), env)

    for i in range(6):
        obs = mgr.compute_group("policy_2", update_history=True)
        if isinstance(obs, dict):
            for term_name, term_obs in obs.items():
                print(f"step {i} term '{term_name}' obs shape: {term_obs.shape}")
                print(term_obs)
        else:
            print(f"step {i} obs shape: {obs.shape}")
            print(obs)
