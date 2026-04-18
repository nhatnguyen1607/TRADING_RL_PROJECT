"""Môi trường giao dịch (Gymnasium). Kế thừa và triển khai MDP: state, action, reward."""

from gymnasium import Env


class TradingEnv(Env):
    """Simulator giao dịch: phí, số dư, OHLCV + chỉ báo."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # TODO: định nghĩa observation_space, action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
