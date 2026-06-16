import numpy as np


class UAVReference:
    def __init__(self, total_steps=220, target=(5.0, 5.0, 3.0), kind='smooth_s'):
        self.total_steps = int(total_steps)
        self.target = np.asarray(target, dtype=float)
        self.kind = kind
        self.path = self._make_path()

    def _make_path(self):
        s = np.linspace(0.0, 1.0, self.total_steps)

        if self.kind == 'line':
            x = self.target[0] * s
            y = self.target[1] * s
            z = self.target[2] * s

        elif self.kind == 'smooth_s':
            x = self.target[0] * s
            y = self.target[1] * s + 0.35 * np.sin(2 * np.pi * s) * s * (1 - s)
            z = self.target[2] * s + 0.25 * np.sin(np.pi * s) * s * (1 - s)

        elif self.kind == 'curve3d':
            
            x = self.target[0] * s
            y = (
                self.target[1] * s
                + 0.55 * np.sin(2.0 * np.pi * s)
                + 0.10 * np.sin(4.0 * np.pi * s)
            )
            z = (
                self.target[2] * s
                + 0.28 * np.sin(np.pi * s)
                - 0.08 * np.sin(3.0 * np.pi * s)
            )
            z = np.clip(z, 0.0, self.target[2] + 0.30)

        else:
            x = self.target[0] * s
            y = self.target[1] * s
            z = self.target[2] * s

        pos = np.stack([x, y, z], axis=1)
        vel = np.gradient(pos, axis=0) / 0.08
        att = np.zeros((self.total_steps, 6))
        return np.concatenate([pos, vel, att], axis=1)

    def horizon(self, t, N):
        idx = np.clip(np.arange(t, t + N + 1), 0, self.total_steps - 1)
        return self.path[idx]

    def progress(self, state):
        pos = np.asarray(state[:3], dtype=float)
        d = np.linalg.norm(self.path[:, :3] - pos[None, :], axis=1)
        return float(np.argmin(d) / max(1, self.total_steps - 1))
