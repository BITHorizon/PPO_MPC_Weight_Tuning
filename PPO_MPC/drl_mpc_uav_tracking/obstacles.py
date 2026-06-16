from dataclasses import dataclass
import numpy as np


@dataclass
class CylinderObstacle:
    x: float
    y: float
    radius: float
    height: float
    name: str = 'static'


@dataclass
class MovingSphereObstacle:
    start: tuple
    velocity: tuple
    radius: float
    name: str = 'dynamic'

    def position(self, t, dt):
        return (
            np.asarray(self.start, dtype=float)
            + np.asarray(self.velocity, dtype=float) * (t * dt)
        )


# ============================================================
# Scene validity check
# ============================================================
def _static_static_overlap(static_obs, min_gap=0.05):
    for i in range(len(static_obs)):
        for j in range(i + 1, len(static_obs)):
            oi, oj = static_obs[i], static_obs[j]
            dxy = np.hypot(oi.x - oj.x, oi.y - oj.y)
            if dxy < (oi.radius + oj.radius + min_gap):
                return True
    return False


def _sphere_cylinder_overlap_at(p_sphere, r_sphere, cyl, min_gap=0.05):
    p_sphere = np.asarray(p_sphere, dtype=float)

    horizontal_dist = np.hypot(p_sphere[0] - cyl.x, p_sphere[1] - cyl.y)
    horizontal_overlap = horizontal_dist < (r_sphere + cyl.radius + min_gap)

    vertical_overlap = (
        p_sphere[2] + r_sphere + min_gap >= 0.0
        and p_sphere[2] - r_sphere - min_gap <= cyl.height
    )

    return bool(horizontal_overlap and vertical_overlap)


def _dynamic_static_overlap(dynamic_ob, static_obs, total_steps=220, dt=0.08,
                            min_gap=0.05, sample_every=1):
    for t in range(0, int(total_steps) + 1, max(1, int(sample_every))):
        p = dynamic_ob.position(t, dt)
        for cyl in static_obs:
            if _sphere_cylinder_overlap_at(
                p,
                dynamic_ob.radius,
                cyl,
                min_gap=min_gap,
            ):
                return True
    return False


def _valid_scene(static_obs, dynamic_obs=None, total_steps=220, dt=0.08,
                 min_gap=0.05):
    if _static_static_overlap(static_obs, min_gap=min_gap):
        return False

    dynamic_obs = dynamic_obs or []
    for dyn in dynamic_obs:
        if _dynamic_static_overlap(
            dyn,
            static_obs,
            total_steps=total_steps,
            dt=dt,
            min_gap=min_gap,
            sample_every=1,
        ):
            return False

    return True


# ============================================================
# Obstacle generators
# ============================================================
def static_obstacles_default(rng=None, randomize=False):

    if rng is None:
        rng = np.random.default_rng(0)

    base_obs = [
        dict(x=1.62, y=1.55, radius=0.28, height=1.35, name='low_static'),
        dict(x=2.68, y=2.55, radius=0.34, height=2.35, name='blocking_static'),
    ]

    if not randomize:
        out = [CylinderObstacle(**item) for item in base_obs]
        if _static_static_overlap(out, min_gap=0.05):
            raise RuntimeError("Fixed static obstacle scene is invalid.")
        return out

    out = []
    for item in base_obs:
        out.append(CylinderObstacle(
            x=item['x'] + rng.uniform(-0.10, 0.10),
            y=item['y'] + rng.uniform(-0.10, 0.10),
            radius=item['radius'] * rng.uniform(0.92, 1.12),
            height=item['height'] * rng.uniform(0.95, 1.08),
            name=item['name'],
        ))

    return out


def dynamic_obstacles_default(rng=None, randomize=False):
    
    if rng is None:
        rng = np.random.default_rng(0)

    base_start = np.array([1.95, 3.45, 2.05], dtype=float)
    base_velocity = np.array([0.050, -0.014, 0.000], dtype=float)
    base_radius = 0.28

    if randomize:
        start = base_start + np.array([
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.10, 0.10),
            rng.uniform(-0.06, 0.06),
        ])

        velocity = base_velocity * np.array([
            rng.uniform(0.90, 1.12),
            rng.uniform(0.90, 1.12),
            1.0,
        ])

        radius = base_radius * rng.uniform(0.95, 1.10)
    else:
        start = base_start
        velocity = base_velocity
        radius = base_radius

    return [
        MovingSphereObstacle(
            start=tuple(start),
            velocity=tuple(velocity),
            radius=float(radius),
            name='crossing_sphere',
        )
    ]


def check_default_scene(total_steps=220, dt=0.08):
    statics = static_obstacles_default(randomize=False)
    dynamics = dynamic_obstacles_default(randomize=False)

    min_gap_no_extra = np.inf
    min_gap_with_005 = np.inf
    dyn_static_overlap = False

    for dyn in dynamics:
        for t in range(total_steps + 1):
            p = dyn.position(t, dt)
            for cyl in statics:
                dxy = np.hypot(p[0] - cyl.x, p[1] - cyl.y)
                gap_no_extra = dxy - (dyn.radius + cyl.radius)
                gap_with_005 = dxy - (dyn.radius + cyl.radius + 0.05)
                min_gap_no_extra = min(min_gap_no_extra, float(gap_no_extra))
                min_gap_with_005 = min(min_gap_with_005, float(gap_with_005))
                if _sphere_cylinder_overlap_at(p, dyn.radius, cyl, min_gap=0.0):
                    dyn_static_overlap = True

    return {
        'valid_scene_min_gap_0.05': _valid_scene(
            statics,
            dynamics,
            total_steps=total_steps,
            dt=dt,
            min_gap=0.05,
        ),
        'valid_scene_min_gap_0.00': _valid_scene(
            statics,
            dynamics,
            total_steps=total_steps,
            dt=dt,
            min_gap=0.00,
        ),
        'static_static_overlap': _static_static_overlap(statics, min_gap=0.05),
        'dynamic_static_overlap': dyn_static_overlap,
        'min_dynamic_static_gap_no_extra': min_gap_no_extra,
        'min_dynamic_static_gap_with_0.05': min_gap_with_005,
    }


if __name__ == "__main__":
    print(check_default_scene())
