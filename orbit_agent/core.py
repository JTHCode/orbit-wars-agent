"""Primary Orbit Wars agent logic extracted from the notebook without behavioral changes."""

import math
import time
from dataclasses import dataclass, field
from collections import defaultdict, deque
from itertools import combinations

# ============================================================
# Orbit Wars ROI Agent V7 - physics-helper refactor
# ============================================================
# Strategy summary:
# - Accurate enough physics helpers: future planet positions, discrete launch
#   solving, sun/path collision checks, and fleet-arrival estimation.
# - Defensive-heavy budget model: each friendly planet can spend only ships
#   above its projected safe reserve.
# - Mission marketplace: captures, early enemy pressure, recaptures, crash
#   exploits, reinforcement, snipes, swarms, salvage, and logistics compete
#   by score.
# - Global planned arrivals: every committed move is added to future combat
#   simulation so later missions know what has already been planned.
# - V7 upgrades target valuation with discounted present-value production,
#   nearest-neighbor danger, local ship-strength control, retake-risk checks,
#   safer strategic shipflow, and expiring-comet evacuation.
# ============================================================

BOARD_SIZE = 100.0
CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SHIP_SPEED = 6.0
NEUTRAL_OWNER = -1
EPISODE_STEPS = 500

EPS = 1e-9
SPAWN_PAD = 0.045
# Collision validation pads.
TARGET_HIT_INSET = 0.02
OBSTACLE_COLLISION_PAD = 0.055
SUN_PAD = 0.09

# Main tuning knobs.
DEFENSE_HORIZON = 75
FLEET_SCAN_HORIZON = 100
ATTACK_HORIZON = 100
REACTION_HORIZON = 100
MAX_MOVES = 10
MIN_LAUNCH = 2
MAX_MISSIONS_EVALUATED = 90
MAX_ATTACK_SOURCES = 7
MAX_DEFENSE_SOURCES = 8
MAX_SWARM_SOURCES = 5

# Strategy knobs.
OPENING_END = 60
MID_END = 110
PRESSURE_END = 160
SAFE_NEUTRAL_MARGIN = 2
CONTESTED_NEUTRAL_MARGIN = 4
HOSTILE_SWARM_TOL = 2
NEUTRAL_SWARM_TOL = 4
MAX_SNIPE_DELTA = 2
LOGISTICS_MIN_SEND = 9
SALVAGE_MIN_SEND = 8

# Aggression / logistics extension knobs.
EARLY_ENEMY_END = 130
EARLY_ENEMY_MAX_ETA = 24
EARLY_ENEMY_MAX_BUDGET_FRAC = 0.58
FRONTLINE_STAGING_MIN_SEND = 10
FRONTLINE_STAGING_MAX_ETA = 55
RECAPTURE_MAX_DELAY = 10
CRASH_ETA_TOL = 2

# V7 present-value scoring knobs.
PV_GAMMA = 0.992
PV_COMET_MIN_LIFE_AFTER_CAPTURE = 8

# V7 nearest-neighbor danger heuristic knobs.
NEAREST_DANGER_K = 3
NEAREST_DANGER_OPENING_END = 120
NEAREST_DANGER_MIN_MULT = 0.62
NEAREST_DANGER_MAX_MULT = 1.18

# V7 local strength / regional control knobs.
LOCAL_STRENGTH_BASE = 10.0
LOCAL_STRENGTH_SHIP_WEIGHT = 1.0
LOCAL_STRENGTH_PROD_WEIGHT = 4.0
LOCAL_STRENGTH_DIST_FLOOR = 8.0

# V7 post-capture retake-risk knobs.
RETAKE_RISK_WINDOW = 18
RETAKE_RISK_MAX_ENEMY_SOURCES = 4
RETAKE_RISK_ENEMY_AVAILABLE_FRAC = 0.68
RETAKE_RISK_HARD_REJECT = 0.82
RETAKE_RISK_SCORE_PENALTY = 0.45
RETAKE_RISK_EXTRA_MARGIN_CAP = 12

# V7 strategic shipflow / chaining knobs.
CHAIN_MIN_SEND = 9
CHAIN_MAX_ETA = 55
CHAIN_MIN_FORWARD_GAIN = 8.0
CHAIN_SAFE_RATIO_MIN = 0.58
CHAIN_RECEIVER_DEMAND_MIN = 10
CHAIN_DONOR_MAX_AVAIL_FRAC = 0.55

# V7 comet evacuation knobs.
COMET_EVAC_WINDOW = 18
COMET_EVAC_MIN_SEND = 4
COMET_EVAC_KEEP = 1
COMET_EVAC_ATTACK_BONUS = 1.20

# Timing debug. Keep False for submissions.
DEBUG_TIMING_PRINTS = False
DEBUG_PRINT_EVERY = 50


PHASE_EXPANSION_RACE = "expansion_race"
PHASE_BORDER_CONTEST = "border_contest"
PHASE_CONVERSION_PRESSURE = "conversion_pressure"
PHASE_FINAL_SCORING = "final_scoring"

PHASE_LABELS = (
    PHASE_EXPANSION_RACE,
    PHASE_BORDER_CONTEST,
    PHASE_CONVERSION_PRESSURE,
    PHASE_FINAL_SCORING,
)

PHASE_COMPAT_MAP = {
    PHASE_EXPANSION_RACE: "opening",
    PHASE_BORDER_CONTEST: "mid",
    PHASE_CONVERSION_PRESSURE: "pressure",
    PHASE_FINAL_SCORING: "endgame",
}

_PHASE_MEMORY = {}
PHASE_HISTORY_WINDOW = 6
PHASE_SWITCH_MARGIN = 0.18
PHASE_MIN_PERSISTENCE = 4


@dataclass
class PlanetObj:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: int


@dataclass
class FleetObj:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: float


@dataclass
class Arrival:
    planet_id: int
    owner: int
    ships: int
    eta: int


@dataclass
class LaunchPlan:
    source_id: int
    target_id: int
    ships: int
    angle: float
    eta: int
    score: float = 0.0
    kind: str = ""


@dataclass
class FleetStepResult:
    status: str
    x: float
    y: float
    planet: object = None
    fraction: float = 1.0


@dataclass
class Mission:
    kind: str
    score: float
    target_id: int
    plans: list = field(default_factory=list)
    required: int = 0
    eta: int = 0
    deadline: int = 0
    note: str = ""


class RuntimeStats:
    def __init__(self):
        self.turns = 0
        self.game_turn = 0
        self.game_key = None
        self.total = 0.0
        self.min_t = 10**9
        self.max_t = 0.0
        self.last = 0.0
        self.last_error = None

    def record(self, elapsed):
        self.turns += 1
        self.game_turn += 1
        self.total += elapsed
        self.last = elapsed
        if elapsed < self.min_t:
            self.min_t = elapsed
        if elapsed > self.max_t:
            self.max_t = elapsed

    def as_dict(self):
        return {
            "global_turns": self.turns,
            "game_turns": self.game_turn,
            "avg_turn_time": round(self.total / self.turns if self.turns else 0.0, 5),
            "min_turn_time": round(0.0 if self.min_t == 10**9 else self.min_t, 5),
            "max_turn_time": round(self.max_t, 5),
            "last_turn_time": round(self.last, 5),
            "last_error": self.last_error,
        }


_RUNTIME = RuntimeStats()


def get_agent_stats():
    return _RUNTIME.as_dict()


# ============================================================
# Geometry / physics helpers
# ============================================================
# Refactored with applicable math primitives from the Orbit Wars physics-helper
# notebook.  The core formulas are preserved, while wrappers keep this V5 bot's
# existing pads/clearances and calling signatures so strategy behavior stays the
# same.

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def dist_xy(x1, y1, x2, y2):
    """Physics-helper `dist`: Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


def angle_xy(x1, y1, x2, y2):
    return math.atan2(y2 - y1, x2 - x1)


def orbital_radius_xy(x, y):
    """Physics-helper `orbital_radius`: distance from the sun center."""
    return dist_xy(x, y, CENTER_X, CENTER_Y)


def is_static_planet_xy(x, y, radius):
    """Physics-helper `is_static_planet`: static when orbital_radius + radius >= 50."""
    return orbital_radius_xy(x, y) + radius >= ROTATION_RADIUS_LIMIT


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Physics-helper primitive used for continuous segment/circle checks."""
    dx = x2 - x1
    dy = y2 - y1
    seg_sq = dx * dx + dy * dy
    if seg_sq <= EPS:
        return dist_xy(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_sq
    t = clamp(t, 0.0, 1.0)
    return dist_xy(px, py, x1 + t * dx, y1 + t * dy)


def segment_intersects_circle(x1, y1, x2, y2, cx, cy, radius):
    """Physics-helper `segment_intersects_circle` boolean wrapper."""
    return point_to_segment_distance(cx, cy, x1, y1, x2, y2) <= radius


def segment_hits_sun(x1, y1, x2, y2, pad=SUN_PAD):
    """Physics-helper `segment_hits_sun`, using this bot's existing SUN_PAD."""
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) <= SUN_RADIUS + pad


def is_in_bounds(x, y):
    return 0.0 <= x <= BOARD_SIZE and 0.0 <= y <= BOARD_SIZE


def segment_bounds_fraction(x1, y1, x2, y2):
    """First fraction where a segment exits the board, else None."""
    if not is_in_bounds(x1, y1):
        return 0.0
    if is_in_bounds(x2, y2):
        return None

    dx = x2 - x1
    dy = y2 - y1
    candidates = []

    if abs(dx) > EPS:
        for xb in (0.0, BOARD_SIZE):
            t = (xb - x1) / dx
            if 0.0 <= t <= 1.0:
                y = y1 + dy * t
                if -EPS <= y <= BOARD_SIZE + EPS:
                    candidates.append(t)

    if abs(dy) > EPS:
        for yb in (0.0, BOARD_SIZE):
            t = (yb - y1) / dy
            if 0.0 <= t <= 1.0:
                x = x1 + dx * t
                if -EPS <= x <= BOARD_SIZE + EPS:
                    candidates.append(t)

    return min(candidates) if candidates else 1.0


def fleet_speed(ships, max_speed=MAX_SHIP_SPEED):
    """Physics-helper speed formula from the game spec, preserving max_speed arg."""
    ships = max(1.0, float(ships))
    if ships <= 1.0:
        return 1.0
    ratio = math.log(ships) / math.log(1000.0)
    ratio = clamp(ratio, 0.0, 1.0)
    return min(1.0 + (max_speed - 1.0) * (ratio ** 1.5), max_speed)


def segment_circle_fraction(x1, y1, x2, y2, cx, cy, radius):
    """First fraction t in [0, 1] where segment hits circle, else None."""
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    if a <= EPS:
        return 0.0 if dist_xy(x1, y1, cx, cy) <= radius else None

    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    if c <= 0:
        return 0.0

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    if 0.0 <= t1 <= 1.0:
        return t1
    if 0.0 <= t2 <= 1.0:
        return t2
    return None


def segment_circle_hit(x1, y1, x2, y2, cx, cy, radius):
    return segment_intersects_circle(x1, y1, x2, y2, cx, cy, radius)


def direct_sun_blocked(x1, y1, x2, y2, pad=SUN_PAD):
    return segment_hits_sun(x1, y1, x2, y2, pad)


def launch_point_xy(x, y, radius, angle, clearance=SPAWN_PAD):
    """Physics-helper `launch_point`, adapted to the V5 bot's SPAWN_PAD."""
    c = radius + clearance
    return x + math.cos(angle) * c, y + math.sin(angle) * c


def launch_xy(source, angle):
    return launch_point_xy(source.x, source.y, source.radius, angle, SPAWN_PAD)


def predict_planet_position_xy(planet_id, cur_x, cur_y, radius, initial_by_id, angular_velocity, turns_ahead):
    """Physics-helper planet prediction adapted for this bot's dict/objects.

    The helper notebook anchors orbital radius to initial_planets.  We keep the
    current observed angle as the starting angle, then advance by angular_velocity
    * turns_ahead.  If initial_planets are unavailable, current radius is used as
    a compatibility fallback.
    """
    init = initial_by_id.get(planet_id) if initial_by_id else None
    if init is not None:
        ix, iy = init["x"], init["y"]
        r = orbital_radius_xy(ix, iy)
        if r + radius >= ROTATION_RADIUS_LIMIT:
            return cur_x, cur_y
    else:
        r = orbital_radius_xy(cur_x, cur_y)
        if r + radius >= ROTATION_RADIUS_LIMIT:
            return cur_x, cur_y
    theta = math.atan2(cur_y - CENTER_Y, cur_x - CENTER_X) + angular_velocity * turns_ahead
    return CENTER_X + r * math.cos(theta), CENTER_Y + r * math.sin(theta)


def predict_comet_position_from_paths(comet_path_by_id, planet_id, turns):
    """Physics-helper `predict_comet_position`, adapted to cached comet paths."""
    data = comet_path_by_id.get(planet_id)
    if not data:
        return None
    path, idx = data
    if not path:
        return None
    future = idx + int(round(turns))
    if 0 <= future < len(path):
        return float(path[future][0]), float(path[future][1])
    return None


def comet_remaining_life_from_paths(comet_path_by_id, planet_id, exclude_current=True):
    """Physics-helper `comet_remaining_life`, preserving V5's exclude-current convention."""
    data = comet_path_by_id.get(planet_id)
    if not data:
        return 0
    path, idx = data
    remaining = len(path) - idx
    if exclude_current:
        remaining -= 1
    return max(0, remaining)


# ============================================================
# World model
# ============================================================

class World:
    def __init__(self, obs):
        self.obs = obs
        self.player = int(obs.get("player", 0))
        self.angular_velocity = float(obs.get("angular_velocity", 0.0))
        self.comet_ids = set(int(x) for x in obs.get("comet_planet_ids", []))

        self.planets = [
            PlanetObj(int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                      float(p[4]), float(p[5]), int(p[6]))
            for p in obs.get("planets", [])
        ]
        self.fleets = [
            FleetObj(int(f[0]), int(f[1]), float(f[2]), float(f[3]),
                     float(f[4]), int(f[5]), float(f[6]))
            for f in obs.get("fleets", [])
        ]
        self.by_id = {p.id: p for p in self.planets}
        self.initial_by_id = {}
        for p in obs.get("initial_planets", []):
            if len(p) >= 7:
                self.initial_by_id[int(p[0])] = {
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "radius": float(p[4]),
                    "production": int(p[6]),
                }
        if not self.initial_by_id:
            self.initial_by_id = {
                p.id: {"x": p.x, "y": p.y, "radius": p.radius, "production": p.production}
                for p in self.planets if p.id not in self.comet_ids
            }

        self.comet_path_by_id = {}
        self._parse_comet_paths(obs.get("comets", []))
        self.step = self._infer_step(obs)
        self._future_xy_cache = {}

        self.my_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner not in (self.player, NEUTRAL_OWNER)]
        self.neutral_planets = [p for p in self.planets if p.owner == NEUTRAL_OWNER]
        self.targets = [p for p in self.planets if p.owner != self.player]
        self.moving_planets = [
            p for p in self.planets
            if p.id in self.comet_path_by_id or self.is_orbiting(p)
        ]

        self.enemy_owner_ids = sorted(set(p.owner for p in self.enemy_planets))
        self.enemy_source_pressure = defaultdict(int)
        self.fleet_ships_by_owner = defaultdict(float)
        for f in self.fleets:
            self.fleet_ships_by_owner[f.owner] += f.ships
            if f.owner != self.player:
                self.enemy_source_pressure[f.from_planet_id] += int(f.ships)

        self.enemy_strength_by_owner = defaultdict(float)
        self.enemy_prod_by_owner = defaultdict(float)
        self.enemy_planet_count_by_owner = defaultdict(int)
        for p in self.enemy_planets:
            self.enemy_strength_by_owner[p.owner] += p.ships
            self.enemy_prod_by_owner[p.owner] += p.production
            self.enemy_planet_count_by_owner[p.owner] += 1
        for owner, ships in self.fleet_ships_by_owner.items():
            if owner != self.player:
                self.enemy_strength_by_owner[owner] += ships

        self.arrivals = self._estimate_arrivals()
        self.arrivals_by_planet = defaultdict(list)
        for a in self.arrivals:
            self.arrivals_by_planet[a.planet_id].append(a)
        for pid in self.arrivals_by_planet:
            self.arrivals_by_planet[pid].sort(key=lambda a: a.eta)

        self.modes_snapshot = self._build_modes_base()
        self.phase_state = self._init_phase_state()
        phase_signals = self.compute_phase_signals()
        phase_scores = self.compute_phase_scores(phase_signals)
        self.phase_state["current_phase"] = self.select_phase_with_hysteresis(
            phase_scores,
            self.phase_state["current_phase"],
            self.phase_state,
        )
        _PHASE_MEMORY[_RUNTIME.game_key] = {
            "current_phase": self.phase_state["current_phase"],
            "phase_persistence_turns": self.phase_state["phase_persistence_turns"],
            "last_phase_change_step": self.phase_state["last_phase_change_step"],
            "score_history": {k: list(v) for k, v in self.phase_state["score_history"].items()},
        }

        self.importance = self._planet_importance()
        self.modes = self._build_modes()
        self.reaction_cache = {}

    def _parse_comet_paths(self, comet_groups):
        for group in comet_groups:
            ids = group.get("planet_ids", [])
            paths = group.get("paths", [])
            idx = int(group.get("path_index", 0))
            for i, pid in enumerate(ids):
                if i < len(paths):
                    self.comet_path_by_id[int(pid)] = (paths[i], idx)

    def _make_game_key(self, obs):
        initial = obs.get("initial_planets") or obs.get("planets", [])
        key_rows = []
        for p in initial:
            if len(p) < 7:
                continue
            pid = int(p[0])
            if pid in self.comet_ids:
                continue
            key_rows.append((
                pid,
                round(float(p[2]), 3),
                round(float(p[3]), 3),
                round(float(p[4]), 3),
                int(p[6]),
            ))
        return tuple(sorted(key_rows))

    def _infer_step(self, obs):
        if "step" in obs:
            step = int(obs.get("step", 0))
            _RUNTIME.game_turn = step
            _RUNTIME.game_key = self._make_game_key(obs)
            return step

        key = self._make_game_key(obs)
        reset_like = len(self.fleets) == 0 and sum(1 for p in self.planets if p.owner == self.player) <= 1
        if _RUNTIME.game_key != key or (reset_like and _RUNTIME.game_turn > 20):
            _RUNTIME.game_key = key
            _RUNTIME.game_turn = 0
        return int(_RUNTIME.game_turn)

    def is_comet(self, p):
        return p.id in self.comet_ids

    def is_orbiting(self, p):
        if p.id in self.comet_ids:
            return False
        init = self.initial_by_id.get(p.id)
        if init is not None:
            return not is_static_planet_xy(init["x"], init["y"], p.radius)
        return not is_static_planet_xy(p.x, p.y, p.radius)

    def is_static(self, p):
        return (not self.is_comet(p)) and (not self.is_orbiting(p))

    def future_xy(self, p, turns):
        turns = max(0.0, float(turns))
        if abs(turns - round(turns)) < 1e-9:
            key_turn = int(round(turns))
        else:
            key_turn = round(turns, 4)
        key = (p.id, key_turn)
        cached = self._future_xy_cache.get(key, "missing")
        if cached != "missing":
            return cached

        if p.id in self.comet_path_by_id:
            ans = predict_comet_position_from_paths(self.comet_path_by_id, p.id, turns)
            self._future_xy_cache[key] = ans
            return ans

        if self.is_orbiting(p):
            ans = predict_planet_position_xy(
                p.id,
                p.x,
                p.y,
                p.radius,
                self.initial_by_id,
                self.angular_velocity,
                turns,
            )
            self._future_xy_cache[key] = ans
            return ans

        ans = (p.x, p.y)
        self._future_xy_cache[key] = ans
        return ans

    def _init_phase_state(self):
        key = _RUNTIME.game_key or self._make_game_key(self.obs)
        prev = _PHASE_MEMORY.get(key, {})
        hist = {}
        prev_hist = prev.get("score_history", {})
        for label in PHASE_LABELS:
            hist[label] = deque(prev_hist.get(label, []), maxlen=PHASE_HISTORY_WINDOW)
        return {
            "current_phase": prev.get("current_phase", PHASE_EXPANSION_RACE),
            "phase_persistence_turns": int(prev.get("phase_persistence_turns", 0)),
            "last_phase_change_step": int(prev.get("last_phase_change_step", 0)),
            "score_history": hist,
        }

    def compute_phase_signals(self):
        my_count = len(self.my_planets)
        neutral_count = len(self.neutral_planets)
        enemy_count = len(self.enemy_planets)
        total_count = max(1, len(self.planets))
        neutral_ratio = neutral_count / total_count
        enemy_ratio = enemy_count / total_count
        fleet_ratio = sum(f.ships for f in self.fleets) / max(1.0, sum(p.ships for p in self.planets))
        turns_remaining = self.turns_remaining()
        return {
            "step": float(self.step),
            "my_count": float(my_count),
            "neutral_ratio": neutral_ratio,
            "enemy_ratio": enemy_ratio,
            "domination": self.modes_snapshot["domination"],
            "prod_domination": self.modes_snapshot["prod_domination"],
            "fleet_ratio": fleet_ratio,
            "turns_remaining": float(turns_remaining),
        }

    def compute_phase_scores(self, signals):
        step = signals["step"]
        scores = {label: 0.0 for label in PHASE_LABELS}
        scores[PHASE_EXPANSION_RACE] += 1.6 * signals["neutral_ratio"] + 0.5 * (1.0 - signals["enemy_ratio"])
        scores[PHASE_BORDER_CONTEST] += 1.2 * signals["enemy_ratio"] + 0.8 * abs(signals["domination"])
        scores[PHASE_CONVERSION_PRESSURE] += 1.0 * signals["fleet_ratio"] + 1.0 * max(0.0, signals["prod_domination"])
        scores[PHASE_FINAL_SCORING] += 2.2 * (1.0 - min(1.0, signals["turns_remaining"] / 120.0))

        # Soft priors / guardrails only.
        if step < OPENING_END:
            scores[PHASE_EXPANSION_RACE] += 0.55
        elif step < MID_END:
            scores[PHASE_BORDER_CONTEST] += 0.45
        elif step < PRESSURE_END:
            scores[PHASE_CONVERSION_PRESSURE] += 0.35
        else:
            scores[PHASE_FINAL_SCORING] += 0.55
        return scores

    def select_phase_with_hysteresis(self, scores, current_phase, persistence_meta):
        history = persistence_meta["score_history"]
        for label, score in scores.items():
            history[label].append(score)
        rolling = {label: (sum(vals) / max(1, len(vals))) for label, vals in history.items()}

        best_phase = max(PHASE_LABELS, key=lambda label: rolling[label])
        if current_phase not in PHASE_LABELS:
            current_phase = PHASE_EXPANSION_RACE
        current_score = rolling[current_phase]
        best_score = rolling[best_phase]

        persistence_turns = int(persistence_meta.get("phase_persistence_turns", 0)) + 1
        if best_phase != current_phase and (
            persistence_turns < PHASE_MIN_PERSISTENCE or
            (best_score - current_score) < PHASE_SWITCH_MARGIN
        ):
            best_phase = current_phase
            persistence_turns += 1
        elif best_phase != current_phase:
            persistence_turns = 0
            persistence_meta["last_phase_change_step"] = self.step

        persistence_meta["phase_persistence_turns"] = persistence_turns
        persistence_meta["current_phase"] = best_phase
        return best_phase

    def _compat_phase_label(self, phase_label):
        return PHASE_COMPAT_MAP.get(phase_label, "opening")

    def phase(self):
        return self._compat_phase_label(self.phase_state["current_phase"])

    def turns_remaining(self):
        return max(0, EPISODE_STEPS - self.step)

    def _estimate_arrivals(self):
        arrivals = []
        for f in self.fleets:
            a = estimate_fleet_arrival(self, f)
            if a is not None:
                arrivals.append(a)
        return arrivals

    def _planet_importance(self):
        vals = {}
        sample_turns = (0, 22, 52, 90)

        for p in self.planets:
            base = 24.0 * p.production + 2.2 * p.radius
            if p.owner == self.player:
                base += 4.0
            elif p.owner == NEUTRAL_OWNER:
                base += 7.0
            else:
                base += 13.0

            # Static planets are easier to hit, hold, and use as logistics hubs.
            if self.is_static(p):
                base *= 1.12
            elif self.is_orbiting(p):
                base *= 0.96 if self.phase() == "opening" else 1.02

            hub = 0.0
            for q in self.planets:
                if q.id == p.id:
                    continue
                qv = 1.0 + 2.6 * q.production
                if q.owner not in (self.player, NEUTRAL_OWNER):
                    qv *= 1.28
                best = 999.0
                for t in sample_turns:
                    pxy = self.future_xy(p, t)
                    qxy = self.future_xy(q, t)
                    if pxy is None or qxy is None:
                        continue
                    d = dist_xy(pxy[0], pxy[1], qxy[0], qxy[1])
                    if d < best:
                        best = d
                if best < 999.0:
                    hub += qv / max(7.0, best)

            val = base + 15.0 * hub
            if p.id in self.comet_ids:
                val *= 0.43
            if dist_xy(p.x, p.y, CENTER_X, CENTER_Y) < SUN_RADIUS + p.radius + 3.0:
                val *= 0.82

            vals[p.id] = max(1.0, val)
        return vals

    def _build_modes_base(self):
        my_planet_ships = sum(p.ships for p in self.my_planets)
        my_fleet_ships = sum(f.ships for f in self.fleets if f.owner == self.player)
        my_total = my_planet_ships + my_fleet_ships
        enemy_total = 0.0
        enemy_prod = 0.0
        for p in self.enemy_planets:
            enemy_total += p.ships
            enemy_prod += p.production
        for f in self.fleets:
            if f.owner != self.player:
                enemy_total += f.ships
        my_prod = sum(p.production for p in self.my_planets)
        total = max(1.0, my_total + enemy_total)
        domination = (my_total - enemy_total) / total
        prod_domination = (my_prod - enemy_prod) / max(1.0, my_prod + enemy_prod)
        return {
            "phase": None,
            "my_total": my_total,
            "enemy_total": enemy_total,
            "my_prod": my_prod,
            "enemy_prod": enemy_prod,
            "domination": domination,
            "prod_domination": prod_domination,
            "is_behind": domination < -0.18 or prod_domination < -0.18,
            "is_ahead": domination > 0.16 or prod_domination > 0.16,
            "is_dominating": domination > 0.34 or prod_domination > 0.30,
            "is_finishing": False,
            "is_opening": False,
        }


    def _build_modes(self):
        modes = dict(self.modes_snapshot)
        compat_phase = self._compat_phase_label(self.phase_state["current_phase"])
        modes["phase"] = compat_phase
        modes["phase_label"] = self.phase_state["current_phase"]
        modes["is_opening"] = self.phase_state["current_phase"] == PHASE_EXPANSION_RACE
        modes["is_finishing"] = self.phase_state["current_phase"] == PHASE_FINAL_SCORING or self.step >= 400
        return modes

    def comet_turns_left(self, comet):
        return comet_remaining_life_from_paths(self.comet_path_by_id, comet.id, exclude_current=True)

    def approximate_reaction_time(self, owner, target):
        """Fast heuristic reaction time from any planet owned by `owner` to target."""
        key = (owner, target.id)
        if key in self.reaction_cache:
            return self.reaction_cache[key]
        best = 999
        planets = [p for p in self.planets if p.owner == owner]
        for src in planets:
            if src.id == target.id or src.ships < MIN_LAUNCH:
                continue
            pxy = self.future_xy(target, 0)
            if pxy is None:
                continue
            ang = angle_xy(src.x, src.y, pxy[0], pxy[1])
            sx, sy = launch_xy(src, ang)
            if direct_sun_blocked(sx, sy, pxy[0], pxy[1], SUN_PAD):
                continue
            probe = max(MIN_LAUNCH, min(20, int(src.ships)))
            speed = fleet_speed(probe)
            eta = int(math.ceil(max(1.0, (dist_xy(src.x, src.y, pxy[0], pxy[1]) - src.radius - target.radius) / speed)))
            if eta < best:
                best = eta
        self.reaction_cache[key] = best
        return best

    def enemy_weakness_bonus(self, target):
        """Small targeted bonus for finishing off weak enemy owners."""
        if target.owner in (self.player, NEUTRAL_OWNER):
            return 1.0
        strength = self.enemy_strength_by_owner.get(target.owner, 0.0)
        planets = self.enemy_planet_count_by_owner.get(target.owner, 0)
        if strength <= 0:
            return 1.18
        bonus = 1.0
        if strength <= 45:
            bonus += 0.22
        elif strength <= 70:
            bonus += 0.12
        if planets <= 1:
            bonus += 0.18
        elif planets == 2:
            bonus += 0.08
        if target.production <= 1 and planets > 1:
            bonus -= 0.06
        return clamp(bonus, 0.96, 1.48)

    def estimated_owner_after(self, planet, eta, extra_arrivals=None):
        owner, ships, _ = simulate_planet(self, planet, eta, extra_arrivals=extra_arrivals)
        return owner, ships

    def neutral_status(self, target):
        my_t = self.approximate_reaction_time(self.player, target)
        enemy_t = 999
        for eowner in self.enemy_owner_ids:
            enemy_t = min(enemy_t, self.approximate_reaction_time(eowner, target))
        if my_t <= enemy_t - SAFE_NEUTRAL_MARGIN:
            return "safe", my_t, enemy_t
        if abs(my_t - enemy_t) <= CONTESTED_NEUTRAL_MARGIN:
            return "contested", my_t, enemy_t
        if my_t > enemy_t + SAFE_NEUTRAL_MARGIN:
            return "enemy_favored", my_t, enemy_t
        return "neutral", my_t, enemy_t


# ============================================================
# Fleet prediction / launch solving
# ============================================================

def _skip_source_collision(planet, from_planet_id, turn_index):
    # Match game rules: collisions are continuous on each movement segment.
    # Only skip turn 0 to avoid self-collision at spawn; no multi-turn immunity.
    return from_planet_id is not None and planet.id == from_planet_id and turn_index == 0


def _moving_planet(world, planet):
    return planet.id in world.comet_path_by_id or world.is_orbiting(planet)


def advance_fleet_one_turn(world, x1, y1, angle, speed, turn_index, from_planet_id=None, target_planet_id=None):
    """Advance one fleet movement turn using the environment's turn order.

    Checks:
    1. fleet segment vs. sun / pre-rotation planets / board edge,
    2. then moving planet/comet center segment vs. the fleet's post-move point.
    """
    vx = math.cos(angle)
    vy = math.sin(angle)
    x2 = x1 + vx * speed
    y2 = y1 + vy * speed

    exit_t = segment_bounds_fraction(x1, y1, x2, y2)
    segment_limit = 1.0 if exit_t is None else max(0.0, min(1.0, exit_t))

    best_status = None
    best_planet = None
    best_t = None

    sun_t = segment_circle_fraction(
        x1, y1, x2, y2,
        CENTER_X, CENTER_Y,
        SUN_RADIUS + SUN_PAD
    )
    if sun_t is not None and sun_t <= segment_limit + EPS:
        best_status = "sun"
        best_t = sun_t

    for p in world.planets:
        if _skip_source_collision(p, from_planet_id, turn_index):
            continue
        pxy = world.future_xy(p, turn_index)
        if pxy is None:
            continue
        if target_planet_id is not None and p.id == target_planet_id:
            hit_radius = max(0.01, p.radius - TARGET_HIT_INSET)
        else:
            hit_radius = p.radius + OBSTACLE_COLLISION_PAD
        hit_t = segment_circle_fraction(
            x1, y1, x2, y2,
            pxy[0], pxy[1],
            hit_radius
        )
        if hit_t is None or hit_t > segment_limit + EPS:
            continue
        if best_t is None or hit_t < best_t - EPS:
            best_status = "hit"
            best_planet = p
            best_t = hit_t

    if best_status == "hit":
        hx = x1 + (x2 - x1) * best_t
        hy = y1 + (y2 - y1) * best_t
        return FleetStepResult("hit", hx, hy, best_planet, best_t)

    if best_status == "sun":
        hx = x1 + (x2 - x1) * best_t
        hy = y1 + (y2 - y1) * best_t
        return FleetStepResult("sun", hx, hy, None, best_t)

    if exit_t is not None:
        hx = x1 + (x2 - x1) * segment_limit
        hy = y1 + (y2 - y1) * segment_limit
        return FleetStepResult("out", hx, hy, None, segment_limit)

    # Planet rotation / comet movement sweep.  Static planets were already
    # checked above, so only moving bodies can create a new collision here.
    best_sweep = None
    best_sweep_t = None
    for p in world.moving_planets:
        if _skip_source_collision(p, from_planet_id, turn_index):
            continue
        old_xy = world.future_xy(p, turn_index)
        new_xy = world.future_xy(p, turn_index + 1)
        if old_xy is None or new_xy is None:
            continue
        if target_planet_id is not None and p.id == target_planet_id:
            hit_radius = max(0.01, p.radius - TARGET_HIT_INSET)
        else:
            hit_radius = p.radius + OBSTACLE_COLLISION_PAD
        sweep_t = segment_circle_fraction(
            old_xy[0], old_xy[1], new_xy[0], new_xy[1],
            x2, y2,
            hit_radius
        )
        if sweep_t is None:
            continue
        if best_sweep_t is None or sweep_t < best_sweep_t - EPS:
            best_sweep = p
            best_sweep_t = sweep_t

    if best_sweep is not None:
        return FleetStepResult("hit", x2, y2, best_sweep, best_sweep_t)

    return FleetStepResult("continue", x2, y2, None, 1.0)


def trace_fleet_until_collision(world, x, y, angle, ships, max_turns, from_planet_id=None, target_planet_id=None):
    speed = fleet_speed(ships)
    for turn in range(int(max(1, max_turns))):
        step = advance_fleet_one_turn(world, x, y, angle, speed, turn, from_planet_id, target_planet_id)
        if step.status == "hit":
            return step.planet, turn + 1, step
        if step.status in ("sun", "out"):
            return None, None, step
        x, y = step.x, step.y
    return None, None, FleetStepResult("continue", x, y, None, 1.0)


def estimate_fleet_arrival(world, fleet):
    planet, eta, _step = trace_fleet_until_collision(
        world,
        fleet.x,
        fleet.y,
        fleet.angle,
        fleet.ships,
        FLEET_SCAN_HORIZON,
        fleet.from_planet_id,
    )
    if planet is None:
        return None
    return Arrival(planet.id, fleet.owner, int(fleet.ships), int(eta))


def _launch_candidate_angle(source, tx, ty):
    angle = angle_xy(source.x, source.y, tx, ty)
    sx, sy = launch_xy(source, angle)
    return angle, sx, sy


def _candidate_viable(source, tx, ty, target_radius, speed, movement_turn):
    angle, sx, sy = _launch_candidate_angle(source, tx, ty)
    d_center = dist_xy(sx, sy, tx, ty)
    d_hit = max(0.0, d_center - max(0.01, target_radius - TARGET_HIT_INSET))
    seg_start = (movement_turn - 1) * speed
    seg_end = movement_turn * speed

    # Movement hit candidate if the circle boundary falls inside this movement
    # segment. Sweep candidate if the endpoint is inside/very near the target
    # after it moves. Keep this tight; path_is_clear performs the full
    # turn-order validation for accepted candidates.
    movement_gap = abs(d_hit - clamp(d_hit, seg_start, seg_end))
    endpoint_gap = abs(d_center - seg_end)
    movement_tol = max(0.08, 0.018 * speed)
    endpoint_tol = target_radius + 0.08
    return movement_gap <= movement_tol or endpoint_gap <= endpoint_tol, min(movement_gap, endpoint_gap), angle


def solve_launch_discrete(world, source, target, ships, max_turns=ATTACK_HORIZON, require_clear=True):
    """Turn-order-validated interception search. Returns a LaunchPlan or None."""
    if ships < 1 or source.id == target.id:
        return None

    speed = fleet_speed(ships)
    max_turns = int(max(1, max_turns))
    seen = set()

    for movement_turn in range(1, max_turns + 1):
        sample_turns = [movement_turn - 1, movement_turn]

        # A midpoint candidate helps orbiting planets/comets whose sweep catches
        # the fleet after the fleet movement step.
        if world.is_orbiting(target) or world.is_comet(target):
            sample_turns.append(movement_turn - 0.5)

        candidates = []
        for target_turn in sample_turns:
            pxy = world.future_xy(target, max(0.0, target_turn))
            if pxy is None:
                continue
            ok, gap, angle = _candidate_viable(
                source, pxy[0], pxy[1], target.radius, speed, movement_turn
            )
            if ok:
                candidates.append((gap, angle))

        candidates.sort(key=lambda x: x[0])

        for _gap, angle in candidates[:5]:
            key = round(angle, 7)
            if key in seen:
                continue
            seen.add(key)
            sx, sy = launch_xy(source, angle)
            if not is_in_bounds(sx, sy):
                continue

            hit_planet, hit_eta, _step = trace_fleet_until_collision(
                world,
                sx,
                sy,
                angle,
                ships,
                min(max_turns, movement_turn + 2),
                source.id,
                target.id,
            )
            if hit_planet is not None and hit_planet.id == target.id and hit_eta <= max_turns:
                return LaunchPlan(source.id, target.id, int(ships), angle, int(hit_eta))

    return None


def path_is_clear(world, source, target, angle, speed, eta):
    sx, sy = launch_xy(source, angle)
    if not is_in_bounds(sx, sy):
        return False

    ships_est = 1
    # Recover a close-enough ship estimate for callers that already computed
    # speed.  Existing callers pass speeds generated by fleet_speed(ships); the
    # trace uses speed only through advance_fleet_one_turn below.
    x, y = sx, sy
    for turn in range(int(max(1, math.ceil(eta)))):
        step = advance_fleet_one_turn(world, x, y, angle, speed, turn, source.id)
        if step.status == "hit":
            return step.planet is not None and step.planet.id == target.id
        if step.status in ("sun", "out"):
            return False
        x, y = step.x, step.y
    return False



# ============================================================
# Combat / simulation helpers
# ============================================================

def resolve_combat(planet_owner, garrison, arrivals):
    if not arrivals:
        return planet_owner, max(0.0, garrison)
    by_owner = defaultdict(int)
    for a in arrivals:
        if a.ships > 0:
            by_owner[a.owner] += int(a.ships)
    if not by_owner:
        return planet_owner, max(0.0, garrison)

    forces = sorted(by_owner.items(), key=lambda kv: kv[1], reverse=True)
    top_owner, top_ships = forces[0]
    if len(forces) >= 2:
        second_ships = forces[1][1]
        if top_ships == second_ships:
            return planet_owner, max(0.0, garrison)
        top_ships -= second_ships
    if top_ships <= 0:
        return planet_owner, max(0.0, garrison)

    if top_owner == planet_owner:
        return planet_owner, max(0.0, garrison + top_ships)
    if top_ships > garrison:
        return top_owner, max(0.0, top_ships - garrison)
    return planet_owner, max(0.0, garrison - top_ships)


def simulate_planet(world, planet, horizon, extra_arrivals=None, planned_arrivals=None, planned_departures=None):
    events = defaultdict(list)
    horizon = max(0, int(math.ceil(horizon)))

    for a in world.arrivals_by_planet.get(planet.id, []):
        if 0 < a.eta <= horizon:
            events[int(a.eta)].append(a)

    if planned_arrivals:
        for a in planned_arrivals.get(planet.id, []):
            if 0 < a.eta <= horizon:
                events[int(a.eta)].append(a)

    if extra_arrivals:
        for a in extra_arrivals:
            if a.planet_id == planet.id and 0 < a.eta <= horizon:
                events[int(a.eta)].append(a)

    owner = planet.owner
    ships = float(planet.ships)
    if planned_departures and owner == world.player:
        ships = max(0.0, ships - float(planned_departures.get(planet.id, 0)))
    last_t = 0
    timeline = []

    for eta in sorted(events.keys()):
        if owner != NEUTRAL_OWNER:
            ships += planet.production * max(0, eta - last_t)
        owner, ships = resolve_combat(owner, ships, events[eta])
        timeline.append((eta, owner, ships))
        last_t = eta

    if horizon > last_t and owner != NEUTRAL_OWNER:
        ships += planet.production * (horizon - last_t)

    return owner, ships, timeline


def defense_margin(world, planet):
    phase = world.phase()
    if phase == "opening":
        base = 1 + planet.production
    elif phase == "endgame":
        base = 3 + 2 * planet.production
    else:
        base = 2 + 2 * planet.production
    if world.modes["is_behind"]:
        base = max(1, base - 1)
    if world.modes["is_dominating"]:
        base += 1
    return int(base + min(7.0, world.importance.get(planet.id, 1.0) / 30.0))


def base_reserve(world, planet):
    phase = world.phase()
    incoming_enemy = sum(
        a.ships for a in world.arrivals_by_planet.get(planet.id, [])
        if a.owner != world.player and a.eta <= 65
    )
    if phase == "opening":
        reserve = 1 + int(0.65 * planet.production)
    elif phase == "mid":
        reserve = 2 + int(1.25 * planet.production)
    elif phase == "pressure":
        reserve = 3 + int(1.75 * planet.production)
    else:
        reserve = 5 + int(2.15 * planet.production)

    if incoming_enemy:
        reserve += int(0.55 * incoming_enemy)

    # Keep this small; actual inbound fleets should dominate reserve decisions.
    local_enemy_prod = 0.0
    for e in world.enemy_planets:
        d = dist_xy(planet.x, planet.y, e.x, e.y)
        if d < 32:
            local_enemy_prod += e.production / max(8.0, d)
    reserve += int(min(5.0, 11.0 * local_enemy_prod))

    if world.modes["is_behind"] and phase != "endgame":
        reserve = max(0, reserve - 1)
    if world.modes["is_dominating"]:
        reserve += 1
    if planet.id in world.comet_ids:
        reserve = min(reserve, 2)
    return min(int(planet.ships), max(0, reserve))


def projected_ship_budget(world, planet):
    reserve = base_reserve(world, planet)
    owner = planet.owner
    ships = float(planet.ships)
    last_t = 0
    min_surplus = ships - reserve

    events_by_eta = defaultdict(list)
    for a in world.arrivals_by_planet.get(planet.id, []):
        if 0 < a.eta <= DEFENSE_HORIZON:
            events_by_eta[a.eta].append(a)

    for eta in sorted(events_by_eta.keys()):
        if owner != NEUTRAL_OWNER:
            ships += planet.production * max(0, eta - last_t)
        owner, ships = resolve_combat(owner, ships, events_by_eta[eta])
        last_t = eta
        if owner != world.player:
            return 0
        min_surplus = min(min_surplus, ships - reserve)

    budget = int(max(0, min(planet.ships - reserve, min_surplus)))
    return min(int(planet.ships), budget)


def first_loss_eta(world, planet, horizon=DEFENSE_HORIZON, planned_arrivals=None, planned_departures=None):
    _owner, _ships, timeline = simulate_planet(
        world,
        planet,
        horizon,
        planned_arrivals=planned_arrivals,
        planned_departures=planned_departures,
    )
    for eta, owner, _ships in timeline:
        if owner != world.player:
            return eta
    return None


def required_defense_reinforcement(world, planet, loss_eta, planned_arrivals=None):
    margin = defense_margin(world, planet)

    def survives(extra):
        extras = [Arrival(planet.id, world.player, int(extra), max(1, int(loss_eta)))] if extra > 0 else []
        owner, ships, _ = simulate_planet(world, planet, max(1, int(loss_eta)), extras, planned_arrivals)
        return owner == world.player and ships >= margin

    if survives(0):
        return 0

    enemy_in = sum(
        a.ships for a in world.arrivals_by_planet.get(planet.id, [])
        if a.owner != world.player and a.eta <= loss_eta
    )
    hi = int(max(8, enemy_in + planet.ships + margin + 10))
    while not survives(hi) and hi < 6000:
        hi *= 2

    lo = 0
    for _ in range(15):
        mid = (lo + hi) // 2
        if survives(mid):
            hi = mid
        else:
            lo = mid + 1
    return int(hi)


# ============================================================
# Valuation helpers
# ============================================================

def local_support_balance(world, target, future_turn=35):
    txy = world.future_xy(target, future_turn)
    if txy is None:
        return 0.0
    tx, ty = txy
    friendly = 0.0
    enemy = 0.0
    for p in world.planets:
        if p.id == target.id:
            continue
        pxy = world.future_xy(p, future_turn)
        if pxy is None:
            continue
        d = dist_xy(tx, ty, pxy[0], pxy[1])
        val = p.production / max(8.0, d)
        if p.owner == world.player:
            friendly += val
        elif p.owner != NEUTRAL_OWNER:
            enemy += val
    return friendly - enemy


def discounted_production_value(world, target, eta, gamma=PV_GAMMA):
    """Discount future production based on when the planet will start producing for us."""
    eta = int(max(1, math.ceil(eta)))
    horizon = int(max(0, world.turns_remaining()))
    if target.id in world.comet_ids:
        comet_left = world.comet_turns_left(target)
        if comet_left <= eta + PV_COMET_MIN_LIFE_AFTER_CAPTURE:
            return 0.0
        horizon = min(horizon, int(comet_left))
    if horizon <= eta:
        return 0.0
    gamma = clamp(float(gamma), 0.80, 0.9995)
    return float(target.production) * (gamma ** eta - gamma ** horizon) / max(EPS, 1.0 - gamma)


def nearest_owner_danger_mult(world, target, future_turn=35, k=NEAREST_DANGER_K):
    """Article-style nearest-neighbor danger heuristic, with tunable neighbor count."""
    txy = world.future_xy(target, future_turn)
    if txy is None:
        return 1.0
    tx, ty = txy
    neighbors = []
    for p in world.planets:
        if p.id == target.id:
            continue
        pxy = world.future_xy(p, future_turn)
        if pxy is None:
            continue
        d = dist_xy(tx, ty, pxy[0], pxy[1])
        neighbors.append((d, p.owner, p.production))
    if not neighbors:
        return 1.0

    neighbors.sort(key=lambda x: x[0])
    k = max(1, int(k))
    chosen = neighbors[:k]
    friendly = 0.0
    enemy = 0.0
    neutral = 0.0
    for idx, (_d, owner, prod) in enumerate(chosen):
        # Keep this mostly count-based like the article, with a tiny production nudge.
        w = 1.0 / (1.0 + 0.35 * idx)
        w *= 1.0 + 0.06 * max(0, prod - 1)
        if owner == world.player:
            friendly += w
        elif owner == NEUTRAL_OWNER:
            neutral += w
        else:
            enemy += w

    total = max(EPS, friendly + enemy + neutral)
    friend_share = friendly / total
    enemy_share = enemy / total

    # Mostly a neutral-target opening/midgame heuristic. It should warn without dominating.
    mult = 1.0 + 0.30 * (friend_share - enemy_share)
    if enemy >= 1.75 and friendly <= 0.25:
        mult *= 0.76
    elif enemy > friendly + 0.75:
        mult *= 0.88
    elif friendly >= enemy + 1.0:
        mult *= 1.08

    return clamp(mult, NEAREST_DANGER_MIN_MULT, NEAREST_DANGER_MAX_MULT)


def local_strength_ratio(world, target, future_turn=35):
    """Return friendly regional ship-strength share around target at a future turn."""
    txy = world.future_xy(target, future_turn)
    if txy is None:
        return 0.5
    tx, ty = txy

    friendly = LOCAL_STRENGTH_BASE
    enemy = LOCAL_STRENGTH_BASE
    for p in world.planets:
        if p.id == target.id or p.owner == NEUTRAL_OWNER:
            continue
        pxy = world.future_xy(p, future_turn)
        if pxy is None:
            continue
        d = dist_xy(tx, ty, pxy[0], pxy[1])
        strength = (
            LOCAL_STRENGTH_SHIP_WEIGHT * max(0.0, p.ships)
            + LOCAL_STRENGTH_PROD_WEIGHT * max(0, p.production)
        ) / max(LOCAL_STRENGTH_DIST_FLOOR, d)

        if p.owner == world.player:
            friendly += strength
        else:
            enemy += strength

    return clamp(friendly / max(EPS, friendly + enemy), 0.0, 1.0)


def approximate_eta_between_planets(world, source, target, start_turn, ships):
    """Fast ETA estimate from source at start_turn to target after movement during the flight."""
    ships = max(MIN_LAUNCH, int(ships))
    speed = fleet_speed(ships)
    start_xy = world.future_xy(source, start_turn)
    if start_xy is None:
        return 999
    delay = 8
    for _ in range(4):
        target_xy = world.future_xy(target, start_turn + delay)
        if target_xy is None:
            return 999
        sx, sy = start_xy
        tx, ty = target_xy
        d = max(0.0, dist_xy(sx, sy, tx, ty) - source.radius - target.radius)
        delay = int(math.ceil(d / max(1.0, speed)))
        delay = max(1, min(999, delay))
    target_xy = world.future_xy(target, start_turn + delay)
    if target_xy is None:
        return 999
    angle = angle_xy(start_xy[0], start_xy[1], target_xy[0], target_xy[1])
    lx = start_xy[0] + math.cos(angle) * (source.radius + SPAWN_PAD)
    ly = start_xy[1] + math.sin(angle) * (source.radius + SPAWN_PAD)
    if direct_sun_blocked(lx, ly, target_xy[0], target_xy[1], SUN_PAD):
        return 999
    return delay


def post_capture_retake_risk(world, target, capture_eta, surplus_after_capture, planned_arrivals=None):
    """Estimate whether enemies can cheaply retake a target shortly after we capture it.

    This is intentionally approximate: it should price risk and add margin, not replace
    exact combat/launch solving. Known incoming fleets are still handled by simulate_planet.
    """
    if not world.enemy_planets or world.phase() == "endgame":
        return {"risk": 0.0, "extra_margin": 0, "fastest_delay": 999, "enemy_power": 0.0}

    capture_eta = int(max(1, math.ceil(capture_eta)))
    surplus = max(0.0, float(surplus_after_capture))
    txy = world.future_xy(target, capture_eta)
    if txy is None:
        return {"risk": 0.0, "extra_margin": 0, "fastest_delay": 999, "enemy_power": 0.0}

    # Only inspect nearby enemy sources. This keeps the helper cheap enough to call
    # during scoring while still catching the article's "close enemy retake" problem.
    candidates = []
    for enemy in world.enemy_planets:
        if enemy.id == target.id:
            continue
        exy = world.future_xy(enemy, capture_eta)
        if exy is None:
            continue
        d = dist_xy(exy[0], exy[1], txy[0], txy[1])
        candidates.append((d, enemy))
    candidates.sort(key=lambda x: x[0])

    risk = 0.0
    extra_margin = 0
    fastest_delay = 999
    enemy_power = 0.0

    for _d, enemy in candidates[:RETAKE_RISK_MAX_ENEMY_SOURCES]:
        owner_at_capture, ships_at_capture, _ = simulate_planet(
            world,
            enemy,
            capture_eta,
            planned_arrivals=planned_arrivals,
        )
        if owner_at_capture == world.player or owner_at_capture == NEUTRAL_OWNER:
            continue

        available = max(0.0, ships_at_capture) * RETAKE_RISK_ENEMY_AVAILABLE_FRAC
        if available < MIN_LAUNCH:
            continue

        delay = approximate_eta_between_planets(world, enemy, target, capture_eta, available)
        if delay <= 0 or delay > RETAKE_RISK_WINDOW:
            continue

        fastest_delay = min(fastest_delay, delay)
        our_when_enemy_arrives = surplus + target.production * max(0, delay)
        need_to_survive = available - our_when_enemy_arrives + 1.0
        if need_to_survive <= 0:
            # Still include small pressure when the enemy is close, but not enough to retake.
            pressure = 0.15 * (1.0 - delay / max(1.0, RETAKE_RISK_WINDOW))
            risk = max(risk, pressure)
            enemy_power = max(enemy_power, available)
            continue

        severity = clamp(need_to_survive / max(6.0, our_when_enemy_arrives + 1.0), 0.0, 1.2)
        time_factor = 1.0 - 0.55 * (delay / max(1.0, RETAKE_RISK_WINDOW))
        source_factor = 1.0 + min(0.25, enemy.production / 20.0)
        risk = max(risk, severity * time_factor * source_factor)
        extra_margin = max(extra_margin, int(math.ceil(min(RETAKE_RISK_EXTRA_MARGIN_CAP, need_to_survive))))
        enemy_power = max(enemy_power, available)

    return {
        "risk": clamp(risk, 0.0, 1.35),
        "extra_margin": int(extra_margin),
        "fastest_delay": int(fastest_delay),
        "enemy_power": float(enemy_power),
    }



def target_extra_margin(world, target, eta):
    eta_i = min(80, max(10, int(math.ceil(eta))))
    support = local_support_balance(world, target, eta_i)
    pressure = max(0.0, -support)
    strength_ratio = local_strength_ratio(world, target, eta_i)
    danger_mult = nearest_owner_danger_mult(world, target, eta_i, NEAREST_DANGER_K)
    importance = world.importance.get(target.id, 1.0)

    if target.owner == NEUTRAL_OWNER:
        if world.phase() == "opening":
            margin = 1 + (1 if target.production >= 4 else 0)
        else:
            margin = 1 + int(0.45 * target.production + min(4.0, importance / 45.0))
        status, _my_t, _enemy_t = world.neutral_status(target)
        if status == "contested":
            margin += 2
        elif status == "enemy_favored":
            margin += 3

        # V7: if nearest planets and local strength say the area is enemy-leaning,
        # take neutrals with more surplus instead of tiny one-ship captures.
        if danger_mult < 0.90:
            margin += int(math.ceil((0.90 - danger_mult) * 8.0))
        if strength_ratio < 0.45:
            margin += int(math.ceil((0.45 - strength_ratio) * 12.0))
    else:
        margin = 2 + int(1.55 * target.production + min(10.0, importance / 24.0) + min(8.0, 28.0 * pressure))
        if world.enemy_source_pressure.get(target.id, 0) > 0:
            margin = max(1, margin - 1)
        if strength_ratio < 0.40:
            margin += int(math.ceil((0.40 - strength_ratio) * 10.0))

    if world.modes["is_behind"]:
        margin = max(1, int(margin * 0.82))
    elif world.modes["is_dominating"]:
        margin = int(margin * 1.15) + 1

    if world.phase() == "endgame":
        margin = max(1, int(margin * 0.45))
    if target.id in world.comet_ids:
        margin = min(margin, 2)
    return max(1, margin)



def target_required_ships(world, target, eta, planned_arrivals=None):
    eta = max(1, int(math.ceil(eta)))
    owner, ships, _ = simulate_planet(world, target, eta, planned_arrivals=planned_arrivals)
    if owner == world.player:
        return 0
    return int(math.floor(ships)) + 1 + target_extra_margin(world, target, eta)


def target_roi_score(world, target, required, eta, source_dist, planned_arrivals=None):
    eta = int(max(1, math.ceil(eta)))

    if eta > world.turns_remaining() + 2:
        return -1e9

    if target.id in world.comet_ids:
        comet_left = world.comet_turns_left(target)
        if comet_left <= eta + PV_COMET_MIN_LIFE_AFTER_CAPTURE or eta > 55:
            return -1e9

    # V7: production is valued by discounted present value instead of raw turns-left.
    prod_value = discounted_production_value(world, target, eta, PV_GAMMA)
    if prod_value <= 0.0 and not world.modes["is_finishing"]:
        return -1e9

    if target.owner not in (world.player, NEUTRAL_OWNER):
        # Capturing enemy production is both a gain for us and a denial to them.
        prod_value *= 2.12
    elif target.owner == NEUTRAL_OWNER:
        status, my_t, enemy_t = world.neutral_status(target)
        if status == "safe":
            prod_value *= 1.22
        elif status == "contested":
            prod_value *= 0.92
        elif status == "enemy_favored":
            prod_value *= 0.72

    # Late game values immediate ship swing more and production less.
    if world.modes["is_finishing"]:
        prod_value *= 0.45
        if target.owner not in (world.player, NEUTRAL_OWNER):
            prod_value += min(160.0, target.ships * 1.05)
        else:
            prod_value += min(55.0, target.ships * 0.22)

    punish = 1.0
    if target.owner not in (world.player, NEUTRAL_OWNER):
        launched = world.enemy_source_pressure.get(target.id, 0)
        if launched > 0:
            punish += min(0.55, launched / max(18.0, target.ships + launched))
        punish *= world.enemy_weakness_bonus(target)

    support = local_support_balance(world, target, min(75, max(12, eta)))
    hold_mult = 1.0 + clamp(0.36 * support, -0.30, 0.30)

    # V7: regional military control based on ships/distance.
    strength_ratio = local_strength_ratio(world, target, min(75, max(12, eta)))
    strength_mult = 0.76 + 0.54 * strength_ratio

    # V7: opening/midgame nearest-owner danger heuristic from the article.
    nearest_mult = 1.0
    if target.owner == NEUTRAL_OWNER and world.step < NEAREST_DANGER_OPENING_END:
        nearest_mult = nearest_owner_danger_mult(world, target, min(75, max(12, eta)), NEAREST_DANGER_K)

    # V7: approximate post-capture retake risk. Estimate capture surplus from the
    # proposed send amount, then discount/reject fragile captures near enemy sources.
    retake = {"risk": 0.0, "extra_margin": 0}
    if world.phase() != "endgame":
        owner_at_eta, ships_at_eta, _ = simulate_planet(world, target, eta, planned_arrivals=planned_arrivals)
        if owner_at_eta != world.player:
            surplus_est = max(0.0, float(required) - max(0.0, ships_at_eta))
        else:
            surplus_est = max(0.0, float(required) + max(0.0, ships_at_eta))
        retake = post_capture_retake_risk(world, target, eta, surplus_est, planned_arrivals)
    retake_mult = 1.0 - RETAKE_RISK_SCORE_PENALTY * min(1.0, retake["risk"])
    if target.owner != NEUTRAL_OWNER:
        # Enemy attacks should not become too timid; sometimes disruption is still useful.
        retake_mult = max(retake_mult, 0.72)
    else:
        retake_mult = max(retake_mult, 0.45)

    importance_mult = 0.78 + min(0.42, world.importance.get(target.id, 1.0) / 170.0)

    static_mult = 1.0
    if world.is_static(target):
        if target.owner == NEUTRAL_OWNER:
            static_mult = 1.18 if world.phase() == "opening" else 1.10
        elif target.owner != world.player:
            static_mult = 1.14
    elif world.is_orbiting(target) and world.phase() == "opening" and target.owner == NEUTRAL_OWNER:
        # Opening filter: avoid risky rotating neutrals unless valuable and reachable.
        if target.production <= 3 and eta > 26:
            static_mult = 0.55
        else:
            static_mult = 0.82

    if world.modes["is_behind"]:
        if target.owner == NEUTRAL_OWNER:
            mode_mult = 1.15
        else:
            mode_mult = 0.88
    elif world.modes["is_dominating"]:
        mode_mult = 1.18 if target.owner != NEUTRAL_OWNER else 0.94
    else:
        mode_mult = 1.0

    # PV already discounts slow captures, so the explicit travel penalty is softened.
    travel_penalty = 1.0 + 0.018 * eta + 0.0035 * source_dist
    cost = max(1.0, required) ** 0.94
    score = (
        prod_value
        * punish
        * hold_mult
        * strength_mult
        * nearest_mult
        * retake_mult
        * importance_mult
        * static_mult
        * mode_mult
        / (cost * travel_penalty)
    )

    if world.phase() == "opening":
        if target.owner == NEUTRAL_OWNER:
            score *= 1.32
        else:
            score *= 0.88
    elif world.phase() == "endgame":
        if eta > world.turns_remaining() - 3:
            return -1e9
        score *= 0.70 if target.owner == NEUTRAL_OWNER else 1.15

    if target.id in world.comet_ids:
        # Comet PV already accounts for remaining life. Keep a modest penalty because
        # comet garrisons can disappear, but do not suppress all comet opportunities.
        score *= 0.72
    return score



# ============================================================
# Mission planner
# ============================================================

class Planner:
    def __init__(self, world):
        self.w = world
        self.moves = []
        self.committed = defaultdict(int)
        self.available = {p.id: projected_ship_budget(world, p) for p in world.my_planets}
        self.my_by_id = {p.id: p for p in world.my_planets}
        self.planned_arrivals = defaultdict(list)
        self.claimed_targets = set()
        self.missions = []
        self.shot_cache = {}
        self.source_info_cache = {}
        self.direct_score_cache = {}
        self.start_time = time.perf_counter()

    def current_available(self, p):
        return max(0, min(self.available.get(p.id, 0), int(p.ships) - self.committed[p.id]))

    def time_exceeded(self, limit=0.82):
        return (time.perf_counter() - self.start_time) > limit

    def solve_launch(self, source, target, ships, max_turns=ATTACK_HORIZON, require_clear=True):
        ships = int(ships)
        if ships < 1:
            return None
        key = (source.id, target.id, ships, int(max_turns), bool(require_clear))
        cached = self.shot_cache.get(key, "missing")
        if cached != "missing":
            if cached is None:
                return None
            return LaunchPlan(
                cached.source_id,
                cached.target_id,
                cached.ships,
                cached.angle,
                cached.eta,
                cached.score,
                cached.kind,
            )
        solved = solve_launch_discrete(self.w, source, target, ships, max_turns, require_clear)
        self.shot_cache[key] = solved
        if solved is None:
            return None
        return LaunchPlan(
            solved.source_id,
            solved.target_id,
            solved.ships,
            solved.angle,
            solved.eta,
            solved.score,
            solved.kind,
        )

    def commit_resolved_launch(self, plan):
        if len(self.moves) >= MAX_MOVES:
            return False
        source = self.w.by_id.get(plan.source_id)
        target = self.w.by_id.get(plan.target_id)
        if source is None or target is None or source.owner != self.w.player:
            return False
        ships = int(plan.ships)
        if ships < MIN_LAUNCH or ships > self.current_available(source):
            return False
        self.moves.append([int(source.id), float(plan.angle), int(ships)])
        self.committed[source.id] += ships
        self.available[source.id] = max(0, self.available.get(source.id, 0) - ships)
        self.add_planned_arrival(plan)
        return True

    def resolve_mission_plans(self, mission):
        resolved = []
        temp_used = defaultdict(int)
        for plan in mission.plans:
            source = self.w.by_id.get(plan.source_id)
            target = self.w.by_id.get(plan.target_id)
            if source is None or target is None or source.owner != self.w.player:
                return None
            ships = int(plan.ships)
            if ships < MIN_LAUNCH:
                return None
            if ships > self.current_available(source) - temp_used[source.id]:
                return None
            solved = self.solve_launch(source, target, ships, require_clear=True)
            if solved is None:
                return None
            solved.score = plan.score
            solved.kind = plan.kind
            resolved.append(solved)
            temp_used[source.id] += ships
        return resolved

    def add_planned_arrival(self, plan):
        self.planned_arrivals[plan.target_id].append(
            Arrival(plan.target_id, self.w.player, int(plan.ships), int(plan.eta))
        )

    def add_launch(self, plan):
        source = self.w.by_id.get(plan.source_id)
        target = self.w.by_id.get(plan.target_id)
        if source is None or target is None:
            return False
        solved = self.solve_launch(source, target, int(plan.ships), require_clear=True)
        if solved is None:
            return False
        solved.score = plan.score
        solved.kind = plan.kind
        return self.commit_resolved_launch(solved)

    def plan(self):
        self.build_missions()
        self.execute_missions()
        if not self.time_exceeded(0.68):
            self.salvage_expiring_comets()
        if not self.time_exceeded(0.76):
            self.strategic_shipflow_logistics()
        if not self.time_exceeded(0.82):
            self.followup_attacks()
        if not self.time_exceeded(0.88):
            self.salvage_doomed_planets()
        if not self.time_exceeded(0.94):
            self.logistics_funnel()
        return self.moves

    # --------------------------------------------------------
    # Mission generation
    # --------------------------------------------------------

    def build_missions(self):
        missions = []
        missions.extend(self.build_emergency_defense_missions())
        missions.extend(self.build_planned_recapture_missions())
        missions.extend(self.build_early_enemy_opportunity_missions())
        missions.extend(self.build_capture_missions())
        missions.extend(self.build_crash_exploit_missions())
        missions.extend(self.build_snipe_missions())
        missions.extend(self.build_swarm_missions())
        missions.extend(self.build_scored_reinforcement_missions())
        missions.sort(reverse=True, key=lambda m: m.score)
        self.missions = missions[:MAX_MISSIONS_EVALUATED]

    def build_emergency_defense_missions(self):
        out = []
        for p in self.w.my_planets:
            loss_eta = first_loss_eta(self.w, p, DEFENSE_HORIZON, self.planned_arrivals)
            if loss_eta is None:
                continue
            need = required_defense_reinforcement(self.w, p, loss_eta, self.planned_arrivals)
            if need <= 0:
                continue
            plans = self.plan_reinforcement_to(p, loss_eta, need)
            if not plans:
                continue
            imp = self.w.importance.get(p.id, 1.0)
            emergency = 260.0 / max(1, loss_eta)
            score = 55.0 + emergency + 0.55 * imp + 3.0 * p.production
            # Do not over-save low-value doomed planets unless help is cheap.
            if p.production <= 1 and loss_eta > 18 and sum(x.ships for x in plans) > p.ships + 8:
                score *= 0.58
            out.append(Mission("emergency_defense", score, p.id, plans, need, loss_eta, loss_eta))
        return out

    def build_scored_reinforcement_missions(self):
        out = []
        for p in self.w.my_planets:
            loss_eta = first_loss_eta(self.w, p, DEFENSE_HORIZON + 35, self.planned_arrivals)
            if loss_eta is None or loss_eta <= 12:
                continue
            need = required_defense_reinforcement(self.w, p, loss_eta, self.planned_arrivals)
            if need <= 0:
                continue
            plans = self.plan_reinforcement_to(p, loss_eta, need)
            if not plans:
                continue
            sent = sum(pl.ships for pl in plans)
            if sent <= 0:
                continue
            imp = self.w.importance.get(p.id, 1.0)
            score = (imp + 12.0 * p.production) / max(6.0, sent) + 65.0 / max(5, loss_eta)
            if self.w.modes["is_behind"] and p.production <= 2:
                score *= 0.70
            out.append(Mission("reinforce", score, p.id, plans, need, loss_eta, loss_eta))
        return out

    def build_planned_recapture_missions(self):
        """Let doomed friendly planets fall, then retake them when that is cheaper than saving them."""
        out = []
        for target in self.w.my_planets:
            loss_eta = first_loss_eta(self.w, target, DEFENSE_HORIZON + 45, self.planned_arrivals, self.committed)
            if loss_eta is None or loss_eta < 3:
                continue

            owner_at_loss, ships_at_loss, _ = simulate_planet(
                self.w,
                target,
                loss_eta,
                planned_arrivals=self.planned_arrivals,
                planned_departures=self.committed,
            )
            if owner_at_loss == self.w.player or ships_at_loss > 45:
                continue
            if target.production <= 1 and self.w.importance.get(target.id, 1.0) < 65:
                continue

            best = None
            for src in self.w.my_planets:
                if src.id == target.id:
                    continue
                avail = self.current_available(src)
                if avail < MIN_LAUNCH:
                    continue
                # Try to arrive shortly after the predicted loss.
                for delay in (1, 2, 4, 7, RECAPTURE_MAX_DELAY):
                    eta_goal = int(loss_eta + delay)
                    required = target_required_ships(self.w, target, eta_goal, self.planned_arrivals)
                    if required < MIN_LAUNCH or required > avail:
                        continue
                    plan = self.settle_capture_plan(src, target, required, max_turns=min(ATTACK_HORIZON, eta_goal + 12))
                    if plan is None:
                        continue
                    if plan.eta < loss_eta or plan.eta > loss_eta + RECAPTURE_MAX_DELAY:
                        continue
                    if not self.group_captures(target, [plan]):
                        continue
                    recapture_gap = max(1, plan.eta - loss_eta)
                    score = (0.78 * self.w.importance.get(target.id, 1.0) + 12.0 * target.production) / max(8.0, plan.ships)
                    score += 12.0 / recapture_gap
                    if target.production >= 4:
                        score *= 1.16
                    plan.kind = "recapture"
                    plan.score = score
                    m = Mission("recapture", score, target.id, [plan], plan.ships, plan.eta, loss_eta)
                    if best is None or m.score > best.score:
                        best = m
                    break
            if best is not None and best.score > 1.05:
                out.append(best)
        return out

    def build_early_enemy_opportunity_missions(self):
        """Targeted opening/midgame enemy pressure without globally lowering all attack thresholds."""
        out = []
        if self.w.step >= EARLY_ENEMY_END or not self.w.enemy_planets:
            return out

        targets = sorted(
            [t for t in self.w.enemy_planets if self.target_allowed(t)],
            key=lambda p: self.target_priority(p),
            reverse=True,
        )[:12]
        for target in targets:
            if self.time_exceeded(0.60):
                break
            weak_owner = self.w.enemy_strength_by_owner.get(target.owner, 999.0) <= 75
            recently_launched = self.w.enemy_source_pressure.get(target.id, 0) > 0
            if target.production < 2 and not weak_owner and not recently_launched:
                continue
            if self.w.is_orbiting(target) and target.production <= 3:
                # Early rotating targets are okay only when close/cheap; source validation below enforces ETA.
                pass

            best = None
            for info in self.source_infos_for_target(target):
                src = info["src"]
                avail = self.current_available(src)
                if avail < MIN_LAUNCH:
                    continue
                required = target_required_ships(self.w, target, info["eta"], self.planned_arrivals)
                if required < MIN_LAUNCH:
                    continue
                if required > avail or required > max(MIN_LAUNCH, int(avail * EARLY_ENEMY_MAX_BUDGET_FRAC)):
                    continue
                plan = self.settle_capture_plan(src, target, required, max_turns=EARLY_ENEMY_MAX_ETA + 6)
                if plan is None or plan.eta > EARLY_ENEMY_MAX_ETA:
                    continue
                if self.w.is_orbiting(target) and plan.eta > 18 and target.production <= 3:
                    continue
                if not self.group_captures(target, [plan]):
                    continue
                d = info["dist"]
                score = target_roi_score(self.w, target, plan.ships, plan.eta, d, self.planned_arrivals)
                score *= 1.34
                score += 0.12 * self.w.importance.get(target.id, 1.0)
                if recently_launched:
                    score += 7.0
                if weak_owner:
                    score += 5.0
                if target.production >= 4:
                    score += 4.0
                if score < 0.80:
                    continue
                plan.kind = "early_enemy"
                plan.score = score
                m = Mission("early_enemy", score, target.id, [plan], plan.ships, plan.eta, plan.eta)
                if best is None or m.score > best.score:
                    best = m
            if best is not None:
                out.append(best)
        return out

    def build_crash_exploit_missions(self):
        """In multi-enemy games, arrive after enemy-vs-enemy arrivals weaken a planet."""
        out = []
        if len(self.w.enemy_owner_ids) < 2:
            return out
        for target in self.w.targets:
            if not self.target_allowed(target):
                continue
            enemy_arrivals = [
                a for a in self.w.arrivals_by_planet.get(target.id, [])
                if a.owner != self.w.player and a.eta <= ATTACK_HORIZON - 2
            ]
            if len(enemy_arrivals) < 2:
                continue
            enemy_arrivals.sort(key=lambda a: a.eta)
            best = None
            for i in range(len(enemy_arrivals)):
                for j in range(i + 1, min(len(enemy_arrivals), i + 5)):
                    a = enemy_arrivals[i]
                    b = enemy_arrivals[j]
                    if a.owner == b.owner:
                        continue
                    if abs(a.eta - b.eta) > CRASH_ETA_TOL:
                        continue
                    crash_eta = max(a.eta, b.eta) + 1
                    required = target_required_ships(self.w, target, crash_eta, self.planned_arrivals)
                    if required < MIN_LAUNCH:
                        continue
                    plan = self.best_source_for_required(target, required, eta_hint=crash_eta, eta_tolerance=CRASH_ETA_TOL + 3)
                    if plan is None:
                        continue
                    if not self.group_captures(target, [plan]):
                        continue
                    d = dist_xy(self.w.by_id[plan.source_id].x, self.w.by_id[plan.source_id].y, target.x, target.y)
                    score = target_roi_score(self.w, target, plan.ships, plan.eta, d, self.planned_arrivals)
                    score += 10.0 / max(1, abs(plan.eta - crash_eta) + 1)
                    score += min(12.0, 0.10 * (a.ships + b.ships))
                    if target.owner != NEUTRAL_OWNER:
                        score *= 1.10
                    plan.kind = "crash_exploit"
                    plan.score = score
                    m = Mission("crash_exploit", score, target.id, [plan], plan.ships, plan.eta, crash_eta)
                    if best is None or m.score > best.score:
                        best = m
            if best is not None and best.score > self.capture_score_floor(target):
                out.append(best)
        return out

    def build_capture_missions(self):
        out = []
        targets = sorted(
            [t for t in self.w.targets if self.target_allowed(t)],
            key=lambda p: self.target_priority(p),
            reverse=True,
        )[:26]
        for target in targets:
            if self.time_exceeded(0.62):
                break
            best = self.best_single_capture_mission(target)
            if best is not None:
                out.append(best)
        return out

    def build_snipe_missions(self):
        # Snipe neutral targets where enemy arrivals are already committed.
        out = []
        for target in self.w.neutral_planets:
            if not self.target_allowed(target):
                continue
            enemy_arrivals = [a for a in self.w.arrivals_by_planet.get(target.id, []) if a.owner != self.w.player]
            if not enemy_arrivals:
                continue
            for enemy_a in enemy_arrivals[:3]:
                snipe_eta = enemy_a.eta + 1
                if snipe_eta > ATTACK_HORIZON:
                    continue
                extra_margin = 2 + (1 if target.production >= 4 else 0)
                # Required ships after the enemy combat has resolved.
                required = target_required_ships(self.w, target, snipe_eta, self.planned_arrivals) + extra_margin
                plan = self.best_source_for_required(target, required, eta_hint=snipe_eta, eta_tolerance=MAX_SNIPE_DELTA + 1)
                if plan is None:
                    continue
                d = dist_xy(self.w.by_id[plan.source_id].x, self.w.by_id[plan.source_id].y, target.x, target.y)
                base = target_roi_score(self.w, target, required, plan.eta, d, self.planned_arrivals)
                timing_bonus = 8.0 / max(1, abs(plan.eta - snipe_eta) + 1)
                score = base * 1.25 + timing_bonus + 0.22 * self.w.importance.get(target.id, 1.0)
                plan.kind = "snipe"
                out.append(Mission("snipe", score, target.id, [plan], required, plan.eta, snipe_eta))
                break
        return out

    def build_swarm_missions(self):
        out = []
        valuable_targets = sorted(
            [t for t in self.w.targets if self.target_allowed(t)],
            key=lambda p: self.target_priority(p) * (1.18 if p.owner != NEUTRAL_OWNER else 1.0),
            reverse=True
        )[:12]
        for target in valuable_targets:
            if self.time_exceeded(0.70):
                break
            source_infos = self.source_infos_for_target(target)
            if len(source_infos) < 2:
                continue
            tol = NEUTRAL_SWARM_TOL if target.owner == NEUTRAL_OWNER else max(HOSTILE_SWARM_TOL, 4)
            best = None
            # Try pairs/triples, and allow 4-source hostile swarms for tougher enemy planets.
            limited = source_infos[:10]
            swarm_sizes = (2, 3, 4) if target.owner != NEUTRAL_OWNER else (2, 3)
            for r in swarm_sizes:
                if r > len(limited):
                    continue
                for combo in combinations(limited, r):
                    etas = [c["eta"] for c in combo]
                    if max(etas) - min(etas) > tol:
                        continue
                    eta_est = max(etas)
                    required = target_required_ships(self.w, target, eta_est, self.planned_arrivals)
                    if required < MIN_LAUNCH:
                        continue
                    if sum(self.current_available(c["src"]) for c in combo) < required:
                        continue
                    plans = self.allocate_swarm(target, list(combo), required)
                    if not plans:
                        continue
                    actual_eta = max(p.eta for p in plans)
                    required2 = target_required_ships(self.w, target, actual_eta, self.planned_arrivals)
                    if required2 > sum(p.ships for p in plans):
                        plans = self.allocate_swarm(target, list(combo), required2)
                        if not plans:
                            continue
                        actual_eta = max(p.eta for p in plans)
                    if not self.group_captures(target, plans):
                        continue
                    sent = sum(p.ships for p in plans)
                    d = min(c["dist"] for c in combo)
                    score = target_roi_score(self.w, target, max(required, sent), actual_eta, d, self.planned_arrivals)
                    score *= 0.96 if r == 2 else (0.88 if r == 3 else 0.80)
                    score -= 0.025 * sent
                    if score < self.capture_score_floor(target):
                        continue
                    if best is None or score > best.score:
                        best = Mission("swarm", score, target.id, plans, required, actual_eta, actual_eta)
            if best is not None:
                out.append(best)
        return out

    # --------------------------------------------------------
    # Mission utilities
    # --------------------------------------------------------

    def target_priority(self, target):
        """Rank targets for expensive launch-solving work."""
        value = self.w.importance.get(target.id, 1.0)
        if target.owner == NEUTRAL_OWNER:
            status, _my_t, _enemy_t = self.w.neutral_status(target)
            if status == "safe":
                value *= 1.10
            elif status == "contested":
                value *= 1.04
            elif status == "enemy_favored":
                value *= 0.78
            if self.w.phase() == "opening":
                value *= 1.08
        elif target.owner != self.w.player:
            value *= 1.24 * self.w.enemy_weakness_bonus(target)
            launched = self.w.enemy_source_pressure.get(target.id, 0)
            if launched > 0:
                value *= 1.0 + min(0.22, launched / max(45.0, target.ships + launched))
            if self.w.step < EARLY_ENEMY_END and target.ships <= 28 + 4 * target.production:
                value *= 1.18
            if self.w.enemy_planet_count_by_owner.get(target.owner, 0) <= 2:
                value *= 1.10
        if self.w.is_static(target):
            value *= 1.05
        elif self.w.is_orbiting(target) and self.w.phase() == "opening" and target.production <= 3:
            value *= 0.82
        if target.id in self.w.comet_ids:
            left = self.w.comet_turns_left(target)
            # V7: comets are less valuable than permanent planets, but not useless
            # when they have enough life left or can act as temporary staging assets.
            value *= 0.50 + min(0.24, max(0, left - 12) / 80.0)
        return value

    def target_allowed(self, target):
        if target.id in self.w.comet_ids:
            left = self.w.comet_turns_left(target)
            if left < PV_COMET_MIN_LIFE_AFTER_CAPTURE + 4:
                return False
        if self.w.phase() == "opening" and self.w.is_orbiting(target) and target.owner == NEUTRAL_OWNER:
            status, my_t, enemy_t = self.w.neutral_status(target)
            if target.production <= 3 and (my_t > 26 or status == "enemy_favored"):
                return False
        return True

    def capture_score_floor(self, target):
        phase = self.w.phase()
        if target.owner == NEUTRAL_OWNER:
            if phase == "opening":
                floor = 0.40
            elif phase == "mid":
                floor = 0.70
            elif phase == "pressure":
                floor = 0.90
            else:
                floor = 1.20
        else:
            if self.w.modes["is_dominating"]:
                floor = 0.70
            elif self.w.modes["is_behind"]:
                floor = 1.25
            elif phase == "endgame":
                floor = 0.85
            else:
                floor = 1.00
        if target.id in self.w.comet_ids:
            # Remaining-life PV now handles most comet selectivity.
            floor += 0.18
        return floor

    def source_infos_for_target(self, target):
        cache_key = target.id
        cached = self.source_info_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        infos = []
        txy0 = self.w.future_xy(target, 0)
        if txy0 is None:
            return infos

        candidates = []
        for src in self.w.my_planets:
            avail = self.current_available(src)
            if avail < MIN_LAUNCH:
                continue
            if target.owner not in (self.w.player, NEUTRAL_OWNER):
                rough_share = int(math.ceil((target.ships + 8 + 4 * target.production) / 3.0))
                probe = min(avail, max(MIN_LAUNCH, min(avail, rough_share)))
            else:
                probe = min(avail, max(MIN_LAUNCH, min(12, avail)))

            speed = fleet_speed(probe)
            approx_dist = max(0.0, dist_xy(src.x, src.y, txy0[0], txy0[1]) - src.radius - target.radius)
            approx_eta = approx_dist / max(1.0, speed)
            if approx_eta > ATTACK_HORIZON + 8:
                continue

            # Cheap current-ray sun filter. Orbiting targets may become shootable
            # later, so only treat this as a sorting penalty.
            current_angle = angle_xy(src.x, src.y, txy0[0], txy0[1])
            sx, sy = launch_xy(src, current_angle)
            sun_penalty = 18.0 if direct_sun_blocked(sx, sy, txy0[0], txy0[1], SUN_PAD) else 0.0
            score = approx_eta + sun_penalty - 0.018 * avail
            candidates.append((score, src, avail, probe, approx_dist))

        candidates.sort(key=lambda x: x[0])
        for _score, src, avail, probe, approx_dist in candidates[:MAX_ATTACK_SOURCES]:
            sol = self.solve_launch(src, target, probe, require_clear=True)
            if sol is None or sol.eta > ATTACK_HORIZON:
                continue
            infos.append({
                "src": src,
                "avail": avail,
                "eta": sol.eta,
                "dist": approx_dist,
            })
        infos.sort(key=lambda d: (d["eta"], -d["avail"]))
        self.source_info_cache[cache_key] = list(infos)
        return infos

    def settle_capture_plan(self, source, target, initial_send, max_turns=ATTACK_HORIZON, extra_margin=0):
        """Iteratively settle ship count/ETA because fleet speed depends on ships."""
        avail = self.current_available(source)
        if avail < MIN_LAUNCH:
            return None
        send = int(clamp(int(math.ceil(initial_send + extra_margin)), MIN_LAUNCH, avail))
        best = None
        seen = set()
        for _ in range(4):
            if send < MIN_LAUNCH or send > avail:
                return None
            marker = send
            if marker in seen:
                break
            seen.add(marker)
            sol = self.solve_launch(source, target, send, max_turns=max_turns, require_clear=True)
            if sol is None or sol.eta > max_turns:
                return None
            required = target_required_ships(self.w, target, sol.eta, self.planned_arrivals)
            required = int(math.ceil(required + extra_margin))
            if required < MIN_LAUNCH:
                return None
            if required > avail:
                return None
            best = sol
            if abs(required - send) <= 1:
                send = max(send, required)
                break
            # Avoid oscillation from speed/ETA feedback by biasing upward slightly.
            send = max(required, min(avail, int(math.ceil(0.55 * send + 0.45 * required))))
        if best is None:
            return None
        if best.ships != send:
            best = self.solve_launch(source, target, send, max_turns=max_turns, require_clear=True)
            if best is None:
                return None
        required_final = target_required_ships(self.w, target, best.eta, self.planned_arrivals)
        required_final = int(math.ceil(required_final + extra_margin))
        if self.w.phase() != "endgame":
            owner_at_eta, ships_at_eta, _ = simulate_planet(self.w, target, best.eta, planned_arrivals=self.planned_arrivals)
            surplus_est = max(0.0, float(required_final) - max(0.0, ships_at_eta if owner_at_eta != self.w.player else 0.0))
            retake = post_capture_retake_risk(self.w, target, best.eta, surplus_est, self.planned_arrivals)
            if retake["extra_margin"] > 0:
                required_final = int(math.ceil(required_final + retake["extra_margin"]))
        if required_final > best.ships:
            if required_final > avail:
                return None
            best = self.solve_launch(source, target, required_final, max_turns=max_turns, require_clear=True)
            if best is None:
                return None
        best.ships = int(best.ships)
        return best

    def best_single_capture_mission(self, target):
        best = None
        for info in self.source_infos_for_target(target):
            src = info["src"]
            eta = info["eta"]
            required = target_required_ships(self.w, target, eta, self.planned_arrivals)
            if required < MIN_LAUNCH:
                continue
            sol = self.settle_capture_plan(src, target, required)
            if sol is None or sol.eta > ATTACK_HORIZON:
                continue
            if not self.group_captures(target, [sol]):
                continue
            sol.kind = "capture"
            score = target_roi_score(self.w, target, sol.ships, sol.eta, info["dist"], self.planned_arrivals)
            if score < self.capture_score_floor(target):
                continue
            sol.score = score
            mission_kind = "capture_neutral" if target.owner == NEUTRAL_OWNER else "capture_enemy"
            m = Mission(mission_kind, score, target.id, [sol], sol.ships, sol.eta, sol.eta)
            if best is None or m.score > best.score:
                best = m
        return best

    def best_source_for_required(self, target, required, eta_hint=None, eta_tolerance=999):
        best = None
        for src in self.w.my_planets:
            avail = self.current_available(src)
            if avail < required or required < MIN_LAUNCH:
                continue
            sol = self.settle_capture_plan(src, target, required)
            if sol is None or sol.eta > ATTACK_HORIZON:
                continue
            if eta_hint is not None and abs(sol.eta - eta_hint) > eta_tolerance:
                continue
            d = dist_xy(src.x, src.y, target.x, target.y)
            score = -abs((eta_hint if eta_hint is not None else sol.eta) - sol.eta) * 4.0 - 0.02 * d + 0.01 * avail
            if best is None or score > best[0]:
                best = (score, sol)
        return None if best is None else best[1]

    def allocate_swarm(self, target, source_infos, required):
        source_infos = sorted(source_infos, key=lambda d: (-d["avail"], d["eta"]))
        remaining = int(required)
        plans = []
        for i, info in enumerate(source_infos):
            src = info["src"]
            avail = self.current_available(src)
            if avail < MIN_LAUNCH:
                continue
            left_avail = sum(self.current_available(g["src"]) for g in source_infos[i+1:])
            min_now = max(0, remaining - left_avail)
            fair = int(math.ceil(remaining / max(1, len(source_infos) - i)))
            send = min(avail, max(MIN_LAUNCH, min_now, fair))
            sol = self.solve_launch(src, target, send, require_clear=True)
            if sol is None:
                continue
            sol.kind = "swarm"
            plans.append(sol)
            remaining -= send
            if remaining <= 0:
                break
        if remaining > 0:
            return None
        return plans

    def group_captures(self, target, plans):
        if not plans:
            return False
        extra = [Arrival(target.id, self.w.player, int(p.ships), int(p.eta)) for p in plans]
        capture_eta = max(p.eta for p in plans)

        if self.w.phase() == "endgame":
            horizon = min(self.w.turns_remaining(), capture_eta + 3)
        else:
            hold_window = 10 if target.owner == NEUTRAL_OWNER else 14
            known_followup = [
                a.eta for a in self.w.arrivals_by_planet.get(target.id, [])
                if capture_eta < a.eta <= capture_eta + hold_window + 8
            ]
            horizon = capture_eta + hold_window
            if known_followup:
                horizon = max(horizon, max(known_followup) + 1)

        owner, ships, _timeline = simulate_planet(
            self.w,
            target,
            horizon,
            extra,
            self.planned_arrivals,
            planned_departures=self.committed,
        )
        if owner != self.w.player:
            return False

        # V7: if the target looks captured under known arrivals, also check whether
        # nearby enemy planets can cheaply retake it after capture.
        if self.w.phase() != "endgame":
            owner_at_cap, ships_at_cap, _ = simulate_planet(
                self.w,
                target,
                capture_eta,
                extra,
                self.planned_arrivals,
                planned_departures=self.committed,
            )
            surplus_at_capture = ships_at_cap if owner_at_cap == self.w.player else 0.0
            retake = post_capture_retake_risk(self.w, target, capture_eta, surplus_at_capture, self.planned_arrivals)
            if retake["risk"] >= RETAKE_RISK_HARD_REJECT:
                # Neutral expansion should be held to a stricter standard. Enemy captures
                # may still be worthwhile as disruption if we have a decent surplus.
                if target.owner == NEUTRAL_OWNER:
                    return False
                if surplus_at_capture < retake["extra_margin"] + max(1, target.production):
                    return False

        if self.w.phase() != "endgame" and target.owner != NEUTRAL_OWNER:
            return ships >= max(1, int(0.8 * target.production))
        return ships >= 0

    def plan_reinforcement_to(self, target, deadline, need):
        candidates = []
        for src in self.w.my_planets:
            if src.id == target.id:
                continue
            avail = self.current_available(src)
            if avail < MIN_LAUNCH:
                continue
            probe = min(avail, max(MIN_LAUNCH, min(need, 12)))
            sol = self.solve_launch(src, target, probe, max_turns=min(ATTACK_HORIZON, int(deadline + 5)), require_clear=True)
            if sol is None or sol.eta > deadline:
                continue
            d = dist_xy(src.x, src.y, target.x, target.y)
            source_cost = 0.020 * self.w.importance.get(src.id, 1.0)
            score = sol.eta + 0.016 * d + source_cost - 0.012 * avail
            candidates.append((score, sol.eta, src))
        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:MAX_DEFENSE_SOURCES]
        remaining = int(need)
        plans = []
        for i, (_score, _eta, src) in enumerate(candidates):
            if remaining <= 0:
                break
            avail = self.current_available(src)
            if avail < MIN_LAUNCH:
                continue
            left = max(1, len(candidates) - i)
            send = min(avail, max(MIN_LAUNCH, int(math.ceil(1.10 * remaining / left))))
            sol = self.solve_launch(src, target, send, max_turns=min(ATTACK_HORIZON, int(deadline + 5)), require_clear=True)
            if sol is None or sol.eta > deadline:
                send = min(avail, max(send, int(math.ceil(0.72 * remaining))))
                sol = self.solve_launch(src, target, send, max_turns=min(ATTACK_HORIZON, int(deadline + 5)), require_clear=True)
            if sol is None or sol.eta > deadline:
                continue
            sol.kind = "reinforce"
            plans.append(sol)
            remaining -= send
        if remaining > 0:
            return None
        return plans

    # --------------------------------------------------------
    # Mission execution
    # --------------------------------------------------------

    def execute_missions(self):
        for m in self.missions:
            if len(self.moves) >= MAX_MOVES:
                break
            if m.target_id in self.claimed_targets and m.kind not in ("emergency_defense", "reinforce"):
                continue
            if len(self.moves) + len(m.plans) > MAX_MOVES:
                continue
            if not self.mission_still_valid(m):
                continue

            resolved = self.resolve_mission_plans(m)
            if not resolved:
                continue
            original_plans = m.plans
            m.plans = resolved
            still_valid = self.mission_still_valid(m)
            if not still_valid:
                m.plans = original_plans
                continue

            launched_all = True
            for plan in resolved:
                if not self.commit_resolved_launch(plan):
                    launched_all = False
                    break
            if not launched_all:
                # resolve_mission_plans pre-validates availability, so this should
                # rarely happen. Avoid marking the target as claimed unless the
                # complete mission was committed.
                continue
            if m.kind not in ("emergency_defense", "reinforce"):
                self.claimed_targets.add(m.target_id)

    def mission_still_valid(self, mission):
        target = self.w.by_id.get(mission.target_id)
        if target is None:
            return False

        for p in mission.plans:
            src = self.w.by_id.get(p.source_id)
            if src is None or self.current_available(src) < p.ships:
                return False

        if mission.kind in ("emergency_defense", "reinforce"):
            owner, ships, _ = simulate_planet(
                self.w,
                target,
                mission.deadline,
                planned_arrivals=self.planned_arrivals,
                planned_departures=self.committed,
            )
            if owner == self.w.player and ships >= defense_margin(self.w, target):
                return False
            extras = [Arrival(target.id, self.w.player, int(p.ships), int(p.eta)) for p in mission.plans]
            owner2, ships2, _ = simulate_planet(
                self.w,
                target,
                mission.deadline,
                extras,
                self.planned_arrivals,
                planned_departures=self.committed,
            )
            return owner2 == self.w.player and ships2 >= max(0, defense_margin(self.w, target) - 1)

        if mission.kind == "recapture":
            loss_eta = first_loss_eta(self.w, target, DEFENSE_HORIZON + 45, self.planned_arrivals, self.committed)
            if loss_eta is None or loss_eta > mission.deadline + 2:
                return False

        return self.group_captures(target, mission.plans)

    # --------------------------------------------------------
    # Follow-up / salvage / logistics
    # --------------------------------------------------------

    def followup_attacks(self):
        if len(self.moves) >= MAX_MOVES:
            return
        sources = sorted(self.w.my_planets, key=lambda p: self.current_available(p), reverse=True)
        for src in sources:
            if len(self.moves) >= MAX_MOVES:
                break
            avail = self.current_available(src)
            if avail < MIN_LAUNCH:
                continue
            best_plan = None
            best_score = -1e18
            # Local follow-up captures with leftover budget.
            follow_targets = sorted(
                [t for t in self.w.targets if t.id not in self.claimed_targets and self.target_allowed(t)],
                key=lambda p: self.w.importance.get(p.id, 1.0),
                reverse=True,
            )[:16]
            for target in follow_targets:
                if self.time_exceeded(0.82):
                    break
                if target.id in self.claimed_targets or not self.target_allowed(target):
                    continue
                probe = min(avail, max(MIN_LAUNCH, min(10, avail)))
                sol0 = self.solve_launch(src, target, probe, require_clear=True)
                if sol0 is None or sol0.eta > ATTACK_HORIZON:
                    continue
                required = target_required_ships(self.w, target, sol0.eta, self.planned_arrivals)
                if required < MIN_LAUNCH or required > avail:
                    continue
                sol = self.solve_launch(src, target, required, require_clear=True)
                if sol is None:
                    continue
                d = dist_xy(src.x, src.y, target.x, target.y)
                score = target_roi_score(self.w, target, required, sol.eta, d, self.planned_arrivals)
                # Follow-up pass threshold should be lower but not suicidal.
                if target.owner == NEUTRAL_OWNER:
                    threshold = 0.70 if self.w.phase() != "opening" else 1.10
                else:
                    threshold = 1.15
                if score > threshold and score > best_score:
                    sol.kind = "followup"
                    sol.score = score
                    best_score = score
                    best_plan = sol
            if best_plan and self.group_captures(self.w.by_id[best_plan.target_id], [best_plan]):
                if self.add_launch(best_plan):
                    self.claimed_targets.add(best_plan.target_id)

    def salvage_doomed_planets(self):
        if len(self.moves) >= MAX_MOVES:
            return
        doomed = []
        for p in self.w.my_planets:
            loss_eta = first_loss_eta(self.w, p, min(70, DEFENSE_HORIZON), self.planned_arrivals)
            if loss_eta is None:
                continue
            # If still doomed even with all currently available ships kept at home, salvage.
            margin = defense_margin(self.w, p)
            owner, ships, _ = simulate_planet(self.w, p, loss_eta, planned_arrivals=self.planned_arrivals, planned_departures=self.committed)
            if owner != self.w.player or ships < margin:
                doomed.append((loss_eta, p))
        doomed.sort(key=lambda x: x[0])

        for loss_eta, src in doomed:
            if len(self.moves) >= MAX_MOVES:
                break
            avail = max(0, int(src.ships) - self.committed[src.id] - 1)
            if avail < SALVAGE_MIN_SEND:
                continue
            best = self.best_salvage_plan(src, avail, loss_eta)
            if best is not None:
                self.add_launch(best)

    def best_salvage_plan(self, src, avail, deadline):
        best = None
        best_score = -1e18
        # First prefer a real capture.
        for target in self.w.targets:
            if target.id == src.id or target.id in self.claimed_targets or not self.target_allowed(target):
                continue
            probe = min(avail, max(MIN_LAUNCH, min(10, avail)))
            sol0 = self.solve_launch(src, target, probe, max_turns=min(ATTACK_HORIZON, max(10, int(deadline + 35))), require_clear=True)
            if sol0 is None:
                continue
            required = target_required_ships(self.w, target, sol0.eta, self.planned_arrivals)
            if required < MIN_LAUNCH or required > avail:
                continue
            sol = self.solve_launch(src, target, required, max_turns=min(ATTACK_HORIZON, max(10, int(deadline + 35))), require_clear=True)
            if sol is None:
                continue
            d = dist_xy(src.x, src.y, target.x, target.y)
            score = target_roi_score(self.w, target, required, sol.eta, d, self.planned_arrivals) + 2.0
            if score > best_score:
                sol.kind = "salvage_capture"
                best = sol
                best_score = score
        if best is not None and best_score > 0.55:
            return best

        # Otherwise evacuate to a high-value friendly planet likely to survive.
        for recv in self.w.my_planets:
            if recv.id == src.id:
                continue
            loss = first_loss_eta(self.w, recv, 75, self.planned_arrivals)
            if loss is not None and loss < 20:
                continue
            send = min(avail, max(SALVAGE_MIN_SEND, int(avail * 0.80)))
            sol = self.solve_launch(src, recv, send, max_turns=80, require_clear=True)
            if sol is None:
                continue
            score = self.w.importance.get(recv.id, 1.0) / max(8.0, sol.eta) - 0.03 * self.w.importance.get(src.id, 1.0)
            if score > best_score:
                sol.kind = "salvage_retreat"
                best = sol
                best_score = score
        return best if best_score > 0.4 else None

    def frontier_distance(self, planet, frontier=None):
        frontier = frontier or (self.w.enemy_planets if self.w.enemy_planets else self.w.neutral_planets)
        if not frontier:
            return 999.0
        best = 999.0
        for t in frontier:
            d = dist_xy(planet.x, planet.y, t.x, t.y)
            if d < best:
                best = d
        return best

    def incoming_planned_to(self, planet_id, horizon):
        total = 0
        for a in self.planned_arrivals.get(planet_id, []):
            if a.eta <= horizon:
                total += a.ships
        return total

    def direct_opportunity_score(self, donor):
        cache_key = (donor.id, self.current_available(donor))
        cached = self.direct_score_cache.get(cache_key)
        if cached is not None:
            return cached
        avail = self.current_available(donor)
        if avail < MIN_LAUNCH:
            return 0.0
        best = 0.0
        targets = sorted(
            [t for t in self.w.targets if t.id not in self.claimed_targets and self.target_allowed(t)],
            key=lambda p: self.target_priority(p),
            reverse=True,
        )[:7]
        for target in targets:
            probe = min(avail, max(MIN_LAUNCH, min(10, avail)))
            sol0 = self.solve_launch(donor, target, probe, max_turns=70, require_clear=True)
            if sol0 is None or sol0.eta > 70:
                continue
            required = target_required_ships(self.w, target, sol0.eta, self.planned_arrivals)
            if required < MIN_LAUNCH or required > avail:
                continue
            d = dist_xy(donor.x, donor.y, target.x, target.y)
            score = target_roi_score(self.w, target, required, sol0.eta, d, self.planned_arrivals)
            if score > best:
                best = score
        self.direct_score_cache[cache_key] = best
        return best

    def attack_demand_for_frontline(self, recv):
        owner, projected, _ = simulate_planet(
            self.w,
            recv,
            42,
            planned_arrivals=self.planned_arrivals,
            planned_departures=self.committed,
        )
        if owner != self.w.player:
            return None

        current_budget = max(self.current_available(recv), int(max(0.0, projected - base_reserve(self.w, recv))))
        incoming = self.incoming_planned_to(recv.id, 35)
        current_budget += incoming

        target_pool = sorted(
            [t for t in self.w.enemy_planets if t.id not in self.claimed_targets and self.target_allowed(t)],
            key=lambda p: self.target_priority(p),
            reverse=True,
        )[:9]
        if len(target_pool) < 4:
            target_pool += sorted(
                [t for t in self.w.neutral_planets if t.id not in self.claimed_targets and self.target_allowed(t)],
                key=lambda p: self.target_priority(p),
                reverse=True,
            )[:6]

        best = None
        for target in target_pool:
            if target.id == recv.id:
                continue
            probe = max(MIN_LAUNCH, min(36, int(max(12, current_budget + 12))))
            sol0 = self.solve_launch(recv, target, probe, max_turns=70, require_clear=True)
            if sol0 is None or sol0.eta > 42:
                continue
            required = target_required_ships(self.w, target, sol0.eta, self.planned_arrivals)
            if required < MIN_LAUNCH:
                continue
            demand = int(math.ceil(required - current_budget))
            if demand < CHAIN_RECEIVER_DEMAND_MIN:
                continue
            d = dist_xy(recv.x, recv.y, target.x, target.y)
            score = self.target_priority(target) / max(8.0, sol0.eta)
            score += max(0.0, 44.0 - d) / 42.0
            if target.owner != NEUTRAL_OWNER:
                score *= 1.22
            if best is None or score > best["score"]:
                best = {
                    "receiver": recv,
                    "target": target,
                    "demand": demand,
                    "required": required,
                    "eta": sol0.eta,
                    "score": score,
                }
        return best

    def salvage_expiring_comets(self):
        """Evacuate or spend ships from owned comets before they leave the board."""
        if len(self.moves) >= MAX_MOVES:
            return
        owned_comets = [p for p in self.w.my_planets if p.id in self.w.comet_ids]
        if not owned_comets:
            return
        owned_comets.sort(key=lambda p: self.w.comet_turns_left(p))

        for src in owned_comets:
            if len(self.moves) >= MAX_MOVES or self.time_exceeded(0.70):
                break
            left = self.w.comet_turns_left(src)
            if left <= 0 or left > COMET_EVAC_WINDOW:
                continue
            avail = max(0, int(src.ships) - self.committed[src.id] - COMET_EVAC_KEEP)
            if avail < COMET_EVAC_MIN_SEND:
                continue
            best = self.best_comet_evacuation_plan(src, avail, left)
            if best is not None:
                self.add_launch(best)

    def best_comet_evacuation_plan(self, src, avail, turns_left):
        best = None
        best_score = -1e18
        max_turns = min(ATTACK_HORIZON, max(8, int(turns_left + 38)))

        # First try to convert expiring comet ships into a real capture.
        targets = sorted(
            [t for t in self.w.targets if t.id != src.id and t.id not in self.claimed_targets and self.target_allowed(t)],
            key=lambda p: self.target_priority(p),
            reverse=True,
        )[:14]
        for target in targets:
            if self.time_exceeded(0.72):
                break
            probe = min(avail, max(MIN_LAUNCH, min(14, avail)))
            sol0 = self.solve_launch(src, target, probe, max_turns=max_turns, require_clear=True)
            if sol0 is None:
                continue
            required = target_required_ships(self.w, target, sol0.eta, self.planned_arrivals)
            if required < MIN_LAUNCH or required > avail:
                continue
            sol = self.solve_launch(src, target, required, max_turns=max_turns, require_clear=True)
            if sol is None:
                continue
            if not self.group_captures(target, [sol]):
                continue
            d = dist_xy(src.x, src.y, target.x, target.y)
            score = target_roi_score(self.w, target, required, sol.eta, d, self.planned_arrivals)
            score *= COMET_EVAC_ATTACK_BONUS
            score += 4.0 + max(0.0, COMET_EVAC_WINDOW - turns_left) * 0.30
            if target.owner != NEUTRAL_OWNER:
                score += 2.0
            if score > best_score:
                sol.kind = "comet_evac_attack"
                sol.score = score
                best = sol
                best_score = score

        if best is not None and best_score > 0.35:
            return best

        # Otherwise transfer to the best friendly receiver likely to survive and attack.
        send = max(COMET_EVAC_MIN_SEND, min(avail, int(max(COMET_EVAC_MIN_SEND, avail * 0.88))))
        for recv in self.w.my_planets:
            if recv.id == src.id:
                continue
            loss = first_loss_eta(self.w, recv, 55, self.planned_arrivals, self.committed)
            if loss is not None and loss < max(10, turns_left):
                continue
            sol = self.solve_launch(src, recv, send, max_turns=max_turns, require_clear=True)
            if sol is None:
                continue
            ratio = local_strength_ratio(self.w, recv, min(55, max(8, sol.eta)))
            demand = self.attack_demand_for_frontline(recv)
            demand_score = 0.0 if demand is None else min(5.0, demand["score"])
            score = (
                self.w.importance.get(recv.id, 1.0) / max(8.0, sol.eta)
                + 1.4 * ratio
                + 0.55 * demand_score
                + max(0.0, COMET_EVAC_WINDOW - turns_left) * 0.10
            )
            if score > best_score:
                sol.kind = "comet_evac_transfer"
                sol.score = score
                best = sol
                best_score = score

        return best if best_score > 0.25 else None

    def planet_safety_score(self, planet):
        """How safe is this planet as a donor? Higher means safer to export surplus."""
        ratio = local_strength_ratio(self.w, planet, 28)
        nearest = nearest_owner_danger_mult(self.w, planet, 28, NEAREST_DANGER_K)
        loss = first_loss_eta(self.w, planet, 60, self.planned_arrivals, self.committed)
        incoming_enemy = sum(
            a.ships for a in self.w.arrivals_by_planet.get(planet.id, [])
            if a.owner != self.w.player and a.eta <= 50
        )
        score = ratio
        score += clamp((nearest - 1.0) * 0.55, -0.18, 0.12)
        if loss is not None:
            score -= 0.36 if loss <= 35 else 0.18
        if incoming_enemy > 0:
            score -= min(0.22, incoming_enemy / max(30.0, planet.ships + incoming_enemy))
        if planet.id in self.w.comet_ids:
            score -= 0.20
        return clamp(score, 0.0, 1.0)

    def safe_donor_surplus(self, donor):
        avail = self.current_available(donor)
        if avail < CHAIN_MIN_SEND:
            return 0
        if donor.id in self.w.comet_ids:
            # Expiring comet handling has its own dedicated pass.
            return 0
        safety = self.planet_safety_score(donor)
        if safety < CHAIN_SAFE_RATIO_MIN:
            return 0
        loss = first_loss_eta(self.w, donor, 65, self.planned_arrivals, self.committed)
        if loss is not None and loss < 55:
            return 0
        frac = CHAIN_DONOR_MAX_AVAIL_FRAC
        if safety > 0.72:
            frac += 0.10
        if self.w.modes["is_behind"]:
            frac = min(frac, 0.50)
        return int(max(0, min(avail, math.floor(avail * frac))))

    def strategic_shipflow_logistics(self):
        """Chain safe rear surplus into planets that can launch important future attacks."""
        if len(self.moves) >= MAX_MOVES - 1:
            return
        if self.w.phase() == "endgame" or self.w.step < 34 or len(self.w.my_planets) < 3:
            return
        if not self.w.targets:
            return

        demands = []
        for recv in self.w.my_planets:
            if self.time_exceeded(0.70):
                break
            if recv.id in self.w.comet_ids:
                continue
            if first_loss_eta(self.w, recv, 55, self.planned_arrivals, self.committed) is not None:
                continue
            d = self.attack_demand_for_frontline(recv)
            if d is None:
                continue
            # Boost demand for receivers in friendly-controlled regions that are close
            # to meaningful targets; avoid pouring ships into enemy traps.
            ratio = local_strength_ratio(self.w, recv, min(55, max(10, d["eta"])))
            if ratio < 0.40:
                continue
            d["score"] *= 0.78 + 0.50 * ratio
            demands.append(d)

        if not demands:
            return
        demands.sort(key=lambda x: x["score"], reverse=True)
        demands = demands[:4]

        donors = sorted(self.w.my_planets, key=lambda p: (self.planet_safety_score(p), self.frontier_distance(p)), reverse=True)
        for demand in demands:
            if len(self.moves) >= MAX_MOVES or self.time_exceeded(0.76):
                break
            recv = demand["receiver"]
            target = demand["target"]
            remaining_demand = int(demand["demand"] - self.incoming_planned_to(recv.id, 45))
            if remaining_demand < CHAIN_RECEIVER_DEMAND_MIN:
                continue

            recv_front_dist = self.frontier_distance(recv, [target])
            best = None
            best_score = -1e18

            for donor in donors:
                if donor.id == recv.id:
                    continue
                donor_surplus = self.safe_donor_surplus(donor)
                if donor_surplus < CHAIN_MIN_SEND:
                    continue

                donor_front_dist = self.frontier_distance(donor, [target])
                forward_gain = donor_front_dist - recv_front_dist
                if forward_gain < CHAIN_MIN_FORWARD_GAIN:
                    continue

                direct_score = self.direct_opportunity_score(donor)
                if direct_score > max(1.55, demand["score"] * 0.78):
                    continue

                send = min(donor_surplus, max(CHAIN_MIN_SEND, min(remaining_demand, int(donor_surplus))))
                if send < CHAIN_MIN_SEND:
                    continue
                sol = self.solve_launch(donor, recv, send, max_turns=CHAIN_MAX_ETA, require_clear=True)
                if sol is None or sol.eta > CHAIN_MAX_ETA:
                    continue

                receiver_ratio = local_strength_ratio(self.w, recv, min(55, max(8, sol.eta)))
                score = demand["score"]
                score += forward_gain / max(8.0, sol.eta)
                score += 0.75 * receiver_ratio
                score -= 0.20 * direct_score
                score -= 0.010 * send

                if score > best_score:
                    sol.kind = "strategic_shipflow"
                    sol.score = score
                    best = sol
                    best_score = score

            if best is not None and best_score > 1.05:
                self.add_launch(best)

    def frontline_staging_logistics(self):
        """Move rear surplus into planets that have a concrete future attack demand."""
        if len(self.moves) >= MAX_MOVES - 1:
            return
        if self.w.phase() == "endgame" or self.w.step < 38 or len(self.w.my_planets) < 3:
            return
        if not self.w.targets:
            return

        demands = []
        for recv in self.w.my_planets:
            if self.time_exceeded(0.70):
                break
            if first_loss_eta(self.w, recv, 55, self.planned_arrivals, self.committed) is not None:
                continue
            d = self.attack_demand_for_frontline(recv)
            if d is not None:
                demands.append(d)
        if not demands:
            return
        demands.sort(key=lambda x: x["score"], reverse=True)
        demands = demands[:4]

        donors = sorted(self.w.my_planets, key=lambda p: self.frontier_distance(p), reverse=True)
        for demand in demands:
            if len(self.moves) >= MAX_MOVES or self.time_exceeded(0.76):
                break
            recv = demand["receiver"]
            target = demand["target"]
            remaining_demand = int(demand["demand"] - self.incoming_planned_to(recv.id, 45))
            if remaining_demand < FRONTLINE_STAGING_MIN_SEND:
                continue
            recv_front_dist = self.frontier_distance(recv, [target])
            best = None
            best_score = -1e18
            for donor in donors:
                if donor.id == recv.id:
                    continue
                avail = self.current_available(donor)
                if avail < FRONTLINE_STAGING_MIN_SEND:
                    continue
                # Only move ships forward; don't pull from an equally good attack platform.
                donor_front_dist = self.frontier_distance(donor, [target])
                if donor_front_dist <= recv_front_dist + 8.0:
                    continue
                direct_score = self.direct_opportunity_score(donor)
                if direct_score > max(1.45, demand["score"] * 0.72):
                    continue
                send = min(avail, max(FRONTLINE_STAGING_MIN_SEND, min(remaining_demand, int(avail * 0.55))))
                sol = self.solve_launch(donor, recv, send, max_turns=FRONTLINE_STAGING_MAX_ETA, require_clear=True)
                if sol is None or sol.eta > FRONTLINE_STAGING_MAX_ETA:
                    continue
                forward_gain = max(0.0, donor_front_dist - recv_front_dist)
                score = demand["score"] + forward_gain / max(8.0, sol.eta) - 0.18 * direct_score
                score -= 0.010 * send
                if score > best_score:
                    sol.kind = "frontline_staging"
                    sol.score = score
                    best = sol
                    best_score = score
            if best is not None and best_score > 1.10:
                if self.add_launch(best):
                    # Treat the receiver as deliberately stocked, but do not claim the attack target yet.
                    pass

    def logistics_funnel(self):
        if len(self.moves) >= MAX_MOVES - 1:
            return
        if self.w.phase() in ("opening", "endgame") or len(self.w.my_planets) < 4:
            return

        frontier = self.w.enemy_planets if self.w.enemy_planets else self.w.neutral_planets
        if not frontier:
            return

        def frontier_distance(p):
            best = 999.0
            for t in frontier:
                d = dist_xy(p.x, p.y, t.x, t.y)
                if d < best:
                    best = d
            return best

        my_sorted = sorted(self.w.my_planets, key=frontier_distance)
        receivers = my_sorted[:max(1, min(3, len(my_sorted)//3))]
        donors = list(reversed(my_sorted))

        for donor in donors:
            if len(self.moves) >= MAX_MOVES:
                break
            avail = self.current_available(donor)
            if avail < LOGISTICS_MIN_SEND:
                continue
            if self.planet_safety_score(donor) < max(0.50, CHAIN_SAFE_RATIO_MIN - 0.05):
                continue
            dfd = frontier_distance(donor)
            best = None
            best_score = -1e18
            for recv in receivers:
                if recv.id == donor.id:
                    continue
                if frontier_distance(recv) >= dfd - 10:
                    continue
                # Do not funnel into already very fat planets.
                owner, proj, _ = simulate_planet(self.w, recv, 45, planned_arrivals=self.planned_arrivals, planned_departures=self.committed)
                if owner != self.w.player:
                    continue
                desired = desired_frontline_garrison(self.w, recv)
                if proj > desired + 12:
                    continue
                send = min(avail, max(LOGISTICS_MIN_SEND, int(avail * 0.45)))
                sol = self.solve_launch(donor, recv, send, max_turns=70, require_clear=True)
                if sol is None or sol.eta > 50:
                    continue
                score = (dfd - frontier_distance(recv)) / max(6.0, sol.eta) + self.w.importance.get(recv.id, 1.0) / 120.0
                if score > best_score:
                    sol.kind = "logistics"
                    best = sol
                    best_score = score
            if best is not None and best_score > 1.05:
                self.add_launch(best)


def desired_frontline_garrison(world, planet):
    enemy_pressure = 0.0
    for e in world.enemy_planets:
        d = dist_xy(planet.x, planet.y, e.x, e.y)
        if d < 38:
            enemy_pressure += e.production / max(8.0, d)
    base = 4 + 2.5 * planet.production
    if world.modes["is_dominating"]:
        base += 2
    if world.modes["is_behind"]:
        base -= 1
    return int(base + min(12.0, world.importance.get(planet.id, 1.0) / 13.0) + min(14.0, 25.0 * enemy_pressure))


# ============================================================
# Agent entry point
# ============================================================

def agent(obs):
    start = time.perf_counter()
    try:
        world = World(obs)
        planner = Planner(world)
        moves = planner.plan()
        _RUNTIME.last_error = None
    except Exception as exc:
        # Never crash a submission turn.
        _RUNTIME.last_error = repr(exc)
        moves = []

    elapsed = time.perf_counter() - start
    _RUNTIME.record(elapsed)

    if DEBUG_TIMING_PRINTS and _RUNTIME.turns % DEBUG_PRINT_EVERY == 0:
        print(f"Game Turn: {_RUNTIME.game_turn}")
        print("agent_stats", get_agent_stats())

    return moves
