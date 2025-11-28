# fuzzy_aimbot.py

from kesslergame import KesslerController
from typing import Any, Tuple
import math


class FuzzyAimbotController(KesslerController):
    """
    Simple aimbot controller:
    - Always shooting (fire = True every frame)
    - Picks the most threatening asteroid:
        * prioritize ones moving toward the ship (inbound)
        * among those, prefers faster-closing and closer
    - Rotates in place to face that asteroid
    - No thrust (stays roughly in one spot)
    """

    MAX_TURN_RATE = 150.0  # deg/s
    KP_TURN = 2.0          # proportional gain for smooth turning

    def __init__(self):
        self.eval_frames = 0

    # ---------- Helpers for dict OR object state ----------

    @staticmethod
    def _get_pos(obj) -> Tuple[float, float]:
        """Return (x, y) from ship/asteroid state (dict or object)."""
        if isinstance(obj, dict):
            return obj["position"]
        if hasattr(obj, "position"):
            return obj.position
        if hasattr(obj, "ownstate"):
            os = obj.ownstate
            if isinstance(os, dict) and "position" in os:
                return os["position"]
        raise AttributeError("No position found in object")

    @staticmethod
    def _get_vel(obj) -> Tuple[float, float]:
        """Return (vx, vy) from ship/asteroid state (dict or object)."""
        if isinstance(obj, dict):
            return obj.get("velocity", (0.0, 0.0))
        if hasattr(obj, "velocity"):
            return obj.velocity
        if hasattr(obj, "ownstate"):
            os = obj.ownstate
            if isinstance(os, dict):
                return os.get("velocity", (0.0, 0.0))
        return (0.0, 0.0)

    @staticmethod
    def _get_heading_deg(ship_state) -> float:
        """Return ship heading/angle in degrees."""
        if isinstance(ship_state, dict):
            return ship_state.get("heading", ship_state.get("angle", 0.0))
        if hasattr(ship_state, "heading"):
            return ship_state.heading
        if hasattr(ship_state, "angle"):
            return ship_state.angle
        if hasattr(ship_state, "ownstate"):
            os = ship_state.ownstate
            if isinstance(os, dict):
                return os.get("heading", os.get("angle", 0.0))
        return 0.0

    @staticmethod
    def _get_asteroids(game_state) -> list:
        """Return list of asteroid states from GameState or dict."""
        if isinstance(game_state, dict):
            return game_state.get("asteroids", [])
        if hasattr(game_state, "asteroids"):
            return game_state.asteroids
        return []

    # ---------- Small math helpers ----------

    @staticmethod
    def _wrap_angle_rad(theta: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _sign(x: float) -> float:
        if x > 0:
            return 1.0
        if x < 0:
            return -1.0
        return 0.0

    # ---------- Threat selection: "coming in the way" ----------

    def _pick_most_threatening(self, ship_state: Any, game_state: Any):
        """
        Pick asteroid with highest threat score.

        Threat score:
        - If inbound (moving toward ship): big base bonus
        - Among inbound: prefer higher closing speed & closer distance
        - Non-inbound only chosen if nothing is inbound
        """
        asteroids = self._get_asteroids(game_state)
        if not asteroids:
            return None

        sx, sy = self._get_pos(ship_state)
        svx, svy = self._get_vel(ship_state)

        best = None
        best_score = -1e9

        for a in asteroids:
            ax, ay = self._get_pos(a)
            avx, avy = self._get_vel(a)

            dx = ax - sx
            dy = ay - sy
            dist = math.hypot(dx, dy) + 1e-6

            # relative velocity: asteroid wrt ship
            rvx = avx - svx
            rvy = avy - svy

            dot = dx * rvx + dy * rvy
            closing_speed = -dot / dist  # >0 => moving toward us
            inbound = closing_speed > 0.0

            if inbound:
                # Very threatening: heading toward us
                score = 1000.0
                # prefer closer & faster-closing
                score += 10.0 * closing_speed / (dist + 1.0)
            else:
                # fallback: just prefer closer
                score = 1.0 / (dist + 1.0)

            if score > best_score:
                best_score = score
                best = a

        return best

    # ---------- Main actions ----------

    def actions(self, ship_state: Any, game_state: Any) -> Tuple[float, float, bool, bool]:
        """
        Returns:
        (thrust [m/s^2], turn_rate [deg/s], fire [bool], drop_mine [bool])
        """
        asteroids = self._get_asteroids(game_state)

        # defaults
        thrust = 0.0
        turn_rate = 0.0
        fire = True          # ALWAYS SHOOTING
        drop_mine = False

        if not asteroids:
            # no asteroids? just spin and shoot anyway
            turn_rate = 40.0
            self.eval_frames += 1
            return thrust, turn_rate, fire, drop_mine

        # pick the most threatening asteroid (coming in the way)
        target = self._pick_most_threatening(ship_state, game_state)
        if target is None:
            turn_rate = 40.0
            self.eval_frames += 1
            return thrust, turn_rate, fire, drop_mine

        sx, sy = self._get_pos(ship_state)
        heading_deg = self._get_heading_deg(ship_state)
        heading_rad = math.radians(heading_deg)

        tx, ty = self._get_pos(target)
        dx = tx - sx
        dy = ty - sy

        # angle we want to face
        desired_angle = math.atan2(dy, dx)

        # angle error between our facing and target direction
        err_rad = self._wrap_angle_rad(desired_angle - heading_rad)
        err_deg = math.degrees(err_rad)

        # turn toward the target
        if abs(err_deg) > 45.0:
            # big error: max turn
            turn_rate = self.MAX_TURN_RATE * self._sign(err_deg)
        else:
            # smaller error: proportional smooth turning
            turn_rate = self.KP_TURN * err_deg
            turn_rate = max(-self.MAX_TURN_RATE,
                            min(self.MAX_TURN_RATE, turn_rate))

        # thrust stays zero so we don't race around = more stable aim
        thrust = 0.0

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "AlwaysShootingThreatBot"
