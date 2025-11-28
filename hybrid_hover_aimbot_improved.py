# hybrid_hover_aimbot_improved.py
from typing import Dict, Tuple, Optional
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import KesslerController


class HybridParams:
    """
    Tunable parameters. These can be learned by GA and injected via __init__(params=...).
    All values here are safe defaults (tamer thrust, speed cap, bigger safety bubble).
    """
    def __init__(
        self,
        hover_distance: float = 110.0,      # slightly farther hover to reduce ramming
        max_thrust_abs: float = 200.0,      # global clamp for safety
        max_speed: float = 150.0,           # slow ship -> fewer crashes
        crowd_radius: float = 120.0,
        safe_bubble: float = 80.0,          # never thrust forward inside this
        w_close: float = 2.0,               # target score weight on closing speed
        w_dist: float = 1.0,                # target score weight on distance

        # Thrust MF shapes (left, peak, right)
        thrust_back_high_left: float = -250.0,
        thrust_back_high_peak: float = -250.0,
        thrust_back_high_right: float = -180.0,

        thrust_back_med_left: float = -190.0,
        thrust_back_med_peak: float = -120.0,
        thrust_back_med_right: float = -30.0,

        thrust_zero_left: float = -40.0,
        thrust_zero_peak: float = 0.0,
        thrust_zero_right: float = 40.0,

        thrust_fwd_low_left: float = 20.0,
        thrust_fwd_low_peak: float = 50.0,
        thrust_fwd_low_right: float = 80.0,

        thrust_fwd_med_left: float = 70.0,
        thrust_fwd_med_peak: float = 110.0,
        thrust_fwd_med_right: float = 160.0,

        thrust_fwd_high_left: float = 130.0,
        thrust_fwd_high_peak: float = 180.0,
        thrust_fwd_high_right: float = 200.0,

        panic_count: int = 6,
        reverse_speed_thresh: float = -100.0,
        bullet_speed: float = 800.0,
        aim_theta_cap: float = math.pi / 6,  # +/- 30 deg for fuzzy input
        aim_dt_lead: float = 1.0 / 30.0      # small lead for movement between frames
    ):
        self.hover_distance = hover_distance
        self.max_thrust_abs = max_thrust_abs
        self.max_speed = max_speed
        self.crowd_radius = crowd_radius
        self.safe_bubble = safe_bubble
        self.w_close = w_close
        self.w_dist = w_dist

        # --- ensure thrust MF assignments exist ---
        self.thrust_back_high_left  = thrust_back_high_left
        self.thrust_back_high_peak  = thrust_back_high_peak
        self.thrust_back_high_right = thrust_back_high_right

        self.thrust_back_med_left   = thrust_back_med_left
        self.thrust_back_med_peak   = thrust_back_med_peak
        self.thrust_back_med_right  = thrust_back_med_right

        self.thrust_zero_left       = thrust_zero_left
        self.thrust_zero_peak       = thrust_zero_peak
        self.thrust_zero_right      = thrust_zero_right

        self.thrust_fwd_low_left    = thrust_fwd_low_left
        self.thrust_fwd_low_peak    = thrust_fwd_low_peak
        self.thrust_fwd_low_right   = thrust_fwd_low_right

        self.thrust_fwd_med_left    = thrust_fwd_med_left
        self.thrust_fwd_med_peak    = thrust_fwd_med_peak
        self.thrust_fwd_med_right   = thrust_fwd_med_right

        self.thrust_fwd_high_left   = thrust_fwd_high_left
        self.thrust_fwd_high_peak   = thrust_fwd_high_peak
        self.thrust_fwd_high_right  = thrust_fwd_high_right

        self.panic_count = panic_count
        self.reverse_speed_thresh = reverse_speed_thresh
        self.bullet_speed = bullet_speed
        self.aim_theta_cap = aim_theta_cap
        self.aim_dt_lead = aim_dt_lead


class HybridHoverAimbot(KesslerController):
    """
    Improved Hybrid controller:
    - Picks the *most dangerous* asteroid (closing on you + reasonably close).
    - Fuzzy aimbot for turn + fire (time-to-hit + angle error).
    - Fuzzy hover controller for thrust (stay around hover_distance).
    - Global thrust clamp + speed cap + close-bubble no-forward rule.
    - Reuses fuzzy simulations each frame (no re-allocation jitter).
    - Intentional mine dropping only under pressure & moving in reverse.
    """

    def __init__(self, params: Optional[HybridParams] = None):
        self.eval_frames = 0
        self.P = params or HybridParams()

        # quick sanity to catch stale imports early
        for attr in ("thrust_back_high_left", "thrust_fwd_high_right"):
            assert hasattr(self.P, attr), f"HybridParams missing {attr}"

        # ==============================
        #   MOVEMENT FUZZY SYSTEM
        # ==============================
        hd = self.P.hover_distance

        # error = actual_distance - hover_distance
        dist_err = ctrl.Antecedent(np.arange(-2 * hd, 2 * hd + 1, 1), 'dist_err')
        crowd = ctrl.Antecedent(np.arange(0, 20, 1), 'crowd')

        # Thrust output (negative = backwards, positive = forwards)
        tmin = min(self.P.thrust_back_high_left, -self.P.max_thrust_abs)
        tmax = max(self.P.thrust_fwd_high_right, self.P.max_thrust_abs)
        thrust_fs = ctrl.Consequent(np.arange(tmin, tmax + 1, 10), 'thrust')

        # distance error membership functions
        dist_err['too_close_fast'] = fuzz.trimf(
            dist_err.universe, [-2 * hd, -2 * hd, -0.9 * hd]
        )
        dist_err['too_close_slow'] = fuzz.trimf(
            dist_err.universe, [-1.0 * hd, -0.5 * hd, 0]
        )
        dist_err['on_target'] = fuzz.trimf(
            dist_err.universe, [-0.2 * hd, 0, 0.2 * hd]
        )
        dist_err['too_far_slow'] = fuzz.trimf(
            dist_err.universe, [0, 0.5 * hd, 1.5 * hd]
        )
        dist_err['too_far_fast'] = fuzz.trimf(
            dist_err.universe, [hd, 2 * hd, 2 * hd]
        )

        # crowding membership
        crowd['few'] = fuzz.trimf(crowd.universe, [0, 0, 7])
        crowd['many'] = fuzz.smf(crowd.universe, 5, 15)

        # thrust membership (tamer by default)
        P = self.P
        thrust_fs['back_high'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_back_high_left, P.thrust_back_high_peak, P.thrust_back_high_right]
        )
        thrust_fs['back_med'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_back_med_left, P.thrust_back_med_peak, P.thrust_back_med_right]
        )
        thrust_fs['zero'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_zero_left, P.thrust_zero_peak, P.thrust_zero_right]
        )
        thrust_fs['fwd_low'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_fwd_low_left, P.thrust_fwd_low_peak, P.thrust_fwd_low_right]
        )
        thrust_fs['fwd_med'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_fwd_med_left, P.thrust_fwd_med_peak, P.thrust_fwd_med_right]
        )
        thrust_fs['fwd_high'] = fuzz.trimf(
            thrust_fs.universe, [P.thrust_fwd_high_left, P.thrust_fwd_high_peak, P.thrust_fwd_high_right]
        )

        # rules – hover around hd, move harder if very off
        m1 = ctrl.Rule(
            dist_err['too_close_fast'] | (dist_err['too_close_slow'] & crowd['many']),
            thrust_fs['back_high']
        )
        m2 = ctrl.Rule(
            dist_err['too_close_slow'] & crowd['few'],
            thrust_fs['back_med']
        )
        m3 = ctrl.Rule(dist_err['on_target'], thrust_fs['zero'])
        m4 = ctrl.Rule(dist_err['too_far_slow'], thrust_fs['fwd_low'])
        m5 = ctrl.Rule(dist_err['too_far_fast'], thrust_fs['fwd_med'])

        self.movement_control = ctrl.ControlSystem([m1, m2, m3, m4, m5])

        # ==============================
        #   AIMING / SHOOTING FUZZY SYSTEM
        # ==============================
        bt = ctrl.Antecedent(np.linspace(0.0, 1.0, 501), 'bullet_time')
        td = ctrl.Antecedent(np.linspace(-self.P.aim_theta_cap, self.P.aim_theta_cap, 301), 'theta_delta')

        turn = ctrl.Consequent(np.linspace(-180.0, 180.0, 361), 'ship_turn')
        fire = ctrl.Consequent(np.linspace(-1.0, 1.0, 201), 'ship_fire')

        # bullet_time
        bt['S'] = fuzz.trimf(bt.universe, [0.0, 0.0, 0.15])
        bt['M'] = fuzz.trimf(bt.universe, [0.05, 0.25, 0.45])
        bt['L'] = fuzz.smf(bt.universe, 0.3, 0.7)

        # theta_delta
        cap = self.P.aim_theta_cap
        td['NL'] = fuzz.zmf(td.universe, -cap, -cap/3.0)
        td['NM'] = fuzz.trimf(td.universe, [-cap, -cap/3.0, -cap/6.0])
        td['NS'] = fuzz.trimf(td.universe, [-cap/3.0, -cap/6.0, 0.0])
        td['Z']  = fuzz.trimf(td.universe, [-cap/18.0, 0.0, cap/18.0])
        td['PS'] = fuzz.trimf(td.universe, [0.0, cap/6.0, cap/3.0])
        td['PM'] = fuzz.trimf(td.universe, [cap/6.0, cap/3.0, cap])
        td['PL'] = fuzz.smf(td.universe, cap/3.0, cap)

        # ship_turn
        turn['NL'] = fuzz.trimf(turn.universe, [-180.0, -180.0, -120.0])
        turn['NM'] = fuzz.trimf(turn.universe, [-150.0, -90.0, -45.0])
        turn['NS'] = fuzz.trimf(turn.universe, [-60.0, -30.0, 0.0])
        turn['Z']  = fuzz.trimf(turn.universe, [-10.0, 0.0, 10.0])
        turn['PS'] = fuzz.trimf(turn.universe, [0.0, 30.0, 60.0])
        turn['PM'] = fuzz.trimf(turn.universe, [45.0, 90.0, 150.0])
        turn['PL'] = fuzz.trimf(turn.universe, [120.0, 180.0, 180.0])

        # ship_fire
        fire['N'] = fuzz.trimf(fire.universe, [-1.0, -1.0, 0.0])
        fire['Y'] = fuzz.trimf(fire.universe, [0.0, 1.0, 1.0])

        rules = []
        # LONG time-to-hit – conservative
        rules += [
            ctrl.Rule(bt['L'] & td['NL'], (turn['NL'], fire['N'])),
            ctrl.Rule(bt['L'] & td['NM'], (turn['NM'], fire['N'])),
            ctrl.Rule(bt['L'] & td['NS'], (turn['NS'], fire['N'])),
            ctrl.Rule(bt['L'] & td['Z'],  (turn['Z'],  fire['Y'])),
            ctrl.Rule(bt['L'] & td['PS'], (turn['PS'], fire['N'])),
            ctrl.Rule(bt['L'] & td['PM'], (turn['PM'], fire['N'])),
            ctrl.Rule(bt['L'] & td['PL'], (turn['PL'], fire['N'])),
        ]
        # MEDIUM time-to-hit
        rules += [
            ctrl.Rule(bt['M'] & td['NL'], (turn['NL'], fire['N'])),
            ctrl.Rule(bt['M'] & td['NM'], (turn['NM'], fire['N'])),
            ctrl.Rule(bt['M'] & td['NS'], (turn['NS'], fire['Y'])),
            ctrl.Rule(bt['M'] & td['Z'],  (turn['Z'],  fire['Y'])),
            ctrl.Rule(bt['M'] & td['PS'], (turn['PS'], fire['Y'])),
            ctrl.Rule(bt['M'] & td['PM'], (turn['PM'], fire['N'])),
            ctrl.Rule(bt['M'] & td['PL'], (turn['PL'], fire['N'])),
        ]
        # SHORT time-to-hit – aggressive
        rules += [
            ctrl.Rule(bt['S'] & td['NL'], (turn['NL'], fire['Y'])),
            ctrl.Rule(bt['S'] & td['NM'], (turn['NM'], fire['Y'])),
            ctrl.Rule(bt['S'] & td['NS'], (turn['NS'], fire['Y'])),
            ctrl.Rule(bt['S'] & td['Z'],  (turn['Z'],  fire['Y'])),
            ctrl.Rule(bt['S'] & td['PS'], (turn['PS'], fire['Y'])),
            ctrl.Rule(bt['S'] & td['PM'], (turn['PM'], fire['Y'])),
            ctrl.Rule(bt['S'] & td['PL'], (turn['PL'], fire['Y'])),
        ]

        self.aim_system = ctrl.ControlSystem(rules)

        # Reusable simulations
        self.aim_sim  = ctrl.ControlSystemSimulation(self.aim_system)
        self.move_sim = ctrl.ControlSystemSimulation(self.movement_control)

    # ================================================================
    #  HELPER: pick the most dangerous asteroid
    # ================================================================
    def _select_dangerous_asteroid(self, ship_state: Dict, asteroids):
        ship_x, ship_y = ship_state["position"]
        vx_ship, vy_ship = ship_state["velocity"]

        best_asteroid = None
        best_score = None
        best_dist_sq = None
        best_closing_speed = 0.0

        for a in asteroids:
            ax, ay = a["position"]
            vx, vy = a["velocity"]

            dx = ax - ship_x
            dy = ay - ship_y
            dist_sq = dx * dx + dy * dy
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-6

            # relative velocity of asteroid w.r.t. ship
            rvx = vx - vx_ship
            rvy = vy - vy_ship

            # radial speed along line-of-sight (negative = approaching)
            ux, uy = dx / dist, dy / dist
            radial_speed = rvx * ux + rvy * uy
            closing_speed = max(0.0, -radial_speed)  # >0 if approaching

            score = self.P.w_close * closing_speed + self.P.w_dist * (1.0 / (dist + 10.0))

            if best_asteroid is None or score > best_score:
                best_asteroid = a
                best_score = score
                best_dist_sq = dist_sq
                best_closing_speed = closing_speed

        return best_asteroid, best_dist_sq, best_closing_speed

    # ================================================================
    #  RUNTIME
    # ================================================================
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self.eval_frames += 1

        asteroids = game_state["asteroids"]
        if not asteroids:
            return 0.0, 0.0, False, False

        ship_x, ship_y = ship_state["position"]
        vx_ship, vy_ship = ship_state["velocity"]
        heading_deg = ship_state["heading"]
        heading_rad = heading_deg * math.pi / 180.0

        # ---------- 1. Pick most dangerous asteroid ----------
        target, dist_sq, closing_speed = self._select_dangerous_asteroid(ship_state, asteroids)
        if target is None:
            return 0.0, 0.0, False, False

        dist = math.sqrt(dist_sq)
        ax, ay = target["position"]
        vx, vy = target["velocity"]

        # ---------- 2. Compute intercept time ----------
        b = self.P.bullet_speed

        asteroid_speed = math.hypot(vx, vy)
        if asteroid_speed < 1e-6 or asteroid_speed >= b:
            bullet_t = dist / b if b > 0 else 1.0
        else:
            asteroid_ship_theta = math.atan2(ship_y - ay, ship_x - ax)
            asteroid_direction = math.atan2(vy, vx)
            theta2 = asteroid_ship_theta - asteroid_direction
            cos_theta2 = math.cos(theta2)

            D = dist
            v = asteroid_speed

            targ_det = (-2 * D * v * cos_theta2) ** 2 - (4 * (v * v - b * b) * (D * D))
            if targ_det < 0:
                bullet_t = dist / b if b > 0 else 1.0
            else:
                sqrt_det = math.sqrt(targ_det)
                denom = 2 * (v * v - b * b)

                t1 = ((2 * D * v * cos_theta2) + sqrt_det) / denom
                t2 = ((2 * D * v * cos_theta2) - sqrt_det) / denom

                candidates = [t for t in (t1, t2) if t >= 0]
                bullet_t = min(candidates) if candidates else (dist / b if b > 0 else 1.0)

        if not math.isfinite(bullet_t):
            bullet_t = 1.0
        bullet_t = max(0.0, min(1.0, bullet_t))

        # ---------- 3. Intercept point + theta_delta ----------
        intrcpt_x = ax + vx * (bullet_t + self.P.aim_dt_lead)
        intrcpt_y = ay + vy * (bullet_t + self.P.aim_dt_lead)

        desired_theta = math.atan2(intrcpt_y - ship_y, intrcpt_x - ship_x)
        theta_delta = desired_theta - heading_rad
        theta_delta = (theta_delta + math.pi) % (2 * math.pi) - math.pi
        theta_cap = self.P.aim_theta_cap
        theta_delta = max(-theta_cap, min(theta_cap, theta_delta))

        # ---------- 4. Fuzzy AIM (turn + fire) ----------
        try:
            self.aim_sim.reset()
            self.aim_sim.input['bullet_time'] = bullet_t
            self.aim_sim.input['theta_delta'] = theta_delta
            self.aim_sim.compute()

            turn_rate = float(self.aim_sim.output['ship_turn'])
            fire_val = float(self.aim_sim.output['ship_fire'])
        except Exception:
            # fallback: proportional
            Kp = 300.0
            turn_rate = Kp * theta_delta * 180.0 / math.pi
            fire_val = 1.0 if abs(theta_delta) < (5.0 * math.pi / 180.0) else -1.0

        turn_rate = max(-180.0, min(180.0, turn_rate))
        fire = bool(fire_val >= 0.0)

        # ---------- 5. Fuzzy MOVEMENT (hover) ----------
        dist_error = dist - self.P.hover_distance

        # crowd: # of nearby asteroids
        crowd_count = 0
        for a in asteroids:
            ax2, ay2 = a["position"]
            if math.hypot(ax2 - ship_x, ay2 - ship_y) < self.P.crowd_radius:
                crowd_count += 1

        try:
            self.move_sim.reset()
            self.move_sim.input['dist_err'] = dist_error
            self.move_sim.input['crowd'] = min(crowd_count, 19)
            self.move_sim.compute()
            thrust = float(self.move_sim.output['thrust'])
        except Exception:
            thrust = -min(self.P.max_thrust_abs, 150.0) if dist_error < 0 else min(self.P.max_thrust_abs, 100.0)

        # Close-range safety: never thrust forward inside bubble
        if dist < self.P.safe_bubble:
            thrust = min(thrust, 0.0)
            # If approaching fast, bias harder backwards
            if closing_speed > 0:
                thrust = min(thrust, -0.75 * self.P.max_thrust_abs)

        # Speed damping
        speed = math.hypot(vx_ship, vy_ship)
        if speed > self.P.max_speed:
            scale = max(0.0, 1.0 - (speed - self.P.max_speed) / self.P.max_speed)
            thrust *= scale

        # Global clamp (extra safety)
        thrust = max(-self.P.max_thrust_abs, min(self.P.max_thrust_abs, thrust))

        # ---------- 6. Mine dropping ----------
        # forward component of velocity along heading
        forward_speed = vx_ship * math.cos(heading_rad) + vy_ship * math.sin(heading_rad)
        very_close = any(
            math.hypot(a["position"][0] - ship_x, a["position"][1] - ship_y) < 80.0
            for a in asteroids
        )
        if (crowd_count >= self.P.panic_count
            and forward_speed < self.P.reverse_speed_thresh
            and very_close):
            drop_mine = True
        else:
            drop_mine = False

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "Hybrid Hover Aimbot (Improved, Safe)"
