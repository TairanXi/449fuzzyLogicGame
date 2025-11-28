from typing import Dict, Tuple
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import KesslerController


class HybridHoverAimbot(KesslerController):
    """
    Hybrid controller:
    - Fuzzy aiming and firing for turn_rate + fire
    - Fuzzy movement to hover around a desired distance
    - Extra safety: avoid backing into asteroids, safer mines
    - Incoming-asteroid detection: bias movement toward safer regions
    """

    def __init__(self):
        self.eval_frames = 0

        # ==============================
        #   TUNABLE HYPERPARAMETERS
        # ==============================
        self.hover_distance = 150.0      # desired distance from nearest asteroid
        self.max_speed = 150.0           # speed cap for damping
        self.panic_distance = 60.0       # hard safety override distance
        self.fire_threshold = 0.2        # threshold on fuzzy ship_fire output
        self.mine_panic_count = 6        # "a lot" of nearby asteroids
        self.turn_gain = 1.6             # multiplier on fuzzy ship_turn output

        # Safety-related hyperparams
        self.safe_back_distance = 80.0   # how far behind to check for asteroids
        self.safe_back_side = 50.0       # lateral radius behind ship for danger
        self.mine_forward_thresh = 60.0  # only drop mines if moving forward faster than this

        # Incoming-asteroid / escape behaviour
        self.danger_distance = 250.0         # consider asteroids within this distance for danger
        self.danger_closing_thresh = 20.0    # min closing speed (units/s) to treat as dangerous
        self.danger_weight_thresh = 40.0     # how much total danger before escape kicks in
        self.escape_thrust = 160.0           # thrust magnitude for escape bias

        # ==============================
        #   MOVEMENT FUZZY SYSTEM
        # ==============================
        hd = self.hover_distance

        # ERROR-based distance: how far from the desired hover distance
        # error = actual_distance - hover_distance
        dist_err = ctrl.Antecedent(np.arange(-2 * hd, 2 * hd + 1, 1), 'dist_err')
        crowd = ctrl.Antecedent(np.arange(0, 20, 1), 'crowd')

        # Thrust output (negative = backwards, positive = forwards)
        thrust_fs = ctrl.Consequent(np.arange(-220, 221, 10), 'thrust')

        # ----- membership functions for distance ERROR -----
        # Negative = too close; Positive = too far

        # Very too close: only when error is REALLY negative
        dist_err['too_close_fast'] = fuzz.trimf(
            dist_err.universe,
            [-2 * hd, -2 * hd, -0.9 * hd]
        )

        # Slightly too close: more moderate band
        dist_err['too_close_slow'] = fuzz.trimf(
            dist_err.universe,
            [-1.0 * hd, -0.5 * hd, 0]
        )

        # Around the perfect hover distance (wider dead-zone)
        dist_err['on_target'] = fuzz.trimf(
            dist_err.universe,
            [-0.4 * hd, 0, 0.4 * hd]
        )

        # Slightly too far
        dist_err['too_far_slow'] = fuzz.trimf(
            dist_err.universe,
            [0, 0.5 * hd, 1.5 * hd]
        )

        # Very too far (slightly shifted)
        dist_err['too_far_fast'] = fuzz.trimf(
            dist_err.universe,
            [0.8 * hd, 1.6 * hd, 2.0 * hd]
        )

        # ----- membership functions for crowding -----
        crowd['few'] = fuzz.trimf(crowd.universe, [0, 0, 7])
        crowd['many'] = fuzz.smf(crowd.universe, 5, 15)

        # ----- membership functions for thrust output (NEG + POS) -----
        # Softer thrust overall
        thrust_fs['back_high'] = fuzz.trimf(thrust_fs.universe, [-220, -220, -160])
        thrust_fs['back_med'] = fuzz.trimf(thrust_fs.universe, [-170, -110, -30])
        thrust_fs['zero'] = fuzz.trimf(thrust_fs.universe, [-30, 0, 30])
        thrust_fs['fwd_low'] = fuzz.trimf(thrust_fs.universe, [20, 60, 90])
        thrust_fs['fwd_med'] = fuzz.trimf(thrust_fs.universe, [80, 120, 160])
        thrust_fs['fwd_high'] = fuzz.trimf(thrust_fs.universe, [150, 220, 220])

        # ----- PID-like hover behaviour around hover_distance -----

        # Very too close -> move backwards fast
        m1 = ctrl.Rule(
            dist_err['too_close_fast'] | (dist_err['too_close_slow'] & crowd['many']),
            thrust_fs['back_high']
        )

        # Slightly too close -> move backwards slowly
        m2 = ctrl.Rule(dist_err['too_close_slow'] & crowd['few'], thrust_fs['back_med'])

        # Around target distance -> hover
        m3 = ctrl.Rule(dist_err['on_target'], thrust_fs['zero'])

        # Slightly too far -> move forwards slowly
        m4 = ctrl.Rule(dist_err['too_far_slow'], thrust_fs['fwd_low'])

        # Very too far -> move forwards faster
        m5 = ctrl.Rule(dist_err['too_far_fast'], thrust_fs['fwd_med'])

        # Collect movement rules into a ControlSystem
        self.movement_control = ctrl.ControlSystem([m1, m2, m3, m4, m5])

        # ==============================
        #   AIMING / SHOOTING FUZZY SYSTEM
        # ==============================

        # bullet_time: time (s) until bullet reaches intercept point
        bt = ctrl.Antecedent(np.linspace(0.0, 2.0, 501), 'bullet_time')

        # theta_delta: angle error (rad) between ship heading and desired firing angle
        # Domain: about ±30 degrees
        td = ctrl.Antecedent(np.linspace(-math.pi / 6, math.pi / 6, 301), 'theta_delta')

        # ship_turn: output turn rate (deg/sec)
        turn = ctrl.Consequent(np.linspace(-180.0, 180.0, 361), 'ship_turn')

        # ship_fire: output "fire strength" in [-1, 1] → thresholded to bool
        fire = ctrl.Consequent(np.linspace(-1.0, 1.0, 201), 'ship_fire')

        # ---------- MEMBERSHIP FUNCTIONS ----------

        # bullet_time: Short / Medium / Long
        bt['S'] = fuzz.trimf(bt.universe, [0.0, 0.0, 0.25])
        bt['M'] = fuzz.trimf(bt.universe, [0.15, 0.6, 1.0])
        bt['L'] = fuzz.smf(bt.universe, 0.8, 2.0)

        # theta_delta: large/medium/small left/right + zero
        td['NL'] = fuzz.zmf(td.universe, -math.pi / 6, -math.pi / 18)
        td['NM'] = fuzz.trimf(td.universe, [-math.pi / 6, -math.pi / 18, -math.pi / 36])
        td['NS'] = fuzz.trimf(td.universe, [-math.pi / 18, -math.pi / 36, 0.0])
        td['Z'] = fuzz.trimf(td.universe, [-math.pi / 90, 0.0, math.pi / 90])  # narrower aligned zone
        td['PS'] = fuzz.trimf(td.universe, [0.0, math.pi / 36, math.pi / 18])
        td['PM'] = fuzz.trimf(td.universe, [math.pi / 36, math.pi / 18, math.pi / 6])
        td['PL'] = fuzz.smf(td.universe, math.pi / 18, math.pi / 6)

        # ship_turn: large/medium/small left/right + zero
        turn['NL'] = fuzz.trimf(turn.universe, [-180.0, -180.0, -120.0])
        turn['NM'] = fuzz.trimf(turn.universe, [-150.0, -90.0, -45.0])
        turn['NS'] = fuzz.trimf(turn.universe, [-60.0, -30.0, 0.0])
        turn['Z'] = fuzz.trimf(turn.universe, [-10.0, 0.0, 10.0])
        turn['PS'] = fuzz.trimf(turn.universe, [0.0, 30.0, 60.0])
        turn['PM'] = fuzz.trimf(turn.universe, [45.0, 90.0, 150.0])
        turn['PL'] = fuzz.trimf(turn.universe, [120.0, 180.0, 180.0])

        # ship_fire: N = don’t fire, Y = fire (with a gap between them)
        fire['N'] = fuzz.trimf(fire.universe, [-1.0, -1.0, -0.2])
        fire['Y'] = fuzz.trimf(fire.universe, [0.2, 1.0, 1.0])

        # ---------- RULES ----------

        rules = []

        # LONG time-to-hit → conservative firing
        rules.append(ctrl.Rule(bt['L'] & td['NL'], (turn['NL'], fire['N'])))
        rules.append(ctrl.Rule(bt['L'] & td['NM'], (turn['NM'], fire['N'])))
        rules.append(ctrl.Rule(bt['L'] & td['NS'], (turn['NS'], fire['N'])))
        rules.append(ctrl.Rule(bt['L'] & td['Z'],  (turn['Z'],  fire['Y'])))
        rules.append(ctrl.Rule(bt['L'] & td['PS'], (turn['PS'], fire['N'])))
        rules.append(ctrl.Rule(bt['L'] & td['PM'], (turn['PM'], fire['N'])))
        rules.append(ctrl.Rule(bt['L'] & td['PL'], (turn['PL'], fire['N'])))

        # MEDIUM time-to-hit → fire when angle is small-ish
        rules.append(ctrl.Rule(bt['M'] & td['NL'], (turn['NL'], fire['N'])))
        rules.append(ctrl.Rule(bt['M'] & td['NM'], (turn['NM'], fire['N'])))
        rules.append(ctrl.Rule(bt['M'] & td['NS'], (turn['NS'], fire['Y'])))
        rules.append(ctrl.Rule(bt['M'] & td['Z'],  (turn['Z'],  fire['Y'])))
        rules.append(ctrl.Rule(bt['M'] & td['PS'], (turn['PS'], fire['Y'])))
        rules.append(ctrl.Rule(bt['M'] & td['PM'], (turn['PM'], fire['N'])))
        rules.append(ctrl.Rule(bt['M'] & td['PL'], (turn['PL'], fire['N'])))

        # SHORT time-to-hit → aggressive firing whenever somewhat lined up
        rules.append(ctrl.Rule(bt['S'] & td['NL'], (turn['NL'], fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['NM'], (turn['NM'], fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['NS'], (turn['NS'], fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['Z'],  (turn['Z'],  fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['PS'], (turn['PS'], fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['PM'], (turn['PM'], fire['Y'])))
        rules.append(ctrl.Rule(bt['S'] & td['PL'], (turn['PL'], fire['Y'])))

        # Build and store the control system
        self.aim_system = ctrl.ControlSystem(rules)

    # ------------------------------------------------------------------ #
    #  RUNTIME: called every frame
    # ------------------------------------------------------------------ #
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self.eval_frames += 1

        asteroids = game_state["asteroids"]
        if not asteroids:
            # No targets → hover in place (no thrust) and don’t fire
            return 0.0, 0.0, False, False

        # ---------- 1. Find nearest asteroid ----------
        ship_x, ship_y = ship_state["position"]
        closest_asteroid = None
        closest_dist_sq = None

        for a in asteroids:
            ax, ay = a["position"]
            dx = ax - ship_x
            dy = ay - ship_y
            d2 = dx * dx + dy * dy
            if closest_asteroid is None or d2 < closest_dist_sq:
                closest_asteroid = a
                closest_dist_sq = d2

        if closest_asteroid is None:
            return 0.0, 0.0, False, False

        dist = math.sqrt(closest_dist_sq)

        # ---------- 2. Compute intercept time (bullet_t) ----------
        ax, ay = closest_asteroid["position"]
        vx, vy = closest_asteroid["velocity"]

        # Vector from ship to asteroid
        rel_x = ax - ship_x
        rel_y = ay - ship_y

        asteroid_speed = math.sqrt(vx * vx + vy * vy)
        bullet_speed = 800.0  # from KesslerGame's bullet settings

        bullet_t = None

        if asteroid_speed < 1e-6 or asteroid_speed >= bullet_speed:
            # Simple straight-shot estimate
            if bullet_speed > 0:
                bullet_t = dist / bullet_speed
            else:
                bullet_t = 1.0
        else:
            # Law-of-Cosines-based analytic intercept
            asteroid_ship_theta = math.atan2(ship_y - ay, ship_x - ax)
            asteroid_direction = math.atan2(vy, vx)
            theta2 = asteroid_ship_theta - asteroid_direction
            cos_theta2 = math.cos(theta2)

            D = dist
            v = asteroid_speed
            b = bullet_speed

            targ_det = (-2 * D * v * cos_theta2) ** 2 - (4 * (v * v - b * b) * (D * D))

            if targ_det < 0:
                bullet_t = dist / b if b > 0 else 1.0
            else:
                sqrt_det = math.sqrt(targ_det)
                denom = 2 * (v * v - b * b)

                t1 = ((2 * D * v * cos_theta2) + sqrt_det) / denom
                t2 = ((2 * D * v * cos_theta2) - sqrt_det) / denom

                candidates = [t for t in (t1, t2) if t >= 0]
                if candidates:
                    bullet_t = min(candidates)
                else:
                    bullet_t = dist / b if b > 0 else 1.0

        # Clamp bullet_t into the fuzzy domain
        if not math.isfinite(bullet_t):
            bullet_t = 2.0
        bullet_t = max(0.0, min(2.0, bullet_t))

        # ---------- 3. Compute intercept point & theta_delta ----------
        dt = 1.0 / 30.0
        intrcpt_x = ax + vx * (bullet_t + dt)
        intrcpt_y = ay + vy * (bullet_t + dt)

        # Desired firing angle (rad)
        desired_theta = math.atan2(intrcpt_y - ship_y, intrcpt_x - ship_x)

        # Ship heading is in degrees; convert to radians
        heading_rad = ship_state["heading"] * math.pi / 180.0

        theta_delta = desired_theta - heading_rad
        theta_delta = (theta_delta + math.pi) % (2 * math.pi) - math.pi  # wrap into (-pi, pi]

        # Clamp into fuzzy domain
        theta_delta = max(-math.pi / 6, min(math.pi / 6, theta_delta))

        # ---------- 4. Run fuzzy AIM controller ----------
        try:
            sim = ctrl.ControlSystemSimulation(self.aim_system, flush_after_run=1)
            sim.input['bullet_time'] = bullet_t
            sim.input['theta_delta'] = theta_delta
            sim.compute()

            turn_rate = float(sim.output['ship_turn'])
            fire_val = sim.output['ship_fire']

            # Make turning more aggressive
            turn_rate *= self.turn_gain

        except Exception:
            # Fallback: simple proportional controller if fuzzy blows up
            Kp = 400.0
            turn_rate = Kp * theta_delta * 180.0 / math.pi
            fire_val = 1.0 if abs(theta_delta) < (5.0 * math.pi / 180.0) else -1.0

        # Clamp turn rate to safe range
        turn_rate = max(-180.0, min(180.0, float(turn_rate)))
        fire = bool(fire_val >= self.fire_threshold)

        # ---- extra hard aim gate to avoid angle misses while swinging ----
        theta_err_deg = abs(theta_delta * 180.0 / math.pi)
        if bullet_t < 0.3:
            angle_limit_deg = 12.0    # close & fast, a bit forgiving
        elif bullet_t < 0.8:
            angle_limit_deg = 8.0     # medium shots
        else:
            angle_limit_deg = 5.0     # long shots: only when very well aligned

        if theta_err_deg > angle_limit_deg:
            fire = False

        # ---------- 5. FUZZY MOVEMENT, DANGER & SAFETY ----------

        actual_dist = dist
        dist_error = actual_dist - self.hover_distance  # negative = too close; positive = too far

        # Clamp dist_error into its fuzzy universe [-2*hd, 2*hd]
        max_err = 2.0 * self.hover_distance
        dist_error = max(-max_err, min(max_err, dist_error))

        # Ship velocity and forward direction
        vx_ship, vy_ship = ship_state["velocity"]
        speed = math.hypot(vx_ship, vy_ship)
        fwd_x = math.cos(heading_rad)
        fwd_y = math.sin(heading_rad)

        # Count nearby asteroids (for crowd)
        crowd_count = 0
        for a in asteroids:
            ax2, ay2 = a["position"]
            if math.hypot(ax2 - ship_x, ay2 - ship_y) < 100.0:
                crowd_count += 1

        # Clamp crowd into [0, 19] (crowd.universe = 0..19)
        crowd_clamped = max(0, min(19, crowd_count))

        # --------- Incoming asteroid detection / escape direction ----------
        danger_sum_x = 0.0
        danger_sum_y = 0.0
        total_danger_weight = 0.0

        for a in asteroids:
            ax2, ay2 = a["position"]
            vx2, vy2 = a["velocity"]

            rx = ax2 - ship_x
            ry = ay2 - ship_y
            dist_a = math.hypot(rx, ry)
            if dist_a < 1e-3 or dist_a > self.danger_distance:
                continue

            # Relative velocity (asteroid minus ship)
            v_rel_x = vx2 - vx_ship
            v_rel_y = vy2 - vy_ship

            # Closing speed along line-of-sight:
            # closing < 0 means approaching (since r·v_rel < 0)
            closing = (rx * v_rel_x + ry * v_rel_y) / dist_a

            if closing < -self.danger_closing_thresh:
                weight = -closing  # positive weight proportional to closing speed
                total_danger_weight += weight
                danger_sum_x += (rx / dist_a) * weight
                danger_sum_y += (ry / dist_a) * weight

        escape_dir_x = 0.0
        escape_dir_y = 0.0
        escape_bias = 0.0

        if total_danger_weight > 0.0:
            # Direction away from weighted dangerous asteroids
            ex = -danger_sum_x
            ey = -danger_sum_y
            norm_e = math.hypot(ex, ey)
            if norm_e > 1e-6:
                escape_dir_x = ex / norm_e
                escape_dir_y = ey / norm_e
                # How well does our current forward direction align with escape?
                escape_bias = escape_dir_x * fwd_x + escape_dir_y * fwd_y  # in [-1,1]

        # ---------- helper: check if it's dangerous to back up ----------
        def backing_is_dangerous() -> bool:
            for a in asteroids:
                ax2, ay2 = a["position"]
                dx = ax2 - ship_x
                dy = ay2 - ship_y

                # Forward component (along ship heading)
                forward_proj = dx * fwd_x + dy * fwd_y  # >0 in front, <0 behind

                # Lateral component magnitude (perpendicular distance from axis)
                side_proj = abs(dx * (-fwd_y) + dy * fwd_x)

                # Asteroid is behind and within a capsule region
                if -self.safe_back_distance <= forward_proj <= 0 and side_proj <= self.safe_back_side:
                    return True
            return False

        # ---------- Panic safety: very close asteroid ----------
        if dist < self.panic_distance:
            # If we have a clear escape direction along heading, move that way
            if total_danger_weight > 0.0 and abs(escape_bias) > 0.2:
                thrust = 180.0 if escape_bias > 0 else -180.0
            else:
                thrust = -180.0  # default: back up

            # If backing is dangerous, don't go backwards
            if thrust < 0 and backing_is_dangerous():
                thrust = 0.0

            # Speed damping
            if speed > self.max_speed:
                scale = max(0.0, 1.0 - (speed - self.max_speed) / self.max_speed)
                thrust *= scale

            # Mine-drop logic during panic: only when moving forward fast into a crowd
            forward_speed = vx_ship * fwd_x + vy_ship * fwd_y
            if crowd_count >= self.mine_panic_count and forward_speed > self.mine_forward_thresh:
                drop_mine = True
            else:
                drop_mine = False

            return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

        # ---------- Normal fuzzy movement (with escape bias & fallback) ----------
        try:
            movement = ctrl.ControlSystemSimulation(self.movement_control, flush_after_run=1)
            movement.input['dist_err'] = dist_error
            movement.input['crowd'] = crowd_clamped
            movement.compute()

            thrust = float(movement.output['thrust'])
        except Exception:
            # Fallback: simple proportional hover control
            K_hover = 1.0
            thrust = K_hover * dist_error
            thrust = max(-220.0, min(220.0, thrust))

        # If strong incoming danger, bias thrust toward escape direction
        if total_danger_weight > self.danger_weight_thresh and abs(escape_bias) > 0.2:
            escape_thrust = self.escape_thrust * (1.0 if escape_bias > 0 else -1.0)
            # Blend fuzzy hover thrust and escape thrust
            thrust = 0.5 * thrust + 0.5 * escape_thrust

        # Don’t back into asteroids
        if thrust < 0 and backing_is_dangerous():
            thrust = 0.0

        # Simple speed damping to avoid going crazy-fast
        if speed > self.max_speed:
            scale = max(0.0, 1.0 - (speed - self.max_speed) / self.max_speed)
            thrust *= scale

        # Mine-drop logic: only when moving forward fast into a crowd
        forward_speed = vx_ship * fwd_x + vy_ship * fwd_y
        if crowd_count >= self.mine_panic_count and forward_speed > self.mine_forward_thresh:
            drop_mine = True
        else:
            drop_mine = False

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "Hybrid Hover Aimbot"
