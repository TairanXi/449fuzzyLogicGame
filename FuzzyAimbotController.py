from typing import Dict, Tuple
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import KesslerController


class FuzzyAimbotController(KesslerController):
    """
    Fuzzy-logic controller that:
    - Targets the nearest asteroid
    - Computes an intercept point
    - Uses fuzzy logic to pick turn_rate + fire
    - Keeps thrust 0 (stationary sniper)
    """

    def __init__(self):
        self.eval_frames = 0

        # ---------- FUZZY VARIABLES ----------

        # bullet_time: time (s) until bullet reaches intercept point
        bt = ctrl.Antecedent(np.linspace(0.0, 1.0, 501), 'bullet_time')

        # theta_delta: angle error (rad) between ship heading and desired firing angle
        # Domain: about ±30 degrees
        td = ctrl.Antecedent(np.linspace(-math.pi / 6, math.pi / 6, 301), 'theta_delta')

        # ship_turn: output turn rate (deg/sec)
        turn = ctrl.Consequent(np.linspace(-180.0, 180.0, 361), 'ship_turn')

        # ship_fire: output "fire strength" in [-1, 1] → thresholded to bool
        fire = ctrl.Consequent(np.linspace(-1.0, 1.0, 201), 'ship_fire')

        # ---------- MEMBERSHIP FUNCTIONS ----------

        # bullet_time: Short / Medium / Long
        bt['S'] = fuzz.trimf(bt.universe, [0.0, 0.0, 0.15])
        bt['M'] = fuzz.trimf(bt.universe, [0.05, 0.25, 0.45])
        bt['L'] = fuzz.smf(bt.universe, 0.3, 0.7)

        # theta_delta: large/medium/small left/right + zero
        td['NL'] = fuzz.zmf(td.universe, -math.pi / 6, -math.pi / 18)
        td['NM'] = fuzz.trimf(td.universe, [-math.pi / 6, -math.pi / 18, -math.pi / 36])
        td['NS'] = fuzz.trimf(td.universe, [-math.pi / 18, -math.pi / 36, 0.0])
        td['Z']  = fuzz.trimf(td.universe, [-math.pi / 72, 0.0, math.pi / 72])
        td['PS'] = fuzz.trimf(td.universe, [0.0, math.pi / 36, math.pi / 18])
        td['PM'] = fuzz.trimf(td.universe, [math.pi / 36, math.pi / 18, math.pi / 6])
        td['PL'] = fuzz.smf(td.universe, math.pi / 18, math.pi / 6)

        # ship_turn: large/medium/small left/right + zero
        turn['NL'] = fuzz.trimf(turn.universe, [-180.0, -180.0, -120.0])
        turn['NM'] = fuzz.trimf(turn.universe, [-150.0, -90.0, -45.0])
        turn['NS'] = fuzz.trimf(turn.universe, [-60.0, -30.0, 0.0])
        turn['Z']  = fuzz.trimf(turn.universe, [-10.0, 0.0, 10.0])
        turn['PS'] = fuzz.trimf(turn.universe, [0.0, 30.0, 60.0])
        turn['PM'] = fuzz.trimf(turn.universe, [45.0, 90.0, 150.0])
        turn['PL'] = fuzz.trimf(turn.universe, [120.0, 180.0, 180.0])

        # ship_fire: N = don’t fire, Y = fire
        fire['N'] = fuzz.trimf(fire.universe, [-1.0, -1.0, 0.0])
        fire['Y'] = fuzz.trimf(fire.universe, [0.0, 1.0, 1.0])

        # ---------- RULES ----------

        rules = []

        # LONG time-to-hit → be conservative: mostly aim, fire only when angle tiny
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

        # KesslerGame's GameState supports indexing but not .get()
        asteroids = game_state["asteroids"]
        if not asteroids:
            # No targets → do nothing
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

        # If asteroid is basically stationary or slower than bullet, we can approximate:
        if asteroid_speed < 1e-6 or asteroid_speed >= bullet_speed:
            # Simple straight-shot estimate
            if bullet_speed > 0:
                bullet_t = dist / bullet_speed
            else:
                bullet_t = 1.0
        else:
            # Use Law-of-Cosines-based analytic intercept (like Scott's)
            asteroid_ship_theta = math.atan2(ship_y - ay, ship_x - ax)
            asteroid_direction = math.atan2(vy, vx)
            theta2 = asteroid_ship_theta - asteroid_direction
            cos_theta2 = math.cos(theta2)

            # Quadratic coefficients embedded in discriminant calc
            D = dist
            v = asteroid_speed
            b = bullet_speed

            # Discriminant for quadratic in t
            targ_det = (-2 * D * v * cos_theta2) ** 2 - (4 * (v * v - b * b) * (D * D))

            if targ_det < 0:
                # No real solution → fall back to naive distance-based time
                bullet_t = dist / b if b > 0 else 1.0
            else:
                sqrt_det = math.sqrt(targ_det)
                denom = 2 * (v * v - b * b)

                t1 = ((2 * D * v * cos_theta2) + sqrt_det) / denom
                t2 = ((2 * D * v * cos_theta2) - sqrt_det) / denom

                # Choose the smallest non-negative solution
                candidates = [t for t in (t1, t2) if t >= 0]
                if candidates:
                    bullet_t = min(candidates)
                else:
                    bullet_t = dist / b if b > 0 else 1.0

        # Clamp bullet_t into the fuzzy domain
        if not math.isfinite(bullet_t):
            bullet_t = 1.0
        bullet_t = max(0.0, min(1.0, bullet_t))

        # ---------- 3. Compute intercept point & theta_delta ----------
        # Predict asteroid position at (bullet_t + dt)
        dt = 1.0 / 30.0
        intrcpt_x = ax + vx * (bullet_t + dt)
        intrcpt_y = ay + vy * (bullet_t + dt)

        # Desired firing angle (rad)
        desired_theta = math.atan2(intrcpt_y - ship_y, intrcpt_x - ship_x)

        # Ship heading is in degrees; convert to radians
        heading_rad = ship_state["heading"] * math.pi / 180.0

        theta_delta = desired_theta - heading_rad
        # Wrap into (-pi, pi]
        theta_delta = (theta_delta + math.pi) % (2 * math.pi) - math.pi

        # Clamp into fuzzy domain
        theta_delta = max(-math.pi / 6, min(math.pi / 6, theta_delta))

        # ---------- 4. Run fuzzy controller ----------
        try:
            sim = ctrl.ControlSystemSimulation(self.aim_system, flush_after_run=1)
            sim.input['bullet_time'] = bullet_t
            sim.input['theta_delta'] = theta_delta
            sim.compute()

            turn_rate = sim.output['ship_turn']
            fire_val = sim.output['ship_fire']

        except Exception:
            # Fallback: simple proportional controller if fuzzy blows up
            Kp = 300.0  # proportional gain
            turn_rate = Kp * theta_delta * 180.0 / math.pi  # rad → deg
            fire_val = 1.0 if abs(theta_delta) < (5.0 * math.pi / 180.0) else -1.0

        # Clamp turn rate to safe range
        turn_rate = max(-180.0, min(180.0, float(turn_rate)))

        fire = bool(fire_val >= 0.0)

        thrust = 0.0       # Stationary sniper
        drop_mine = False  # No mines

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Fuzzy Aimbot Controller"
