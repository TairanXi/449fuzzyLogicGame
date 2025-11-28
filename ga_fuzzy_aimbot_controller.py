# ga_fuzzy_aimbot_controller.py
# Genetic-Fuzzy "Aimbot" Controller for Kessler Game (ECE 449)
# - Picks the most dangerous asteroid (incoming + short time-to-collision),
#   otherwise the nearest.
# - Solves bullet/asteroid intercept analytically, then uses a fuzzy rulebase
#   to produce turn_rate and fire.
# - Includes a simple Genetic Algorithm (GA) to tune fuzzy parameters.
#
# Usage:
#   from ga_fuzzy_aimbot_controller import GeneticFuzzyAimbotController, GAParams
#   # To just play (uses good default params):
#   ctrl = GeneticFuzzyAimbotController()
#   # To optimize params (optional; takes minutes depending on gens/pop):
#   best = GeneticFuzzyAimbotController.optimize(
#       ga=GAParams(generations=8, population=20, elites=3, mutation_prob=0.25),
#       eval_games=3,   # number of random scenarios per candidate
#       seed=42
#   )
#   ctrl = GeneticFuzzyAimbotController(best_params=best)
#
# Notes:
# - Requires: kesslergame, numpy, scikit-fuzzy.
# - The GA uses TrainerEnvironment (no graphics) if available; otherwise falls back to KesslerGame with graphics off.

from typing import Dict, Tuple, List, Optional, NamedTuple
import math, random, copy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

try:
    # Kessler types
    from kesslergame import KesslerController, Scenario, KesslerGame, GraphicsType, TrainerEnvironment
except ImportError:
    from kesslergame import KesslerController, Scenario, KesslerGame, GraphicsType
    TrainerEnvironment = None  # older versions

# ----------------------------- Tunable parameter vector -----------------------------

class FuzzyParams(NamedTuple):
    # bullet_time universe split points (seconds)
    bt_S_max: float       # end of Small
    bt_M_peak: float      # center of Medium
    bt_L_start: float     # start (foot) of Large S-shaped rise

    # theta_delta partition in radians (about 6° = pi/30 is one-tick max turn)
    th_small: float       # |theta| <= th_small ~ "S"
    th_med: float         # th_small..th_med ~ "M", beyond ~ "L" (saturates)

    # ship_turn output scaling (degrees/sec) — will clip to [-180, 180]
    turn_NS: float        # negative small peak
    turn_PS: float        # positive small peak

    # firing threshold on defuzzified ship_fire ([-1, 1] universe)
    fire_thresh: float

# Good defaults (perform well without GA)
DEFAULT_PARAMS = FuzzyParams(
    bt_S_max=0.05, bt_M_peak=0.08, bt_L_start=0.10,
    th_small=math.pi/90, th_med=2*math.pi/90,
    turn_NS=-90.0, turn_PS=90.0,
    fire_thresh=0.0
)

# ----------------------------- Controller -----------------------------

class GeneticFuzzyAimbotController(KesslerController):
    """
    Genetic-Fuzzy controller:
      - Chooses "dangerous" asteroid by minimizing time-to-collision (TTC) with the ship
        among those closing (radial_rel_vel < 0). If none closing, picks nearest.
      - Computes analytical bullet/asteroid intercept (per course guide).
      - Fuzzy rulebase maps (bullet_time, theta_delta) -> (turn_rate, fire).
      - Parameters are GA-tunable via optimize() below.
    """

    BULLET_SPEED = 800.0   # m/s (Kessler default)

    def __init__(self, best_params: Optional[FuzzyParams] = None):
        self.eval_frames = 0
        self.params: FuzzyParams = best_params or DEFAULT_PARAMS
        self._build_fuzzy(self.params)

    # ---------------- fuzzy system builder ----------------

    def _build_fuzzy(self, P: FuzzyParams):
        # Universes
        bullet_time = ctrl.Antecedent(np.arange(0, 0.8, 0.002), 'bullet_time')  # ample range
        theta_delta = ctrl.Antecedent(np.arange(-math.pi/10, math.pi/10, 0.001), 'theta_delta')
        ship_turn   = ctrl.Consequent(np.arange(-180, 181, 1), 'ship_turn')     # deg/sec
        ship_fire   = ctrl.Consequent(np.arange(-1, 1.01, 0.01), 'ship_fire')   # pseudo-boolean

        # bullet_time sets: S (fast), M, L (slow)
        # S: 0 .. S_max (tri with peak at 0)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0.0, 0.0, P.bt_S_max])
        # M: around M_peak, spanning roughly [S_max/2 .. L_start]
        m_left = max(0.0, P.bt_S_max * 0.5)
        m_right = P.bt_L_start
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [m_left, P.bt_M_peak, m_right])
        # L: smooth S-rise from L_start upward
        bullet_time['L'] = fuzz.smf(bullet_time.universe, P.bt_L_start, P.bt_L_start + 0.12)

        # theta_delta sets (aiming error): NL..PL with saturation for |theta| > th_med
        thS, thM = P.th_small, P.th_med
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -thM, -thS)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-thM, -thS, 0.0])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-thS, 0.0, thS])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [ -0.0, thS, thM])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [ thS, thM, 2*thM])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, thS, thM)

        # ship_turn sets — proportional to error magnitude (symmetric)
        # (We keep triangles; GA tunes peaks for “small” corrections)
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120,  -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [ -120,  P.turn_NS, 0])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [ 0,  P.turn_PS, 120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [  60,  120,  180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [ 120,  180,  180])

        # ship_fire sets
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1.0, -1.0, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [ 0.0,  1.0, 1.0])

        # Rules: reduce |theta|; more aggressive firing when bullet_time is small
        R = []
        def r(bt, th, turn, fire):
            R.append(ctrl.Rule(bullet_time[bt] & theta_delta[th], (ship_turn[turn], ship_fire[fire])))

        # Long bullet time: be pickier; only fire when nearly on target
        r('L','NL','NL','N'); r('L','NM','NM','N'); r('L','NS','NS','Y')
        r('L','PS','PS','Y'); r('L','PM','PM','N'); r('L','PL','PL','N')

        # Medium: moderate
        r('M','NL','NL','N'); r('M','NM','NM','N'); r('M','NS','NS','Y')
        r('M','PS','PS','Y'); r('M','PM','PM','N'); r('M','PL','PL','N')

        # Small: permissive — spray when aim is “good enough”
        r('S','NL','NL','Y'); r('S','NM','NM','Y'); r('S','NS','NS','Y')
        r('S','PS','PS','Y'); r('S','PM','PM','Y'); r('S','PL','PL','Y')

        self._ctrl = ctrl.ControlSystem(R)
        self._sim  = ctrl.ControlSystemSimulation(self._ctrl, flush_after_run=1)
        self._fire_thresh = float(P.fire_thresh)

    # ---------------- targeting ----------------

    @staticmethod
    def _choose_target(ship_state: Dict, asteroids: List[Dict]) -> Optional[Dict]:
        """Pick asteroid with minimum positive time-to-collision (TTC) if closing; else nearest."""
        sx, sy = ship_state["position"]
        svx, svy = ship_state["velocity"]
        # Treat ship as approximately stationary for TTC (stable enough in our strategy)
        best_ttc, best_close = float('inf'), None
        nearest, nearest_d2  = None, float('inf')

        for a in asteroids:
            ax, ay = a["position"]; avx, avy = a["velocity"]
            rx, ry = ax - sx, ay - sy
            rvx, rvy = avx - svx, avy - svy  # relative vel asteroid->ship
            r2 = rx*rx + ry*ry
            if r2 < nearest_d2:
                nearest_d2 = r2; nearest = a
            # radial closing speed (project rel-vel onto line of sight)
            r = math.hypot(rx, ry)
            if r < 1e-6:   # on top of us
                return a
            closing = (rx*rvx + ry*rvy) / r   # >0 means increasing range; <0 closing
            if closing < 0:
                ttc = -r / closing            # seconds to impact along current LOS
                if 0 <= ttc < best_ttc:
                    best_ttc, best_close = ttc, a

        return best_close if best_close is not None else nearest

    @staticmethod
    def _intercept_time(ship_pos, ast_pos, ast_vel, bullet_speed) -> float:
        """Solve quadratic for intercept time; choose smallest nonnegative solution."""
        sx, sy = ship_pos; ax, ay = ast_pos; avx, avy = ast_vel
        dx, dy = ax - sx, ay - sy
        d = math.hypot(dx, dy)

        # angle between D and asteroid velocity
        theta2 = math.atan2(dy, dx) - math.atan2(avy, avx)
        cos2 = math.cos(theta2)
        va = math.hypot(avx, avy); vb = bullet_speed

        # a t^2 + b t + c = 0
        a = va*va - vb*vb
        b = 2.0 * d * va * cos2
        c = d*d
        det = b*b - 4*a*c
        if det < 0:  # no real intercept (rare with vb >> va); shoot where it is
            return max(0.0, c / (vb*vb + 1e-6))

        sqrt_det = math.sqrt(det)
        t1 = (-b + sqrt_det) / (2*a) if a != 0 else float('inf')
        t2 = (-b - sqrt_det) / (2*a) if a != 0 else float('inf')

        # pick smallest nonnegative
        candidates = [t for t in (t1, t2) if t >= 0]
        if not candidates:
            return max(t1, t2)  # least-bad
        return min(candidates)

    # ---------------- action loop ----------------

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust = 0.0      # this controller focuses on aim; feel free to add evasive thrust later
        drop_mine = False

        target = self._choose_target(ship_state, game_state["asteroids"])
        if not target:
            self.eval_frames += 1
            return thrust, 0.0, False, drop_mine

        sx, sy = ship_state["position"]
        ax, ay = target["position"]
        avx, avy = target["velocity"]

        # Intercept time per guide; add 1/30s tick lead to account for frame delay
        t = self._intercept_time((sx, sy), (ax, ay), (avx, avy), self.BULLET_SPEED) + (1.0/30.0)

        ix = ax + avx * t
        iy = ay + avy * t

        desired = math.atan2(iy - sy, ix - sx)               # θ1
        heading = (math.pi/180.0) * ship_state["heading"]    # radians
        dtheta = desired - heading
        # Wrap to (-π, π]
        dtheta = (dtheta + math.pi) % (2*math.pi) - math.pi

        # Fuzzy infer
        self._sim.input['bullet_time'] = float(max(0.0, min(t, 0.79)))
        self._sim.input['theta_delta'] = float(np.clip(dtheta, -math.pi/10, math.pi/10))
        self._sim.compute()

        turn_rate = float(np.clip(self._sim.output['ship_turn'], -180.0, 180.0))
        fire_val = self._sim.output['ship_fire']
        fire = bool(fire_val >= self._fire_thresh)

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "GeneticFuzzy Aimbot"

    # ---------------- GA optimizer ----------------

    class GAParams(NamedTuple):
        generations: int = 6
        population: int = 18
        elites: int = 2
        mutation_prob: float = 0.2
        crossover_prob: float = 0.9

    @staticmethod
    def random_params(r: random.Random) -> FuzzyParams:
        bt_S_max = r.uniform(0.02, 0.10)
        bt_M_peak = r.uniform(bt_S_max*0.6, 0.14)
        bt_L_start = r.uniform(max(0.06, bt_S_max*0.9), 0.20)
        th_small = r.uniform(math.pi/120, math.pi/70)    # ~1.5°..~2.6°
        th_med   = r.uniform(th_small*1.8, th_small*3.5) # ~3°..~9°
        turn_NS  = -r.uniform(50, 120)
        turn_PS  =  r.uniform(50, 120)
        fire_th  = r.uniform(-0.2, 0.2)
        return FuzzyParams(bt_S_max, bt_M_peak, bt_L_start, th_small, th_med, turn_NS, turn_PS, fire_th)

    @staticmethod
    def mutate(p: FuzzyParams, r: random.Random, scale: float = 0.25) -> FuzzyParams:
        def jitter(val, lo, hi, mag):
            delta = (hi - lo) * mag * r.uniform(-1, 1)
            return min(hi, max(lo, val + delta))
        return FuzzyParams(
            jitter(p.bt_S_max, 0.015, 0.12, scale),
            jitter(p.bt_M_peak, 0.02,  0.18, scale),
            jitter(p.bt_L_start,0.06,  0.25, scale),
            jitter(p.th_small,  math.pi/140, math.pi/60, scale),
            jitter(p.th_med,    math.pi/80,  math.pi/25, scale),
            jitter(p.turn_NS,  -140.0, -40.0, scale),
            jitter(p.turn_PS,    40.0,  140.0, scale),
            jitter(p.fire_thresh,-0.4,   0.4,  scale),
        )

    @staticmethod
    def crossover(a: FuzzyParams, b: FuzzyParams, r: random.Random) -> FuzzyParams:
        al = list(a); bl = list(b)
        cut = r.randint(1, len(al)-2)
        child = al[:cut] + bl[cut:]
        return FuzzyParams(*child)

    class Score(NamedTuple):
        fitness: float
        hits: int
        accuracy: float
        deaths: int

    @staticmethod
    def _eval_params(P: FuzzyParams, eval_games: int, rng: random.Random) -> "GeneticFuzzyAimbotController.Score":
        """Run a few randomized scenarios and return a single scalar fitness."""
        # Game environment (no graphics if possible)
        settings = {'perf_tracker': True, 'graphics_type': GraphicsType.None_ if hasattr(GraphicsType, 'None_') else GraphicsType.Tkinter,
                    'realtime_multiplier': 0, 'graphics_obj': None}
        env = TrainerEnvironment(settings=settings) if TrainerEnvironment else KesslerGame(settings=settings)

        total_hits = total_deaths = 0
        acc_list = []

        for _ in range(eval_games):
            # random scenario emphasizing danger (some fast + some near)
            W, H = 1000, 800
            num_ast = rng.randint(6, 10)
            asts = []
            for _i in range(num_ast):
                px = rng.uniform(100, W-100)
                py = rng.uniform(100, H-100)
                speed = rng.uniform(40, 140)
                ang = rng.uniform(0, 2*math.pi)
                vx, vy = speed*math.cos(ang), speed*math.sin(ang)
                asts.append({'position': (px, py), 'velocity': (vx, vy), 'size': rng.choice([1,2,3])})

            scenario = Scenario(
                name='GAEval',
                num_asteroids=num_ast,
                ship_states=[{'position': (W/2, H/2), 'angle': 90, 'lives': 3, 'team': 1}],
                map_size=(W, H),
                time_limit=40,
                ammo_limit_multiplier=0,
                stop_if_no_ammo=False
            )
            # Monkeypatch asteroids into scenario if TrainerEnvironment not exposing constructor details
            # (Most versions spawn randomly; this randomized scenario is enough for selection pressure.)

            ctrlr = GeneticFuzzyAimbotController(best_params=P)
            score, perf = env.run(scenario=scenario, controllers=[ctrlr])

            team = score.teams[0]
            total_hits += team.asteroids_hit
            total_deaths += team.deaths
            acc_list.append(team.accuracy or 0.0)

        mean_acc = float(np.mean(acc_list)) if acc_list else 0.0

        # Fitness: reward hits and accuracy, penalize deaths
        fitness = (2.5 * total_hits) + (1.5 * mean_acc) - (3.0 * total_deaths)
        return GeneticFuzzyAimbotController.Score(fitness=fitness, hits=total_hits, accuracy=mean_acc, deaths=total_deaths)

    GAParams = GAParams  # expose type

    @staticmethod
    def optimize(ga: "GeneticFuzzyAimbotController.GAParams" = None,
                 eval_games: int = 3,
                 seed: Optional[int] = None) -> FuzzyParams:
        """Run a simple GA to find good fuzzy parameters. Returns best FuzzyParams."""
        ga = ga or GeneticFuzzyAimbotController.GAParams()
        rng = random.Random(seed)

        # Initialize population around defaults + randoms
        pop: List[FuzzyParams] = [DEFAULT_PARAMS]
        pop += [GeneticFuzzyAimbotController.random_params(rng) for _ in range(ga.population - 1)]

        for gen in range(ga.generations):
            scored = [(p, GeneticFuzzyAimbotController._eval_params(p, eval_games, rng)) for p in pop]
            scored.sort(key=lambda x: x[1].fitness, reverse=True)
            # print progress (optional)
            bestP, bestS = scored[0]
            print(f"[GA] Gen {gen+1}/{ga.generations}  fitness={bestS.fitness:.2f}  hits={bestS.hits}  acc={bestS.accuracy:.3f}  deaths={bestS.deaths}")

            # Elitism
            new_pop = [scored[i][0] for i in range(min(ga.elites, len(scored)))]

            # Breed
            while len(new_pop) < ga.population:
                a = rng.choice(scored[: max(4, len(scored)//2)])[0]
                b = rng.choice(scored[: max(4, len(scored)//2)])[0]
                child = GeneticFuzzyAimbotController.crossover(a, b, rng) if rng.random() < ga.crossover_prob else copy.deepcopy(a)
                if rng.random() < ga.mutation_prob:
                    child = GeneticFuzzyAimbotController.mutate(child, rng)
                new_pop.append(child)

            pop = new_pop

        # Final evaluation
        final = [(p, GeneticFuzzyAimbotController._eval_params(p, eval_games*2, rng)) for p in pop]
        final.sort(key=lambda x: x[1].fitness, reverse=True)
        bestP, bestS = final[0]
        print(f"[GA] Best fitness={bestS.fitness:.2f}  hits={bestS.hits}  acc={bestS.accuracy:.3f}  deaths={bestS.deaths}")
        return bestP
