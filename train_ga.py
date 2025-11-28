# train_ga.py
"""
Minimal GA to tune HybridParams for HybridHoverAimbot.

Usage:
    python train_ga.py

Notes:
- Pure Python + NumPy (no DEAP).
- Explicitly imports your scenario to avoid engine-specific fallbacks.
- Prints a ready-to-paste HybridParams dict at the end.
"""

import math
import random
import importlib
import numpy as np

from kesslergame import KesslerGame
from hybrid_hover_aimbot_improved import HybridHoverAimbot, HybridParams

# --------------------------------------------------------------------
# Configure where your scenario lives:
#   - File:   scenario_test.py
#   - Symbol: my_test_scenario (callable or object)
# If your names differ, change these two constants.
# --------------------------------------------------------------------
SCENARIO_MODULE = "scenario_test"
SCENARIO_SYMBOL = "my_test_scenario"


# ----------------------------
# Scenario retrieval (explicit)
# ----------------------------
def get_eval_scenarios():
    """
    Return a list with exactly the scenario you want to evaluate on.
    - If the symbol is callable, we keep it callable.
    - If it's a constant object, we wrap it so the evaluator can call it uniformly.
    """
    try:
        mod = importlib.import_module(SCENARIO_MODULE)
        scn = getattr(mod, SCENARIO_SYMBOL)
    except Exception as e:
        raise RuntimeError(
            f"Could not import scenario '{SCENARIO_SYMBOL}' from module "
            f"'{SCENARIO_MODULE}'. Adjust SCENARIO_MODULE/SCENARIO_SYMBOL. "
            f"Original error: {e}"
        )

    if callable(scn):
        return [scn]
    else:
        # Wrap non-callable into a callable that ignores seed
        return [lambda _seed, _scn=scn: _scn]


# ----------------------------
# GA search space
# ----------------------------
# Each gene is (name, lo, hi)
GENES = [
    ("hover_distance",          80.0, 200.0),
    ("max_thrust_abs",          140.0, 260.0),
    ("max_speed",               120.0, 240.0),
    ("crowd_radius",            90.0,  200.0),
    ("safe_bubble",             60.0,  120.0),
    ("w_close",                 1.0,   4.0),
    ("w_dist",                  0.5,   2.0),

    # Thrust MFs (keep ordering sane: left <= peak <= right)
    ("thrust_back_high_left",  -300.0, -180.0),
    ("thrust_back_high_peak",  -300.0, -180.0),
    ("thrust_back_high_right", -260.0, -120.0),

    ("thrust_back_med_left",   -220.0, -100.0),
    ("thrust_back_med_peak",   -160.0,  -80.0),
    ("thrust_back_med_right",   -80.0,   -10.0),

    ("thrust_zero_left",        -80.0,   -10.0),
    ("thrust_zero_peak",        -10.0,    10.0),
    ("thrust_zero_right",         10.0,    80.0),

    ("thrust_fwd_low_left",       10.0,    80.0),
    ("thrust_fwd_low_peak",       30.0,   120.0),
    ("thrust_fwd_low_right",      60.0,   160.0),

    ("thrust_fwd_med_left",       50.0,   120.0),
    ("thrust_fwd_med_peak",       80.0,   160.0),
    ("thrust_fwd_med_right",     120.0,   200.0),

    ("thrust_fwd_high_left",     110.0,   180.0),
    ("thrust_fwd_high_peak",     140.0,   220.0),
    ("thrust_fwd_high_right",    160.0,   260.0),

    ("panic_count",                4.0,    10.0),
    ("reverse_speed_thresh",    -180.0,   -60.0),

    ("bullet_speed",             700.0,   900.0),
    ("aim_theta_cap_deg",         20.0,    40.0),  # will convert to radians
    ("aim_dt_lead",                0.0,     0.05),
]


def clip_ordered(a, b, c):
    """Ensure a <= b <= c by clamping midpoints sensibly."""
    if b < a: b = a
    if c < b: c = b
    return a, b, c


def vec_to_params(vec):
    """
    Convert GA vector -> HybridParams, enforcing ordered MF constraints.
    """
    vals = {name: val for (name, _lo, _hi), val in zip(GENES, vec)}

    # Fix ordered triples
    tbhL, tbhP, tbhR = clip_ordered(vals["thrust_back_high_left"], vals["thrust_back_high_peak"], vals["thrust_back_high_right"])
    tbmL, tbmP, tbmR = clip_ordered(vals["thrust_back_med_left"],  vals["thrust_back_med_peak"],  vals["thrust_back_med_right"])
    tzL,  tzP,  tzR  = clip_ordered(vals["thrust_zero_left"],      vals["thrust_zero_peak"],      vals["thrust_zero_right"])
    tflL, tflP, tflR = clip_ordered(vals["thrust_fwd_low_left"],   vals["thrust_fwd_low_peak"],   vals["thrust_fwd_low_right"])
    tfmL, tfmP, tfmR = clip_ordered(vals["thrust_fwd_med_left"],   vals["thrust_fwd_med_peak"],   vals["thrust_fwd_med_right"])
    tfhL, tfhP, tfhR = clip_ordered(vals["thrust_fwd_high_left"],  vals["thrust_fwd_high_peak"],  vals["thrust_fwd_high_right"])

    return HybridParams(
        hover_distance=vals["hover_distance"],
        max_thrust_abs=vals["max_thrust_abs"],
        max_speed=vals["max_speed"],
        crowd_radius=vals["crowd_radius"],
        safe_bubble=vals["safe_bubble"],
        w_close=vals["w_close"],
        w_dist=vals["w_dist"],
        thrust_back_high_left=tbhL,
        thrust_back_high_peak=tbhP,
        thrust_back_high_right=tbhR,
        thrust_back_med_left=tbmL,
        thrust_back_med_peak=tbmP,
        thrust_back_med_right=tbmR,
        thrust_zero_left=tzL,
        thrust_zero_peak=tzP,
        thrust_zero_right=tzR,
        thrust_fwd_low_left=tflL,
        thrust_fwd_low_peak=tflP,
        thrust_fwd_low_right=tflR,
        thrust_fwd_med_left=tfmL,
        thrust_fwd_med_peak=tfmP,
        thrust_fwd_med_right=tfmR,
        thrust_fwd_high_left=tfhL,
        thrust_fwd_high_peak=tfhP,
        thrust_fwd_high_right=tfhR,
        panic_count=int(round(vals["panic_count"])),
        reverse_speed_thresh=vals["reverse_speed_thresh"],
        bullet_speed=vals["bullet_speed"],
        aim_theta_cap=math.radians(vals["aim_theta_cap_deg"]),
        aim_dt_lead=vals["aim_dt_lead"],
    )


# ----------------------------
# Fitness
# ----------------------------
def evaluate(vec, seeds=(1, 2, 3)):
    """
    Run several scenarios with a controller that uses params from vec.
    Return a scalar fitness (higher = better).
    """
    params = vec_to_params(vec)
    ctrlr = HybridHoverAimbot(params)
    game = KesslerGame()
    scenarios = get_eval_scenarios()

    total_score = 0.0
    total_acc = 0.0
    total_survival = 0.0
    total_deaths = 0
    total_abs_thrust = 0.0

    for seed in seeds:
        for scn in scenarios:
            # Normalize scenario to a concrete object (some callables accept a seed)
            try:
                scenario = scn(seed) if callable(scn) else scn
            except TypeError:
                scenario = scn

            # Run game
            score, perf = game.run(scenario=scenario, controllers=[ctrlr])

            # Aggregate metrics (keys vary by engine; use fallbacks)
            total_score += float(score)

            acc = float(perf.get("accuracy", perf.get("hit_rate", 0.0)))
            total_acc += acc

            survival = float(perf.get("survival_time", perf.get("duration", 0.0)))
            total_survival += survival

            deaths = int(perf.get("deaths", perf.get("collisions", 0)))
            total_deaths += deaths

            total_abs_thrust += float(perf.get("abs_thrust_sum", 0.0))

    n = max(1, len(seeds) * len(scenarios))
    avg_acc = total_acc / n
    avg_survival = total_survival / n

    # Fitness: tweak weights as needed for your rubric
    fitness = (
        total_score
        + 50.0 * avg_acc
        + 0.05 * avg_survival
        - 200.0 * total_deaths
        - 0.001 * total_abs_thrust
    )
    return fitness


# ----------------------------
# GA Core
# ----------------------------
def rand_vec():
    return np.array([random.uniform(lo, hi) for _, lo, hi in GENES], dtype=float)


def mutate(vec, sigma=0.1):
    out = vec.copy()
    for i, (_name, lo, hi) in enumerate(GENES):
        if random.random() < 0.3:
            span = hi - lo
            out[i] += random.gauss(0.0, sigma * span)
            out[i] = min(hi, max(lo, out[i]))
    return out


def crossover(a, b):
    # Blend crossover (BLX-alpha with alpha=0.3)
    child = a.copy()
    alpha = 0.3
    for i, (_name, lo, hi) in enumerate(GENES):
        low = min(a[i], b[i])
        high = max(a[i], b[i])
        span = high - low
        c_lo = max(lo, low - alpha * span)
        c_hi = min(hi, high + alpha * span)
        child[i] = random.uniform(c_lo, c_hi)
    return child


def select_parents(pop, fits, k):
    # Tournament selection
    parents = []
    for _ in range(k):
        i = random.randrange(len(pop))
        j = random.randrange(len(pop))
        winner = i if fits[i] >= fits[j] else j
        parents.append(pop[winner])
    return parents


def main():
    random.seed(1337)
    np.random.seed(1337)

    POP = 16
    GENS = 12

    pop = [rand_vec() for _ in range(POP)]

    for gen in range(GENS):
        fits = [evaluate(ind) for ind in pop]
        best_idx = int(np.argmax(fits))
        print(f"[Gen {gen}] best fitness = {fits[best_idx]:.2f}")

        # Elitism: keep top 2
        order = np.argsort(fits)[::-1]
        elite = [pop[order[0]].copy(), pop[order[1]].copy()]

        # Parent pool
        parents = select_parents(pop, fits, k=POP)

        # Produce children
        children = elite[:]  # start with elites
        while len(children) < POP:
            a = parents[random.randrange(len(parents))]
            b = parents[random.randrange(len(parents))]
            c = crossover(a, b)
            c = mutate(c, sigma=0.08)
            children.append(c)

        pop = children

    # Final best
    fits = [evaluate(ind) for ind in pop]
    best_idx = int(np.argmax(fits))
    best_vec = pop[best_idx]
    best_fit = fits[best_idx]

    print("\n=== GA COMPLETE ===")
    print(f"Best fitness: {best_fit:.2f}")
    print("Best parameter vector (paste into your code if desired):")
    for (name, lo, hi), val in zip(GENES, best_vec):
        print(f"{name} = {val:.6f}")

    # Also print a ready-to-use HybridParams constructor
    P = vec_to_params(best_vec)
    print("\nHybridParams(**{")
    for attr, val in P.__dict__.items():
        print(f"    '{attr}': {repr(val)},")
    print("})")


if __name__ == "__main__":
    main()
