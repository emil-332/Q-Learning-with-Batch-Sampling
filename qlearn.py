import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

ENV_FROZENLAKE = "frozenlake"
ENV_CLIFF = "cliff"
ENV_TAXI = "taxi"

GYM_ENV_BY_KIND = {
    ENV_FROZENLAKE: "FrozenLake-v1",
    ENV_CLIFF: "CliffWalking-v1",
    ENV_TAXI: "Taxi-v3",
}

DEFAULT_MAP_NAME = "4x4"
DEFAULT_SLIPPERY = True
DEFAULT_SEED = 0
DEFAULT_MAX_EPISODE_STEPS = 200
DEFAULT_DP_MAX_EPISODE_STEPS = 10000

DEFAULT_STEPS = 20000
DEFAULT_BATCH_SIZE = 1
DEFAULT_SAMPLES_PER_PAIR = 1
DEFAULT_GAMMA = 0.99
DEFAULT_ALPHA = 0.1
DEFAULT_ALPHA_BETA = 1.0
DEFAULT_ALPHA_TAU = 200000

ALPHA_MODE_CONST = "const"
ALPHA_MODE_GLOBAL = "global"
ALPHA_MODE_SA = "sa"
ALPHA_MODE_SQL_LINEAR = "sql_linear"

DEFAULT_QL_ALPHA_MODE = ALPHA_MODE_SA
DEFAULT_SQL_ALPHA_MODE = ALPHA_MODE_SQL_LINEAR

DEFAULT_DP_TOL = 1e-12
DEFAULT_DP_MAX_ITER = 200000
DEFAULT_USE_EXACT_BELLMAN = False

MIN_BATCH_SIZE = 1
MIN_SAMPLES_PER_PAIR = 1
MIN_POSITIVE_VALUE = 0.0
NORM_EPSILON = 1e-12
UNIT_OFFSET = 1.0


def make_env(
    kind=ENV_FROZENLAKE,
    map_name=DEFAULT_MAP_NAME,
    slippery=DEFAULT_SLIPPERY,
    seed=DEFAULT_SEED,
    max_ep_steps=DEFAULT_MAX_EPISODE_STEPS,
):
    if kind not in GYM_ENV_BY_KIND:
        raise ValueError(f"unknown env: {kind}")

    if kind == ENV_FROZENLAKE:
        env = gym.make(GYM_ENV_BY_KIND[kind], map_name=map_name, is_slippery=slippery)
    else:
        env = gym.make(GYM_ENV_BY_KIND[kind])

    env = TimeLimit(env, max_episode_steps=int(max_ep_steps))
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    return env


def _get_transition_model(env):
    if not hasattr(env.unwrapped, "P"):
        raise ValueError("environment does not expose a tabular transition model")
    return env.unwrapped.P


def bellman_optimality_operator_from_model(P_env, Q, gamma=DEFAULT_GAMMA):
    nS = len(P_env)
    nA = len(P_env[0])
    TQ = np.zeros((nS, nA), dtype=np.float64)
    V = np.max(Q, axis=1)

    for s in range(nS):
        for a in range(nA):
            value = 0.0
            for prob, s2, reward, done in P_env[s][a]:
                if done:
                    value += prob * reward
                else:
                    value += prob * (reward + gamma * V[s2])
            TQ[s, a] = value

    return TQ


def sample_transition_from_model(P_env, s, a, rng):
    transitions = P_env[s][a]
    probabilities = np.asarray([prob for prob, _, _, _ in transitions], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    index = int(rng.choice(len(transitions), p=probabilities))
    _, s2, reward, done = transitions[index]
    return float(reward), int(s2), bool(done)


def sample_bellman_target_from_model(P_env, s, a, Q, gamma, rng):
    reward, s2, done = sample_transition_from_model(P_env, s, a, rng)
    if done:
        return reward
    return float(reward + gamma * np.max(Q[s2]))


def compute_Q_star_stationary_dp(
    kind=ENV_FROZENLAKE,
    map_name=DEFAULT_MAP_NAME,
    slippery=DEFAULT_SLIPPERY,
    gamma=DEFAULT_GAMMA,
    tol=DEFAULT_DP_TOL,
    max_iter=DEFAULT_DP_MAX_ITER,
    seed=DEFAULT_SEED,
    max_ep_steps=DEFAULT_DP_MAX_EPISODE_STEPS,
):
    env = make_env(
        kind=kind,
        map_name=map_name,
        slippery=slippery,
        seed=seed,
        max_ep_steps=max_ep_steps,
    )
    P_env = _get_transition_model(env)
    nS = env.observation_space.n
    nA = env.action_space.n
    env.close()

    Q = np.zeros((nS, nA), dtype=np.float64)

    for _ in range(int(max_iter)):
        Q_next = bellman_optimality_operator_from_model(P_env, Q, gamma=gamma)
        if np.max(np.abs(Q_next - Q)) < tol:
            return Q_next
        Q = Q_next

    return Q


class StabilityLogger:
    def __init__(self, Q_star=None):
        self.Q_star = Q_star
        self.dist_to_Qstar = []
        self.angle = []
        self.noise_var = []
        self.m_eff = []
        self.pg_norm2 = []
        self.pg_norm_inf = []
        self.pg_values = []

    def _reference(self, t_in_ep=None):
        if self.Q_star is None:
            return None
        if getattr(self.Q_star, "ndim", 0) == 2:
            return self.Q_star
        if getattr(self.Q_star, "ndim", 0) == 3:
            t = 0 if t_in_ep is None else int(t_in_ep)
            t = max(0, min(t, self.Q_star.shape[0] - 1))
            return self.Q_star[t]
        return None

    def log_step(self, Q, dQ, batch_var=None, m_eff=None, t_in_ep=None):
        reference = self._reference(t_in_ep=t_in_ep)

        self.noise_var.append(float(batch_var) if batch_var is not None else 0.0)
        self.m_eff.append(int(m_eff) if m_eff is not None else 0)
        self.pg_norm2.append(float(np.linalg.norm(dQ)))
        self.pg_norm_inf.append(float(np.max(np.abs(dQ))))
        self.pg_values.append(dQ.copy())

        if reference is None:
            self.dist_to_Qstar.append(np.nan)
            self.angle.append(np.nan)
            return

        self.dist_to_Qstar.append(float(np.linalg.norm(Q - reference)))

        gradient = (reference - Q).ravel()
        direction = dQ.ravel()
        gradient_norm = np.linalg.norm(gradient)
        direction_norm = np.linalg.norm(direction)

        if gradient_norm < NORM_EPSILON or direction_norm < NORM_EPSILON:
            self.angle.append(0.0)
            return

        cosine = float(np.dot(direction, gradient) / (gradient_norm * direction_norm))
        self.angle.append(float(np.clip(cosine, -UNIT_OFFSET, UNIT_OFFSET)))

    def export(self):
        return {
            "dist_to_Qstar": np.asarray(self.dist_to_Qstar, dtype=float),
            "angle": np.asarray(self.angle, dtype=float),
            "noise_var": np.asarray(self.noise_var, dtype=float),
            "m_eff": np.asarray(self.m_eff, dtype=int),
            "pg_norm2": np.asarray(self.pg_norm2, dtype=float),
            "pg_norm_inf": np.asarray(self.pg_norm_inf, dtype=float),
            "pg_values": np.asarray(self.pg_values, dtype=object),
        }


def _step_size(alpha_mode, alpha, alpha_beta, alpha_tau, step_index, count):
    if alpha_mode == ALPHA_MODE_SA:
        value = float(alpha) / (float(count) ** float(alpha_beta))
    elif alpha_mode == ALPHA_MODE_GLOBAL:
        value = float(alpha) / (UNIT_OFFSET + float(step_index) / float(alpha_tau))
    elif alpha_mode == ALPHA_MODE_CONST:
        value = float(alpha)
    elif alpha_mode == ALPHA_MODE_SQL_LINEAR:
        value = UNIT_OFFSET / (float(step_index) + UNIT_OFFSET)
    else:
        raise ValueError(f"unknown alpha_mode: {alpha_mode}")

    if value <= MIN_POSITIVE_VALUE:
        raise ValueError(f"non-positive step size: {value}")

    return value


def train_batch(
    env,
    *,
    steps=DEFAULT_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    samples_per_pair=DEFAULT_SAMPLES_PER_PAIR,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
    seed=DEFAULT_SEED,
    Q_star=None,
    alpha_mode=DEFAULT_QL_ALPHA_MODE,
    alpha_beta=DEFAULT_ALPHA_BETA,
    alpha_tau=DEFAULT_ALPHA_TAU,
    use_exact_bellman=DEFAULT_USE_EXACT_BELLMAN,
):
    nS = env.observation_space.n
    nA = env.action_space.n
    d = nS * nA

    if batch_size < MIN_BATCH_SIZE or batch_size > d:
        raise ValueError(f"batch_size must be in [{MIN_BATCH_SIZE}, {d}], got {batch_size}")

    if samples_per_pair < MIN_SAMPLES_PER_PAIR:
        raise ValueError(f"samples_per_pair must be >= {MIN_SAMPLES_PER_PAIR}")

    P_env = _get_transition_model(env)
    Q = np.zeros((nS, nA), dtype=np.float64)
    N_sa = np.zeros((nS, nA), dtype=np.int64)
    rng = np.random.default_rng(seed)
    logger = StabilityLogger(Q_star=Q_star)
    returns = np.full(int(steps), np.nan, dtype=np.float64)
    all_indices = np.arange(d)

    for step in range(int(steps)):
        selected = rng.choice(all_indices, size=int(batch_size), replace=False)
        coordinates = [(int(index // nA), int(index % nA)) for index in selected]

        if use_exact_bellman:
            BQ = bellman_optimality_operator_from_model(P_env, Q, gamma=gamma)

        dQ = np.zeros_like(Q)
        empirical_targets = []

        for s, a in coordinates:
            if use_exact_bellman:
                target = float(BQ[s, a])
                empirical_targets.append(target)
            else:
                samples = [
                    sample_bellman_target_from_model(P_env, s, a, Q, gamma, rng)
                    for _ in range(int(samples_per_pair))
                ]
                target = float(np.mean(samples))
                empirical_targets.extend(samples)

            if alpha_mode == ALPHA_MODE_SA:
                N_sa[s, a] += 1
                step_size = _step_size(
                    alpha_mode=alpha_mode,
                    alpha=alpha,
                    alpha_beta=alpha_beta,
                    alpha_tau=alpha_tau,
                    step_index=step,
                    count=N_sa[s, a],
                )
            else:
                step_size = _step_size(
                    alpha_mode=alpha_mode,
                    alpha=alpha,
                    alpha_beta=alpha_beta,
                    alpha_tau=alpha_tau,
                    step_index=step,
                    count=MIN_BATCH_SIZE,
                )

            dQ[s, a] = step_size * (target - Q[s, a])

        Q += dQ

        batch_var = 0.0 if use_exact_bellman or len(empirical_targets) == 0 else float(np.var(empirical_targets))
        logger.log_step(Q, dQ, batch_var=batch_var, m_eff=batch_size, t_in_ep=None)

    return Q, returns, logger.export()


def train_speedy_batch(
    env,
    *,
    steps=DEFAULT_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    samples_per_pair=DEFAULT_SAMPLES_PER_PAIR,
    gamma=DEFAULT_GAMMA,
    seed=DEFAULT_SEED,
    Q_star=None,
    alpha_mode=DEFAULT_SQL_ALPHA_MODE,
    alpha=DEFAULT_ALPHA,
    alpha_beta=DEFAULT_ALPHA_BETA,
    alpha_tau=DEFAULT_ALPHA_TAU,
    use_exact_bellman=DEFAULT_USE_EXACT_BELLMAN,
):
    nS = env.observation_space.n
    nA = env.action_space.n
    d = nS * nA

    if batch_size < MIN_BATCH_SIZE or batch_size > d:
        raise ValueError(f"batch_size must be in [{MIN_BATCH_SIZE}, {d}], got {batch_size}")

    if samples_per_pair < MIN_SAMPLES_PER_PAIR:
        raise ValueError(f"samples_per_pair must be >= {MIN_SAMPLES_PER_PAIR}")

    P_env = _get_transition_model(env)
    rng = np.random.default_rng(seed)

    Q_prev = np.zeros((nS, nA), dtype=np.float64)
    Q_curr = np.zeros((nS, nA), dtype=np.float64)
    N_sa = np.zeros((nS, nA), dtype=np.int64)

    logger = StabilityLogger(Q_star=Q_star)
    returns = np.full(int(steps), np.nan, dtype=np.float64)
    all_indices = np.arange(d)

    for step in range(int(steps)):
        selected = rng.choice(all_indices, size=int(batch_size), replace=False)
        coordinates = [(int(index // nA), int(index % nA)) for index in selected]

        if use_exact_bellman:
            B_prev = bellman_optimality_operator_from_model(P_env, Q_prev, gamma=gamma)
            B_curr = bellman_optimality_operator_from_model(P_env, Q_curr, gamma=gamma)

        Q_next = Q_curr.copy()
        dQ = np.zeros_like(Q_curr)
        empirical_targets = []

        for s, a in coordinates:
            if use_exact_bellman:
                target_prev = float(B_prev[s, a])
                target_curr = float(B_curr[s, a])
            else:
                previous_values = []
                current_values = []

                for _ in range(int(samples_per_pair)):
                    reward, s2, done = sample_transition_from_model(P_env, s, a, rng)

                    if done:
                        previous_target = reward
                        current_target = reward
                    else:
                        previous_target = reward + gamma * np.max(Q_prev[s2])
                        current_target = reward + gamma * np.max(Q_curr[s2])

                    previous_values.append(previous_target)
                    current_values.append(current_target)

                target_prev = float(np.mean(previous_values))
                target_curr = float(np.mean(current_values))
                empirical_targets.extend(current_values)

            if alpha_mode == ALPHA_MODE_SA:
                N_sa[s, a] += 1
                step_size = _step_size(
                    alpha_mode=alpha_mode,
                    alpha=alpha,
                    alpha_beta=alpha_beta,
                    alpha_tau=alpha_tau,
                    step_index=step,
                    count=N_sa[s, a],
                )
            else:
                step_size = _step_size(
                    alpha_mode=alpha_mode,
                    alpha=alpha,
                    alpha_beta=alpha_beta,
                    alpha_tau=alpha_tau,
                    step_index=step,
                    count=MIN_BATCH_SIZE,
                )

            delta = step_size * (target_prev - Q_curr[s, a]) + (UNIT_OFFSET - step_size) * (target_curr - target_prev)
            dQ[s, a] = delta
            Q_next[s, a] += delta

        batch_var = 0.0 if use_exact_bellman or len(empirical_targets) == 0 else float(np.var(empirical_targets))
        logger.log_step(Q_next, dQ, batch_var=batch_var, m_eff=batch_size, t_in_ep=None)

        Q_prev = Q_curr
        Q_curr = Q_next

    return Q_curr, returns, logger.export()