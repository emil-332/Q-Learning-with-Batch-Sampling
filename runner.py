import os
import numpy as np

from qlearn import (
    ENV_FROZENLAKE,
    DEFAULT_ALPHA,
    DEFAULT_ALPHA_BETA,
    DEFAULT_ALPHA_TAU,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DP_MAX_ITER,
    DEFAULT_DP_TOL,
    DEFAULT_GAMMA,
    DEFAULT_MAP_NAME,
    DEFAULT_QL_ALPHA_MODE,
    DEFAULT_SEED,
    DEFAULT_SLIPPERY,
    DEFAULT_SAMPLES_PER_PAIR,
    DEFAULT_SQL_ALPHA_MODE,
    DEFAULT_STEPS,
    DEFAULT_USE_EXACT_BELLMAN,
    compute_Q_star_stationary_dp,
    make_env,
    train_speedy_batch,
    train_batch,
)

DEFAULT_OUT_ROOT = "runs_empirical"


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_run(path, Q, returns, log_dict):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(
        path,
        Q=np.asarray(Q),
        returns=np.asarray(returns),
        dist_to_Qstar=np.asarray(log_dict["dist_to_Qstar"]),
        angle=np.asarray(log_dict["angle"]),
        noise_var=np.asarray(log_dict["noise_var"]),
        m_eff=np.asarray(log_dict["m_eff"]),
        pg_norm2=np.asarray(log_dict["pg_norm2"]),
        pg_norm_inf=np.asarray(log_dict["pg_norm_inf"]),
        pg_values=np.asarray(log_dict["pg_values"], dtype=object),
    )


def _gamma_tag(gamma):
    return str(gamma).replace(".", "p")


def _env_tag(env_name, map_name):
    if env_name == ENV_FROZENLAKE:
        return f"{env_name}_{map_name}"
    return env_name


def run_compute_Qstar_stationary_dp(
    env_name,
    *,
    map_name=DEFAULT_MAP_NAME,
    slippery=DEFAULT_SLIPPERY,
    gamma=DEFAULT_GAMMA,
    seed=DEFAULT_SEED,
    tol=DEFAULT_DP_TOL,
    max_iter=DEFAULT_DP_MAX_ITER,
    out_root=DEFAULT_OUT_ROOT,
):
    ensure_dir(out_root)

    Q_star = compute_Q_star_stationary_dp(
        kind=env_name,
        map_name=map_name,
        slippery=slippery,
        gamma=gamma,
        tol=tol,
        max_iter=max_iter,
        seed=seed,
    )

    tag = f"Qstar_dp_{_env_tag(env_name, map_name)}_gamma{_gamma_tag(gamma)}"
    save_path = os.path.join(out_root, f"{tag}.npy")
    np.save(save_path, Q_star)

    return tag, Q_star


def run_uniform_batch(
    env_name,
    *,
    map_name=DEFAULT_MAP_NAME,
    slippery=DEFAULT_SLIPPERY,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    samples_per_pair=DEFAULT_SAMPLES_PER_PAIR,
    alpha=DEFAULT_ALPHA,
    gamma=DEFAULT_GAMMA,
    Q_star=None,
    alpha_mode=DEFAULT_QL_ALPHA_MODE,
    alpha_beta=DEFAULT_ALPHA_BETA,
    alpha_tau=DEFAULT_ALPHA_TAU,
    use_exact_bellman=DEFAULT_USE_EXACT_BELLMAN,
    out_root=DEFAULT_OUT_ROOT,
):
    env = make_env(env_name, map_name, slippery, seed)

    Q, returns, log = train_batch(
        env,
        steps=steps,
        batch_size=batch_size,
        samples_per_pair=samples_per_pair,
        alpha=alpha,
        gamma=gamma,
        seed=seed,
        Q_star=Q_star,
        alpha_mode=alpha_mode,
        alpha_beta=alpha_beta,
        alpha_tau=alpha_tau,
        use_exact_bellman=use_exact_bellman,
    )
    env.close()

    ensure_dir(out_root)

    mode_tag = "exact" if use_exact_bellman else f"K{samples_per_pair}"
    tag = f"ql_{_env_tag(env_name, map_name)}_m{batch_size}_{mode_tag}_gamma{_gamma_tag(gamma)}_seed{seed}"
    save_path = os.path.join(out_root, f"{tag}.npz")

    save_run(save_path, Q, returns, log)
    return tag, Q, returns, log


def run_speedy_uniform(
    env_name,
    *,
    map_name=DEFAULT_MAP_NAME,
    slippery=DEFAULT_SLIPPERY,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
    samples_per_pair=DEFAULT_SAMPLES_PER_PAIR,
    gamma=DEFAULT_GAMMA,
    Q_star=None,
    alpha_mode=DEFAULT_SQL_ALPHA_MODE,
    alpha=DEFAULT_ALPHA,
    alpha_beta=DEFAULT_ALPHA_BETA,
    alpha_tau=DEFAULT_ALPHA_TAU,
    use_exact_bellman=DEFAULT_USE_EXACT_BELLMAN,
    out_root=DEFAULT_OUT_ROOT,
):
    env = make_env(env_name, map_name, slippery, seed)

    Q, returns, log = train_speedy_batch(
        env,
        steps=steps,
        batch_size=batch_size,
        samples_per_pair=samples_per_pair,
        gamma=gamma,
        seed=seed,
        Q_star=Q_star,
        alpha_mode=alpha_mode,
        alpha=alpha,
        alpha_beta=alpha_beta,
        alpha_tau=alpha_tau,
        use_exact_bellman=use_exact_bellman,
    )
    env.close()

    ensure_dir(out_root)

    mode_tag = "exact" if use_exact_bellman else f"K{samples_per_pair}"
    alpha_tag = alpha_mode.replace("_", "")
    tag = f"sql_{_env_tag(env_name, map_name)}_m{batch_size}_{mode_tag}_a{alpha_tag}_gamma{_gamma_tag(gamma)}_seed{seed}"
    save_path = os.path.join(out_root, f"{tag}.npz")

    save_run(save_path, Q, returns, log)
    return tag, Q, returns, log