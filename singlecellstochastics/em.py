"""EM-style alternation between E and M steps for the LAVOUS ELBO."""

from .optimize import Lq_optimize_torch_OU


def run_em(
    init_params,
    mode,
    x_tensor_list,
    gene_names,
    diverge_list_torch,
    share_list_torch,
    epochs_list_torch,
    beta_list_torch,
    max_iter,
    learning_rate,
    device,
    wandb_flag,
    window,
    tol,
    approx,
    em_iter,
    prior,
    kkt,
    nb,
    library_list_tensor,
    const,
    root_mode="stationary",
):
    """Alternate ELBO E-steps (variational params) and M-steps (OU params).

    Returns (params, loss) using the same layout as ``Lq_optimize_torch_OU``.
    """

    def _step(params, em_phase):
        return Lq_optimize_torch_OU(
            params,
            mode,
            x_tensor_list,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_phase,
            prior,
            kkt,
            nb,
            library_list_tensor,
            const,
            root_mode=root_mode,
        )

    params = init_params
    for _ in range(em_iter):
        params, _ = _step(params, "e")
        params, _ = _step(params, "m")

    # Final E-step so the returned variational beliefs are consistent with
    # the last M-step OU parameters.
    params, loss = _step(params, "e")
    params = [p.clone().detach() for p in params]
    return params, loss.clone().detach()
