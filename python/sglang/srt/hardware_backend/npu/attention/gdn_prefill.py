"""Select an available NPU GDN prefill implementation without losing checkpoints."""


def run_gdn_prefill(
    native_chunk,
    dispatcher,
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
):
    if native_chunk is not None:
        return native_chunk(
            q, k, v, g, beta, ssm_states[cache_indices], query_start_loc
        )
    # The dispatcher includes the NPU Triton implementation shipped by older
    # CANN images. It returns the same final-state and per-chunk checkpoints.
    return dispatcher.extend(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g,
        beta=beta,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )
