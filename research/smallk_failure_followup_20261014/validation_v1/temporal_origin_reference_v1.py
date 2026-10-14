"""Source-only alternative: subtract the response at the actual forecast origin.

No asset loader, model loader, training driver, or execution entry point.
The frozen paired-reference RF3 module and live experiments are unchanged.
"""


def origin_reference_residual(head, past_z, predicted_future_z):
    """Use the same causal RF3 response, anchored at observed time t."""
    if past_z.ndim != 3 or predicted_future_z.ndim != 3:
        raise ValueError('LATENTS_MUST_BE_BATCH_TIME_K')
    if past_z.shape[0] != predicted_future_z.shape[0]:
        raise ValueError('LATENT_BATCH_MISMATCH')
    if past_z.shape[1] < 3 or predicted_future_z.shape[1] < 1:
        raise ValueError('THREE_ACTUAL_PAST_POINTS_AND_NONEMPTY_FORECAST_REQUIRED')
    if past_z.shape[2] != head.latent_channels or predicted_future_z.shape[2] != head.latent_channels:
        raise ValueError('LATENT_K_MISMATCH')
    future_response = head.response(past_z, predicted_future_z)
    origin_response = head.response(past_z[:, -3:-1], past_z[:, -1:])
    return future_response - origin_response
