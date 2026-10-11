"""Small CPU implementation diagnostics, distinct from actual-data GT fits."""

from __future__ import annotations

import time

START_WALL = time.perf_counter()
START_CPU = time.process_time()

import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import psutil
import torch

from models_gt import (
    GTConvAE, ProductGraphShift, PolynomialGTFilter,
    temporal_downsample, temporal_upsample,
)


HERE = Path(__file__).resolve().parent


def dense_shift(graph: torch.Tensor, coefficients: torch.Tensor, length: int) -> torch.Tensor:
    identity_t = torch.eye(length, dtype=graph.dtype)
    identity_c = torch.eye(graph.shape[0], dtype=graph.dtype)
    temporal = torch.diag(torch.ones(length - 1, dtype=graph.dtype), diagonal=-1)
    return (
        coefficients[0] * torch.kron(identity_t, identity_c)
        + coefficients[1] * torch.kron(identity_t, graph)
        + coefficients[2] * torch.kron(temporal, identity_c)
        + coefficients[3] * torch.kron(temporal, graph)
    )


def dense_filter(x: torch.Tensor, graph: torch.Tensor, coefficients: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    product = dense_shift(graph, coefficients, x.shape[1])
    current = x.reshape(x.shape[0], -1, x.shape[-1])
    out = current @ weight[0]
    for degree in range(1, weight.shape[0]):
        current = torch.einsum("dc,bcf->bdf", product, current)
        out = out + current @ weight[degree]
    return out.reshape(x.shape[0], x.shape[1], x.shape[2], weight.shape[-1])


def set_temporal_only(product: ProductGraphShift):
    with torch.no_grad():
        product.s00.zero_()
        product.s10.fill_(1)
        product.s01.zero_()
        product.s11.zero_()


def interpolation_fixture() -> GTConvAE:
    model = GTConvAE(torch.zeros(2, 2)).double()
    set_temporal_only(model.product)
    with torch.no_grad():
        for layer in model.decoder:
            layer.weight.zero_()
            layer.weight[0, 0, 0] = 1
            layer.weight[1, 0, 0] = 1
    return model


def verify() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(92601)
    checks = []

    def check(name: str, condition: bool, **evidence):
        checks.append({"name": name, "status": "PASS" if condition else "FAIL", **evidence})
        if not condition:
            raise AssertionError(name)

    graph = torch.tensor([[0., .25, .4], [.1, 0., .3], [.5, .2, 0.]], dtype=torch.float64)
    product = ProductGraphShift(graph).double()
    with torch.no_grad():
        product.s00.fill_(.11)
        product.s01.fill_(.31)
        product.s10.fill_(.29)
        product.s11.fill_(-.07)
        product.graph.copy_(graph)
    layer = PolynomialGTFilter(2, 3).double()
    x = torch.randn(2, 6, 3, 2, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    coefficients = product.coefficients.detach().clone().requires_grad_()
    reference_weight = layer.weight.detach().clone().requires_grad_()
    actual = layer(x, product)
    reference = dense_filter(reference_x, graph, coefficients, reference_weight)
    error = float((actual - reference).abs().max())
    check("dense_Kronecker_degree4_output", error < 1e-10, max_abs_error=error,
          dtype="float64", shape=list(actual.shape), taps=5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (reference * probe).sum().backward()
    gradients = {
        "input": float((x.grad - reference_x.grad).abs().max()),
        "filter": float((layer.weight.grad - reference_weight.grad).abs().max()),
        "product": float((torch.stack([product.s00.grad, product.s01.grad,
                                        product.s10.grad, product.s11.grad])
                          - coefficients.grad).abs().max()),
    }
    check("dense_Kronecker_input_filter_product_gradients",
          max(gradients.values()) < 1e-10, max_abs_errors=gradients)

    impulse = torch.zeros(1, 6, 3, 1, dtype=torch.float64)
    impulse[0, 2, 0, 0] = 1
    set_temporal_only(product)
    shifted = product(impulse)
    check("temporal_delta_moves_from_t2_to_t3", float(shifted[0, 3, 0, 0]) == 1
          and int(torch.count_nonzero(shifted)) == 1)
    impulse.zero_()
    impulse[0, -1, 0, 0] = 1
    check("temporal_no_circular_wrap", int(torch.count_nonzero(product(impulse))) == 0)

    ramp = torch.arange(16.).reshape(1, 16, 1, 1)
    pooled = temporal_downsample(temporal_downsample(ramp))
    check("max_pooling_four_timestamp_right_edge_phase", torch.equal(
        pooled.flatten(), torch.tensor([3., 7., 11., 15.])), representative_indices=pooled.flatten().tolist())
    pool_indices = []
    for index in range(16):
        impulse = torch.zeros_like(ramp)
        impulse[:, index] = 1
        pooled = temporal_downsample(temporal_downsample(impulse))
        active = int(torch.nonzero(pooled.flatten()).item())
        pool_indices.append(active)
    check("pooling_delta_all_positions", pool_indices == [i // 4 for i in range(16)],
          latent_index_by_original_index=pool_indices)
    up = temporal_upsample(temporal_upsample(torch.ones(1, 4, 1, 1)))
    check("Eq8_right_edge_zero_insertion", torch.equal(
        torch.nonzero(up.flatten()).flatten(), torch.tensor([3, 7, 11, 15])))

    model = interpolation_fixture()
    latent = torch.ones(1, 12, 2, 2, dtype=torch.float64)
    decoded = model.decode(latent)
    check("decoder_length12_to48", decoded.shape == (1, 48, 2))
    check("documented_three_zero_startup_steps", int(torch.count_nonzero(decoded[:, :3])) == 0
          and torch.equal(decoded[:, 3:], torch.ones_like(decoded[:, 3:])))
    check("final_upsampling_does_not_leave_alternating_fixed_zeros",
          bool((decoded[:, 3:] != 0).all()), nonzero_count_by_phase=[
              int(torch.count_nonzero(decoded[:, phase::4])) for phase in range(4)])
    history = torch.ones(1, 8, 2, 2, dtype=torch.float64)
    future = model.decode_future(history, latent, horizon=48)
    check("constant_history_boundary_and_all_future_phases", torch.equal(
        future, torch.ones_like(future)), max_abs_error=float((future - 1).abs().max()))
    impulse = torch.zeros(1, 6, 2, 2, dtype=torch.float64)
    impulse[0, 2, 0, 0] = 1
    decoded = model.decode(impulse)
    active = torch.nonzero(decoded[0, :, 0]).flatten().tolist()
    check("decoder_delta_Eq8_phase_explicit", active == [11, 12, 13, 14],
          active_physical_indices=active, latent_tick=2,
          note="Right-edge injection plus past-directed interpolation; no claim of a zero-phase inverse")
    check("nonmultiple_horizon_predict_extra_then_crop", model.decode_future(
        history, torch.ones(1, 13, 2, 2, dtype=torch.float64), horizon=49).shape == (1, 49, 2)
        and GTConvAE.latent_forecast_steps(49) == 13)

    full = GTConvAE(torch.tensor([[0., 1.], [1., 0.]])).double()
    no_spatial = GTConvAE(torch.tensor([[0., 1.], [1., 0.]]), no_spatial=True).double()
    source = torch.zeros(1, 8, 2, 1, dtype=torch.float64)
    source[0, 2, 0, 0] = 1
    with torch.no_grad():
        full.product.s00.zero_()
        full.product.s10.zero_()
        full.product.s01.fill_(1)
        full.product.s11.zero_()
    spatial_output = full.product(source)
    check("spatial_edge_routes_source_to_adjacent_node", float(spatial_output[0, 2, 1, 0]) == 1)
    with torch.no_grad():
        full.product.graph.zero_()
    check("graph_change_alters_spatial_path", not torch.equal(full.product(source), spatial_output))
    no_spatial_output = no_spatial.product(source)
    with torch.no_grad():
        no_spatial.product.graph.zero_()
    check("no_spatial_output_invariant_to_graph", torch.equal(no_spatial.product(source), no_spatial_output))
    check("no_spatial_s01_s11_literal_zero_and_frozen",
          no_spatial.product.s01.item() == no_spatial.product.s11.item() == 0
          and not no_spatial.product.s01.requires_grad and not no_spatial.product.s11.requires_grad
          and set(dict(no_spatial.product.named_parameters())) == {"s00", "s10"})

    torch.manual_seed(92601)
    model = GTConvAE(torch.tensor([[0., .8, .2], [.8, 0., .3], [.2, .3, 0.]])).double()
    ticks = torch.arange(64, dtype=torch.float64)
    waves = torch.stack([torch.sin(ticks / 15), torch.cos(ticks / 19),
                         torch.sin(ticks / 15) - .4 * torch.cos(ticks / 19)], dim=-1)
    tiny = torch.stack([waves, .85 * waves + .15])
    initial_weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial_mse = float((model(tiny) - tiny).square().mean().detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    curve = []
    first_gradient_report = None
    for step in range(600):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(tiny)
        loss = (reconstructed - tiny).square().mean()
        loss.backward()
        if step == 0:
            first_gradient_report = {name: {"gradient_exists": value.grad is not None,
                "nonzero_elements": int(torch.count_nonzero(value.grad)) if value.grad is not None else 0,
                "elements": value.numel()} for name, value in model.named_parameters()}
        optimizer.step()
        if step % 50 == 0 or step == 599:
            curve.append({"updates": step + 1, "full_mse_before_update": float(loss.detach())})
    final = model(tiny).detach()
    final_mse = float((final - tiny).square().mean())
    changed = {name: float((value.detach() - initial_weights[name]).abs().max())
               for name, value in model.named_parameters()}
    check("encoder_decoder_product_gradients_exist_and_nonzero",
          all(item["gradient_exists"] and item["nonzero_elements"] > 0
              for item in first_gradient_report.values()), gradients=first_gradient_report)
    check("encoder_decoder_product_weights_actual_update", all(value > 0 for value in changed.values()),
          max_abs_update_by_parameter=changed)
    check("tiny_signed_batch_actual_600update_overfit", final_mse < initial_mse * .2,
          initial_full_mse=initial_mse, final_full_mse=final_mse,
          final_mse_after_documented_startup=float((final[:, 3:] - tiny[:, 3:]).square().mean()),
          updates=600, lr=.01, rho=0, implementation_diagnostic_only=True, curve=curve)
    check("linear_final_decoder_can_reconstruct_negative_values", bool((final < -.05).any()))

    old_encoder_state = {name: value.detach().clone() for name, value in model.named_parameters()
                         if name.startswith("encoder.") or name.startswith("product.")}
    old_decoder_state = {name: value.detach().clone() for name, value in model.named_parameters()
                         if name.startswith("decoder.")}
    model.zero_grad(set_to_none=True)
    for value in model.encoder_parameters():
        value.requires_grad_(False)
    latent_history = model.encode(tiny[:, :48]).detach()
    latent_future = latent_history[:, -1:].expand(-1, 4, -1, -1).clone()
    call_counts = [0, 0]
    handles = []
    for index, layer in enumerate(model.decoder):
        def hook(_module, _inputs, _output, index=index):
            call_counts[index] += 1
        handles.append(layer.register_forward_hook(hook))
    optimizer = torch.optim.Adam(model.decoder_parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    predicted = model.decode_future(latent_history, latent_future, horizon=16)
    loss = (predicted - tiny[:, 48:]).square().mean()
    loss.backward()
    optimizer.step()
    for handle in handles:
        handle.remove()
    current = dict(model.named_parameters())
    check("calibration_freezes_encoder_and_shared_product",
          all(torch.equal(current[name].detach(), value) and current[name].grad is None
              for name, value in old_encoder_state.items()))
    check("calibration_updates_same_native_decoder_and_calls_both_GT_layers",
          all((current[name].detach() - value).abs().max() > 0 for name, value in old_decoder_state.items())
          and call_counts == [1, 1], native_decoder_call_counts=call_counts,
          decoder_trainable_coefficients=sum(p.numel() for p in model.decoder_parameters()))
    model.eval()
    with torch.no_grad():
        past = tiny[:, :48]
        actual_future = tiny[:, 48:]
        changed_future = torch.randn_like(actual_future) * 100
        past_latent = model.encode(past)
        first = model.encode(torch.cat([past, actual_future], dim=1))[:, :12]
        second = model.encode(torch.cat([past, changed_future], dim=1))[:, :12]
    check("past_latent_append_future_perturbation_invariance",
          torch.equal(first, second) and torch.equal(first, past_latent),
          actual_TSFM_inference_scope=False,
          scope="Encoder causal pooling/operator only; runner verifies actual frozen-F inference separately")
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = GTConvAE(model.product.graph).double().eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        check("save_load_prediction_exact", torch.equal(model(tiny), restored(tiny)))
    full_report = restored.parameter_report()
    check("paper_feature_bank_parameter_counts", full_report["encoder_filter_coefficients"] == 120
          and full_report["decoder_filter_coefficients"] == 120
          and full_report["product_coefficients_trainable"] == 4
          and full_report["total_trainable"] == 244
          and no_spatial.parameter_report()["total_trainable"] == 242, full=full_report)
    check("CPU_only_no_CUDA_initialization", not torch.cuda.is_initialized())
    return {"schema": "GT_NATIVE_CPU_IMPLEMENTATION_DIAGNOSTICS_V1", "status": "PASS",
            "checks": checks, "checks_passed": len(checks), "scientific_fits": 0,
            "actual_data_reads": 0, "protected_TEST_reads": 0,
            "GPU_work_seconds": 0, "CUDA_initialized": torch.cuda.is_initialized(),
            "tiny_fixture_optimizer_steps": 601, "alignment_L512_H48": GTConvAE.alignment_report(),
            "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                              for name in ["models_gt.py", "verify_gt_cpu.py"]},
            "resources": {"wall_seconds": time.perf_counter() - START_WALL,
                          "whole_process_CPU_seconds_since_import_timer": time.process_time() - START_CPU,
                          "peak_RSS_bytes": getattr(psutil.Process().memory_info(), "peak_wset",
                                                    psutil.Process().memory_info().rss)}}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = HERE / f"GT_CPU_VERIFICATION_{time.time_ns()}.json"
    try:
        result = verify()
    except Exception as error:
        result = {"status": "FAIL", "error": repr(error), "scientific_fits": 0,
                  "protected_TEST_reads": 0, "GPU_work_seconds": 0,
                  "resources": {"wall_seconds": time.perf_counter() - START_WALL,
                                "whole_process_CPU_seconds_since_import_timer": time.process_time() - START_CPU}}
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"path": str(path), **result}, ensure_ascii=False))
        raise
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"path": str(path), "status": result["status"],
                      "checks_passed": result["checks_passed"], "resources": result["resources"]}, ensure_ascii=False))
