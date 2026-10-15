"""Legacy independent latent head with frozen E/center and the existing causal L7 decoder."""
import hashlib

import numpy as np
import torch

from linear_e_adaptation_v1 import LinearLagAdapter,global_fit_macro_mse,require
from native_head_v1 import NativeHead,clone_state,state_hash,score_predictions,select_checkpoint
from run_linear_e_adaptation_v1 import predict_components,byte_equal


class LaggedNativeForecaster(LinearLagAdapter):
    def __init__(self,center,encoder,decoder,lags,L=512,H=48):
        super().__init__(center,encoder,decoder,lags,learn_E=False)
        self.D.requires_grad_(False)
        self.head=NativeHead(self.K,L,H)

    def forward(self,x,bolt=None,*,return_components=False):
        require(bolt is None,"NATIVE_HEAD_NO_F_CALLBACK_ALLOWED")
        past=self.encode(x); future=self.head(past); prediction=self.decode(x[:,-1],past,future)
        return (prediction,past,future) if return_components else prediction


def frozen_status(model):
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    require(names==["head.weight","head.bias"] and not model.E.requires_grad and not model.D.requires_grad and not model.center.requires_grad and model.E.grad is None and model.D.grad is None,"FROZEN_E_D_CENTER_AND_ONLY_HEAD_TRAINABLE")
    return {"trainable_parameter_names":names,"E_requires_grad":False,"D_requires_grad":False,"center_requires_grad":False,"E_grad_is_None":True,"D_grad_is_None":True}


def train_head(model,fit,tune,seed,check,checkpoint,*,updates=600):
    require(type(updates) is int and 0<=updates<=600,"NATIVE_UPDATE_BUDGET_0_TO_600")
    frozen_status(model); require(all(p.grad is None for p in model.parameters()) and all(torch.count_nonzero(p)==0 for p in model.head.parameters()),"FRESH_ZERO_INITIALIZED_HEAD_REQUIRED")
    initial=clone_state(model); device=model.E.device
    tensors={k:torch.as_tensor(fit[k],device=device) for k in ("history","target","observed")}
    counts=tensors["observed"].sum((0,1)); require(bool((counts>0).any()),"FIT_HAS_OBSERVED_CHANNELS")
    order=np.random.default_rng(seed).integers(0,len(fit["history"]),size=(updates,4))
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=.001,weight_decay=0.)
    require({id(p) for g in optimizer.param_groups for p in g["params"]}=={id(p) for p in model.head.parameters()},"OPTIMIZER_EXACTLY_HEAD_PARAMETER_IDS")
    checkpoints=[]; flags_checked=0; bytes_checked=0; clipped=0
    cp_updates=sorted({0,updates,*(u for u in (100,200,400,600) if u<=updates)})
    for update in range(updates+1):
        if update in cp_updates:
            frozen_status(model); state=clone_state(model)
            require(all(byte_equal(state[k].numpy(),initial[k].numpy()) for k in ("E","D","center")),"EVERY_CP_FROZEN_E_D_CENTER_BYTES")
            bytes_checked+=1; output=predict_components(model,None,tune["history"],check); score=score_predictions(output["prediction"],tune["target"],tune["observed"])
            item={"update":update,"tune":score,"state":state,"state_sha256":state_hash(state)}
            item["artifact_metadata"]=checkpoint(update,state,score,output); checkpoints.append(item)
        if update==updates: break
        ids=torch.as_tensor(order[update],device=device,dtype=torch.long); optimizer.zero_grad(set_to_none=True)
        prediction=model(tensors["history"][ids]); loss=global_fit_macro_mse(prediction,tensors["target"][ids],tensors["observed"][ids],counts,len(fit["history"]))
        require(bool(torch.isfinite(loss)),"FINITE_NATIVE_TRAINING_LOSS"); loss.backward()
        weight_norm=float(model.head.weight.grad.norm().detach()); bias_norm=float(model.head.bias.grad.norm().detach())
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.); require(bool(torch.isfinite(norm)),"FINITE_NATIVE_HEAD_GRADIENT")
        optimizer.step(); flags=frozen_status(model); flags_checked+=1
        check({"stage":"OPTIMIZER_STEP_EXECUTED","update":update+1})
        require(all(bool(torch.isfinite(p).all()) for p in parameters),"FINITE_TRAINED_NATIVE_HEAD")
        clipped+=int(norm>1.)
        check({"stage":"TRAINING_UPDATE","update":update+1,"loss":float(loss.detach()),"head_weight_gradient_norm_before_clip":weight_norm,"head_bias_gradient_norm_before_clip":bias_norm,"global_gradient_norm_before_clip":float(norm.detach()),"clipped":bool(norm>1.),"gradient_clipping_parameter_scope":"HEAD_ONLY",**flags})
    trained_head_changed=any(not byte_equal(model.state_dict()[k].detach().cpu().numpy(),initial[k].numpy()) for k in ("head.weight","head.bias"))
    selected=select_checkpoint(checkpoints); model.load_state_dict(selected["state"]); model.eval(); final=clone_state(model); flags=frozen_status(model)
    require(all(byte_equal(final[k].numpy(),initial[k].numpy()) for k in ("E","D","center")),"SELECTED_FROZEN_E_D_CENTER_BYTES")
    return {"model":model,"training_updates_completed":updates,"selected_update":selected["update"],"selected_tune":selected["tune"],"selected_state_sha256":selected["state_sha256"],"order_sha256":hashlib.sha256(order.tobytes()).hexdigest(),"FIT_channel_observed_counts":counts.cpu().tolist(),"loss":"N_fit/B/C_active * sum_c SSE_batch_c/N_fit_c; fixed entire-FIT counts","global_clipping_fraction":clipped/max(1,updates),"post_optimizer_frozen_flag_checks":flags_checked,"frozen_state_checkpoint_byte_checks":bytes_checked,"final_frozen_status":flags,"training_head_values_changed":trained_head_changed,"selected_head_nonzero":any(bool(torch.count_nonzero(p)) for p in model.head.parameters()),"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_update_frozen_bytes_replayed":False,"gradient_clipping_parameter_scope":"HEAD_ONLY"}
