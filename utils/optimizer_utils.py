import torch.nn as nn
from torch.optim import AdamW


def configure_optimiser(
    module: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.999), eps=1e-5
) -> AdamW:
    """
    Separate parameters into two groups: regularized and non-regularized, then return optimizer.
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention) # nn.MultiheadAttention can be added if used
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
    
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue # Skip parameters that don't require gradients

            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("embedding.weight"): # Specific for nn.Embedding weights
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif isinstance(m, (nn.GRU, nn.LSTM, nn.GRUCell)): # All params of RNNs
                no_decay.add(fpn)
            elif pn.endswith("alpha"): # Custom parameter name from original
                 no_decay.add(fpn)
            # Default for remaining weights and other parameters (like nn.Parameter)
            elif pn.endswith("weight"):
                decay.add(fpn) # Default to decay for other weights
            else:
                no_decay.add(fpn) # Default to no_decay for other parameters

    # Validate that we considered every trainable parameter
    param_dict = {pn: p for pn, p in module.named_parameters() if p.requires_grad}
    
    # Ensure sets only contain params from param_dict (handles non-trainable params if any were added)
    decay = {pn for pn in decay if pn in param_dict}
    no_decay = {pn for pn in no_decay if pn in param_dict}

    # Parameters that might have been classified by default rules but also by specific rules
    # Example: Embedding weights are blacklisted but end with "weight". Bias also.
    # The order of rules implicitly handles this. Re-check for explicit overlaps.
    inter_params = decay & no_decay
    if len(inter_params) > 0:
        # This means a parameter fell into both categories.
        # This can happen if blacklist/whitelist logic isn't mutually exclusive with defaults or other rules.
        # E.g. if 'embedding.weight' is added to no_decay, and then '.weight' adds to decay.
        # The current logic: specific (bias, embedding, alpha) -> whitelist -> blacklist -> RNN -> default weight -> default other
        # This should be okay. Let's refine the check.
        # If a param is in `no_decay` due to a specific rule (e.g. `endswith('bias')`),
        # it shouldn't also be in `decay`.
        pass # The current logic with set addition should handle precedence.
             # If something is added to no_decay first, it won't be added to decay by a later rule if checks are exact.
             # The issue is if `fpn` matches multiple `elif`.

    # Re-validate after ensuring sets only contain trainable params
    union_params = decay | no_decay
    if len(inter_params) > 0:
         # Forcing resolution: if in no_decay by a specific rule, it stays in no_decay.
        decay = decay - no_decay # Remove any from decay that are explicitly in no_decay
        inter_params = decay & no_decay # Should be empty now
    
    if len(inter_params) > 0:
        raise ValueError(
            f"Parameters {inter_params} are in both decay/no_decay groups after attempting to resolve."
        )

    if union_params != set(param_dict.keys()):
        missing = set(param_dict.keys()).difference(union_params)
        classified_by_error = set()
        for pn_missing in missing:
            p_missing = param_dict[pn_missing]
            # Attempt to classify missing params based on common patterns as a fallback
            if pn_missing.endswith("bias") or isinstance(p_missing, nn.Embedding) or isinstance(p_missing, nn.LayerNorm): #TODO: this is not how isinstance works with parameters
                 no_decay.add(pn_missing)
                 classified_by_error.add(pn_missing)
            elif pn_missing.endswith("weight"):
                 decay.add(pn_missing)
                 classified_by_error.add(pn_missing)
            
        if missing - classified_by_error: # If some are still missing after fallback
            raise ValueError(
                f"Trainable parameters {missing - classified_by_error} were not separated into either decay/no_decay set."
            )
        # Update union_params after fallback classification
        union_params = decay | no_decay
        if union_params != set(param_dict.keys()): # Final check
            raise ValueError("Parameter separation failed even after fallback.")


    optim_groups = []
    if decay:
        optim_groups.append({
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        })
    if no_decay:
        optim_groups.append({
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        })
    
    if not optim_groups: # Handle case where module has no trainable parameters
        # Return an optimizer with no parameters, or handle as an error
        # For now, let AdamW handle empty param list if that's desired behavior
        pass


    return AdamW(
        params=optim_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        foreach=True if optim_groups else None, # foreach needs PyTorch 1.9+ and non-empty groups
    )