"""Attach controlled mechanisms to the pinned r16 without changing its defaults."""
from pathlib import Path
import json,math,os,random
import numpy as np
import torch
from torch import nn
from experiment_config import atomic_json

def capture_rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

def restore_rng_state(state):
    if state is None: raise RuntimeError('Exact-resume checkpoint is missing its RNG state')
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def override_cross(layer,receiver,codes,private,legacy):
    peer=1-receiver
    if layer.paper_cross_mode=='response_mix':
        weight=layer.paper_mix_logits[receiver].sigmoid()
        other=codes[peer]@layer.lora_B_list[peer].T
        # The caller adds the private update, yielding a weighted sum of the two updates.
        return (1-weight)*(other-private)
    if layer.paper_gate_source=='feature_projection':
        logits=codes[peer]@layer.paper_gate_readout.T
        alpha=1+torch.nn.functional.softplus(logits)-math.log(2)
        gamma=1-alpha.shape[-1]/alpha.sum(-1)
        source=codes[peer]@layer.xgl_Wx.T
        relation=torch.cat([codes[receiver]*source,(codes[receiver]-source).abs()],-1)
        return (relation@layer.xgl_Bx.T)*gamma.unsqueeze(-1)
    return legacy

def configure_model(model,spec=None,write_audit=True):
    if spec is None:
        filename=os.environ.get('R16_ACTIVE_CONFIG')
        if not filename: raise RuntimeError('Use the controlled suite entry point; no active experiment configuration')
        spec=json.loads(Path(filename).read_text(encoding='utf-8'))
    m=spec['mechanism']
    layers=[(name,layer) for name,layer in model.named_modules() if type(layer).__name__=='MMLinear']
    expected=spec.get('expected_mmlinear_sites',48)
    if len(layers)!=expected: raise RuntimeError(f'Expected {expected} r16 MLP linear sites, got {len(layers)}')
    dictionaries={id(layer.evd_wc_shared) for _,layer in layers}
    if len(dictionaries)!=1: raise RuntimeError('The class dictionary is not shared across all sites')
    active_heads=sum(bool(getattr(module,'ev_gate',False)) for module in model.modules())
    if expected==48 and active_heads!=(2 if m['head_refinement'] else 0):
        raise RuntimeError(f'Unexpected head refinement topology: {active_heads}')
    free_gate=None
    if m['gate_source']=='feature_projection':
        first=layers[0][1];K,R=first.evd_wc_shared.shape
        # A nonzero initial projection lets the gate learn alongside the zero-initialized update.
        free_gate=nn.Parameter(torch.randn(K,R,device=first.weight.device,dtype=first.weight.dtype)*0.01)
    for name,layer in layers:
        if not getattr(layer,'_use_ds16',False) or not getattr(layer,'_use_evd_v3',False):
            raise RuntimeError(f'{name}: the required r16 readout branch is inactive')
        layer.xgl_belief_kill=not m.get('class_feedback',True)
        if layer.xgl_belief_kill:
            layer.xgl_Be.requires_grad_(False)
            layer.evd_wc_shared.requires_grad_(False)
        layer.paper_site_name=name
        if layer.r!=spec['recipe']['rank']:raise RuntimeError('Actual MLP rank differs from recipe')
        layer.scaling=spec['recipe'].get('mlp_lora_scale',layer.scaling)
        layer.lora_alpha=layer.scaling*layer.r
        layer.paper_cross_mode=m['cross_mode'];layer.paper_gate_source=m['gate_source']
        layer.xgl_cross_kill=m['cross_mode']=='none'
        layer._ds16_gconst=not m['source_modulation']
        if m['cross_mode'] in ('none','response_mix'):
            layer.xgl_Bx.requires_grad_(False)
            layer.xgl_Wx.requires_grad_(False)
        if m['cross_mode']=='response_mix':
            layer.register_parameter('paper_mix_logits',nn.Parameter(torch.zeros(2,device=layer.weight.device,dtype=layer.weight.dtype)))
            layer.paper_cross_override=override_cross
        if free_gate is not None:
            layer.register_parameter('paper_gate_readout',free_gate)
            layer.paper_cross_override=override_cross
    audit={'condition':spec['recipe']['condition'],'mmlinear_sites':len(layers),
           'shared_dictionary_objects':len(dictionaries),'cross_mode':m['cross_mode'],
           'gate_source':m['gate_source'],'auxiliary_ce':m['auxiliary_ce'],
           'head_refinement':m['head_refinement'],'class_feedback':m.get('class_feedback',True),
           'active_refinement_heads':active_heads,
           'total_parameters':sum(p.numel() for p in model.parameters()),
           'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
           'source_coefficient_constant':1.0 if not m['source_modulation'] else None,
           'feedback_parameterization':'softmax',
           'mlp_rank':spec['recipe']['rank'],'mlp_effective_scale':layers[0][1].scaling,
           'readout_auxiliary_weight':spec['readout_auxiliary_weight'],
           'output_sampling_probability':spec['output_sampling_probability'],
           'extra_feature_gate_initialization':'normal(std=0.01), shared across sites' if free_gate is not None else None}
    if write_audit:
        path=Path(os.environ['MFNET_RUN_DIR'])/'runtime_audit.json'
        atomic_json(path,audit)
    print('[r16 controlled runtime] '+json.dumps(audit))
    return audit
