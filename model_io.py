"""Load one model per process, with explicit recipe and strict checkpoints."""
from pathlib import Path
import importlib,json,os,sys
import torch
from experiment_config import build_environment
from experiment_runtime import configure_model

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source'

def activate(spec,pretrained,run_dir,config_path,mode='Test'):
    env=build_environment(spec,pretrained,run_dir,config_path,mode)
    for key in list(os.environ):
        if key.startswith(('MFNET_','R16_')):os.environ.pop(key)
    os.environ.update({k:v for k,v in env.items() if k.startswith(('MFNET_','R16_')) or k=='CUDA_VISIBLE_DEVICES'})
    sys.path.insert(0,str(SOURCE))
    U=importlib.import_module('utils')
    module=importlib.import_module('MedSAM.UNetFormer_MMSAM')
    if list(U.test_ids)!=spec['dataset_protocol']['tiles']:raise RuntimeError('Dataset split differs from the pinned recipe')
    if U.BATCH_SIZE!=spec['recipe']['batch_size']:raise RuntimeError('Training batch size differs from recipe')
    return U,module.UNetFormer

def require_gpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('GPU 0 binding missing')
    if not torch.cuda.is_available():raise RuntimeError('GPU 0 is not available to PyTorch')
    return torch.device('cuda:0')

def load_model(spec,pretrained,run_dir,config_path,checkpoint=None):
    U,MFNet=activate(spec,pretrained,run_dir,config_path)
    device=require_gpu()
    U.set_seed(spec['recipe']['seed'])
    net=MFNet(num_classes=U.N_CLASSES,n_modalities=U.N_MODALITIES,
              encoder_type=U.ENCODER_TYPE,encoder_size=U.ENCODER_SIZE,
              decoder_type=U.DECODER_TYPE,dataset=U.DATASET).to(device)
    configure_model(net,spec)
    if checkpoint is not None:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        sd=payload.get('model_state_dict',payload) if isinstance(payload,dict) else payload
        dictionaries=[tensor for name,tensor in sd.items() if name.endswith('evd_wc_shared')]
        if dictionaries and any(not torch.equal(dictionaries[0],d) for d in dictionaries[1:]):
            raise RuntimeError('Checkpoint disagrees across supposedly shared class-dictionary entries')
        net.load_state_dict(sd,strict=True)
        del payload,sd
    return net,U
