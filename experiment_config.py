"""Complete, isolated r16 experimental recipes. Does not import the model."""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,os

MAIN_CONDITION='cdm_lora'
MAIN_MECHANISM=dict(auxiliary_ce=True,source_modulation=True,head_refinement=False,
                    cross_mode='relation',gate_source='class_readout',class_feedback=True)
DATASETS={'Vaihingen':dict(classes=6,eval_ids=list(range(5)),val_stride=128,tiles=['5','21','15','30']),
          'Potsdam':dict(classes=6,eval_ids=list(range(5)),val_stride=256,tiles=['4_10','5_11','2_11','3_10','6_11','7_12'])}

def _whu_test_tiles():
    split = Path(__file__).resolve().parent / 'source' / 'division.txt'
    _, test = split.read_text(encoding='utf-8').split('test:', 1)
    return [Path(line.strip()).stem for line in test.splitlines() if line.strip()]


DATASETS['WHU-OPT-SAR'] = dict(classes=8, eval_ids=list(range(1, 8)),
                              val_stride=256, tiles=_whu_test_tiles())


@dataclass(frozen=True)
class Recipe:
    condition:str=MAIN_CONDITION
    dataset:str='Vaihingen'
    seed:int=42
    epochs:int=30
    iters_per_epoch:int=1000
    batch_size:int=10
    learning_rate:float=1e-4
    patience:int=100
    test_stride:int=32
    rank:int=8
    gpu:int=0
    readout_auxiliary_weight:float=0.25
    output_sampling_probability:float=0.3
    mlp_lora_scale:float=1.0

    def dictionary(self):
        if self.condition!=MAIN_CONDITION: raise ValueError(f"Only {MAIN_CONDITION} is supported by this training release")
        if self.dataset not in DATASETS: raise ValueError(self.dataset)
        if self.gpu!=0: raise ValueError('This authorized experiment suite is bound to GPU 0')
        if min(self.epochs,self.iters_per_epoch,self.batch_size,self.rank)<=0: raise ValueError('Invalid positive recipe field')
        recipe=asdict(self)
        if recipe['readout_auxiliary_weight']<0 or not 0<=recipe['output_sampling_probability']<=1:
            raise ValueError('Invalid loss weight or output sampling probability')
        if recipe['mlp_lora_scale']<=0:raise ValueError('MLP residual scale must be positive')
        return {'schema':2,'recipe':recipe,'mechanism':dict(MAIN_MECHANISM),
                'dataset_protocol':DATASETS[self.dataset],
                'feedback':'softmax(Wc z) through Be',
                'head_pixel_losses':True,'output_sampling_probability':recipe['output_sampling_probability'],
                'readout_auxiliary_weight':recipe['readout_auxiliary_weight'],'expected_mmlinear_sites':48,
                'checkpoint_selection_horizons':[]}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build_environment(spec,pretrained,run_dir,config_path,mode='Train',inherited=None):
    # Ignore inherited experiment flags so each recipe runs in isolation.
    inherited=dict(os.environ if inherited is None else inherited)
    env={k:v for k,v in inherited.items() if not k.startswith(('MFNET_','R16_'))}
    r=spec['recipe']; m=spec['mechanism']; d=spec['dataset_protocol']
    if r['gpu']!=0: raise ValueError('GPU 0 only')
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',R16_ACTIVE_CONFIG=str(config_path),
        MFNET_DATASET=r['dataset'],MFNET_SEED=str(r['seed']),MFNET_MODE=mode,
        MFNET_EPOCHS=str(r['epochs']),MFNET_ITERS=str(r['iters_per_epoch']),
        MFNET_PATIENCE=str(r['patience']),MFNET_MIN_DELTA='0.0001',
        MFNET_WARMUP_EPOCHS='0',
        MFNET_RUN_DIR=str(run_dir),MFNET_OPT='adamw',MFNET_LR=str(r['learning_rate']),
        MFNET_ENCODER_TYPE='dinov3',MFNET_ENCODER_SIZE='large',MFNET_DINO3_CKPT=str(pretrained),
        MFNET_ARCH='light',MFNET_TERMS='priv,xgl',MFNET_RANK=str(r['rank']),
        MFNET_XGA='1',MFNET_XGA2='1',MFNET_MDROP=str(spec['output_sampling_probability']),MFNET_XGA_PHL='1',
        MFNET_READOUT_AUX_WEIGHT=str(spec['readout_auxiliary_weight']),
        MFNET_EVD='1',MFNET_EVD_PURE='1',MFNET_EVD_V3='1',MFNET_EVD_FUSE='1',
        MFNET_EVD_EDL='1',MFNET_EVD_DS16='1',MFNET_EVD_K=str(d['classes']),
        MFNET_EVD_DS14='0',MFNET_ALPHA_FIX='0',MFNET_EVD_NOWRITE='0' if m.get('class_feedback',True) else '1',
        MFNET_EVD16_GCONST='0' if m['source_modulation'] else '1',
        MFNET_EVD_NOSUP='0' if m['auxiliary_ce'] else '1',
        MFNET_EREF='1' if m['head_refinement'] else '0',
        MFNET_VAL_STRIDE=str(d['val_stride']))
    return env

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)
