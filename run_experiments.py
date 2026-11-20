"""Prepare and execute immutable, explicit experimental jobs."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
from experiment_config import DATASETS,MAIN_CONDITION,Recipe,build_environment,digest,atomic_json
from metrics import file_sha256

HERE=Path(__file__).resolve().parent
WEIGHT='dinov3_vitl16_pretrain_lvd1689m_8aa4cbdd.pth'

def code_version():
    manifest=json.loads((HERE/'source_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['generated_sha256'].items():
        if file_sha256(HERE/'source'/name)!=expected:raise RuntimeError(f'Pinned source modified: {name}')
    runtime={name:file_sha256(HERE/name) for name in ['experiment_runtime.py','experiment_config.py','checkpoint_selection.py']}
    return {'source':manifest['generated_digest'],'runtime':runtime}

def prepare(recipe,legacy_root,output_root):
    legacy_root=Path(legacy_root).resolve();output_root=Path(output_root).resolve()
    pretrained=legacy_root/'weights'/WEIGHT
    if not pretrained.exists():raise FileNotFoundError(pretrained)
    spec=recipe.dictionary();spec['code']=code_version()
    cache=output_root/'pretrained_fingerprint.json'
    stat=pretrained.stat()
    identity={'path':str(pretrained),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    old=json.loads(cache.read_text()) if cache.exists() else {}
    if old.get('identity')!=identity:
        old={'identity':identity,'sha256':file_sha256(pretrained)};atomic_json(cache,old)
    spec['pretrained_sha256']=old['sha256']
    spec['fingerprint']=digest(spec)
    run_dir=output_root/recipe.dataset/recipe.condition/f'seed{recipe.seed}_{spec["fingerprint"][:10]}'
    config=run_dir/'run_config.json'
    if config.exists() and json.loads(config.read_text())!=spec:raise RuntimeError('Run-directory configuration mismatch')
    atomic_json(config,spec)
    return {'config':str(config),'run_dir':str(run_dir),'pretrained':str(pretrained),
            'legacy_root':str(legacy_root),'fingerprint':spec['fingerprint'],
            'condition':recipe.condition,'dataset':recipe.dataset,'seed':recipe.seed}

def train_job(job):
    config=Path(job['config']);run_dir=Path(job['run_dir'])
    spec=json.loads(config.read_text(encoding='utf-8'))
    if spec['recipe']['condition']!=MAIN_CONDITION:
        raise RuntimeError(f"Only {MAIN_CONDITION} is supported by this training release")
    if code_version()!=spec['code']:raise RuntimeError('Model/runtime code changed after the job was prepared')
    env=build_environment(spec,job['pretrained'],run_dir,config)
    complete=run_dir/'summary.json'
    if complete.exists():
        summary=json.loads(complete.read_text())
        if summary.get('stop_epoch')==spec['recipe']['epochs']:
            print('[train] already complete under the identical configuration',flush=True);return
        raise RuntimeError('Existing summary does not complete the locked budget')
    checkpoint=run_dir/'UNetformer_checkpoint_latest.pth'
    if checkpoint.exists():env['MFNET_RESUME']=str(checkpoint)
    env_record={k:v for k,v in env.items() if k.startswith('MFNET_') or k=='CUDA_VISIBLE_DEVICES'}
    atomic_json(run_dir/'effective_environment.json',env_record)
    atomic_json(run_dir/'phase.json',{'phase':'training','started_utc':time.time(),'fingerprint':spec['fingerprint']})
    with (run_dir/'train_console.log').open('a',encoding='utf-8') as log:
        log.write('\n'+json.dumps({'start':time.time(),'resume':checkpoint.exists(),'config':spec['fingerprint']})+'\n');log.flush()
        child=subprocess.run([sys.executable,str(HERE/'source'/'train.py')],cwd=job['legacy_root'],env=env,stdout=log,stderr=subprocess.STDOUT)
    if child.returncode:raise RuntimeError(f'Training failed with exit {child.returncode}; see {run_dir}/train_console.log')
    summary=json.loads(complete.read_text())
    if summary.get('stop_epoch')!=spec['recipe']['epochs']:raise RuntimeError('Training did not complete the locked epoch budget')
    atomic_json(run_dir/'phase.json',{'phase':'trained','finished_utc':time.time(),'fingerprint':spec['fingerprint']})

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['plan','train'])
    p.add_argument('--condition',choices=[MAIN_CONDITION],default=MAIN_CONDITION)
    p.add_argument('--dataset',choices=list(DATASETS),default='Vaihingen')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--iters',type=int,default=1000)
    p.add_argument('--legacy-root',required=True);p.add_argument('--output-root',required=True)
    a=p.parse_args()
    job=prepare(Recipe(condition=a.condition,dataset=a.dataset,seed=a.seed,epochs=a.epochs,iters_per_epoch=a.iters),a.legacy_root,a.output_root)
    print(json.dumps(job),flush=True)
    if a.action=='train':train_job(job)

if __name__=='__main__':main()
