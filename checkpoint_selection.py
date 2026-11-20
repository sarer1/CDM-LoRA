"""Record selection windows without changing training or consuming randomness."""
from pathlib import Path
import json,os,re,shutil
from experiment_config import atomic_json
from metrics import file_sha256


def restore_best_selection(run_dir,start_epoch,model):
    if start_epoch<=1:return start_epoch-1,None,0
    candidates=list(Path(run_dir).glob(f'{model}_best_epoch*_mIoU*.pth'))
    if len(candidates)!=1:
        raise RuntimeError('Resume requires one preserved best checkpoint for selection bookkeeping')
    path=candidates[0]
    match=re.fullmatch(re.escape(model)+r'_best_epoch(\d+)_mIoU.*\.pth',path.name)
    epoch=int(match[1])
    if epoch>=start_epoch:raise RuntimeError('Best checkpoint comes from after the resume point')
    return epoch,str(path),start_epoch-1-epoch


def preserve_selection(run_dir,epoch,best_epoch,best_score,best_path):
    config=Path(os.environ['R16_ACTIVE_CONFIG'])
    spec=json.loads(config.read_text(encoding='utf-8'))
    if epoch not in spec.get('checkpoint_selection_horizons',[]):return
    if not best_path or not Path(best_path).is_file():
        raise RuntimeError('Selection window ended without a best checkpoint')
    directory=Path(run_dir)/'checkpoint_selections'/f'epoch{epoch}'
    directory.mkdir(parents=True,exist_ok=True)
    target=directory/'best.pth'
    if target.exists():
        read_selection(run_dir,epoch,spec['fingerprint'])
        return
    temporary=target.with_suffix('.pth.tmp')
    shutil.copyfile(best_path,temporary);temporary.replace(target)
    atomic_json(directory/'selection.json',{
        'selection_horizon_epochs':epoch,'learning_rate_horizon_epochs':spec['recipe']['epochs'],
        'best_epoch':best_epoch,'selection_score':best_score,
        'checkpoint':str(target.resolve()),'checkpoint_sha256':file_sha256(target),
        'config_fingerprint':spec['fingerprint'],
        'policy':'Original best-update rule restricted to the first N epochs; training continues unchanged'})


def read_selection(run_dir,epoch,fingerprint):
    path=Path(run_dir)/'checkpoint_selections'/f'epoch{epoch}'/'selection.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    if record['selection_horizon_epochs']!=epoch or record['config_fingerprint']!=fingerprint:
        raise RuntimeError('Checkpoint selection identity mismatch')
    if file_sha256(record['checkpoint'])!=record['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint was modified')
    return record,path
