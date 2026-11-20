"""Sequential evaluation with exact r16 inference and checkpoint-keyed caches."""
from pathlib import Path
import argparse,importlib,json,time
import numpy as np
import torch
from experiment_config import atomic_json,digest
from metrics import file_sha256,aggregate_tiles
from model_io import load_model,SOURCE

def resolve(template,tile):return Path(template(tile) if callable(template) else template.format(tile))

def evaluate(config_path,checkpoint,pretrained,output_root,batch_size=16,selection_metadata=None):
    config_path=Path(config_path);spec=json.loads(config_path.read_text(encoding='utf-8'))
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(selection_metadata).read_text()) if selection_metadata else None
    if selection and (selection['checkpoint_sha256']!=file_sha256(checkpoint)
                      or selection['config_fingerprint']!=spec['fingerprint']):
        raise RuntimeError('Evaluation checkpoint does not match selection metadata')
    net,U=load_model(spec,pretrained,output_root,config_path,checkpoint)
    net.eval()
    inference=importlib.import_module('inference')
    data_manifest=[]
    for tile in U.test_ids:
        paths=[resolve(t,tile) for t in U.MODALITY_FOLDERS]+[resolve(U.ERODED_FOLDER,tile)]
        data_manifest.extend({'tile':tile,'path':str(p),'sha256':file_sha256(p)} for p in paths)
    provenance={'checkpoint_sha256':file_sha256(checkpoint),
                'config_fingerprint':spec['fingerprint'],'dataset':U.DATASET,
                'test_stride':spec['recipe']['test_stride'],'window_size':list(U.WINDOW_SIZE),
                'batch_size':batch_size,'eval_class_ids':list(U.EVAL_CLASS_IDS),
                'inference_source_sha256':file_sha256(SOURCE/'inference.py'),
                'preprocessing_source_sha256':file_sha256(SOURCE/'utils.py'),
                'evaluation_driver_sha256':file_sha256(Path(__file__)),
                'metric_source_sha256':file_sha256(Path(__file__).with_name('metrics.py')),
                'data':data_manifest}
    fingerprint=digest(provenance)
    directory=output_root/'evaluations'/fingerprint[:16]
    tiles_dir=directory/'tiles';tiles_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'evaluation_manifest.json',dict(provenance,evaluation_fingerprint=fingerprint))
    for tile in U.test_ids:
        record=tiles_dir/f'{tile}.json'
        if record.exists():
            old=json.loads(record.read_text(encoding='utf-8'))
            if old.get('evaluation_fingerprint')!=fingerprint:raise RuntimeError('Incompatible evaluation cache')
            continue
        started=time.time()
        mods=[U.load_test_modality(tile,m) for m in range(U.N_MODALITIES)]
        gt=U.load_eroded_label(tile)
        pred=inference.infer_tile(net,mods,spec['recipe']['test_stride'],batch_size)
        cm=np.zeros((U.N_CLASSES,U.N_CLASSES),dtype=np.int64)
        U.add_to_confusion_matrix(cm,pred,gt,U.N_CLASSES)
        atomic_json(record,{'tile':tile,'evaluation_fingerprint':fingerprint,'cm':cm.tolist(),
                            'seconds':time.time()-started,'npix':int(gt.size)})
        aggregate=aggregate_tiles(tiles_dir,list(U.test_ids),fingerprint,U.N_CLASSES,U.EVAL_CLASS_IDS)
        atomic_json(output_root/'evaluation_progress.json',dict(aggregate,evaluation_fingerprint=fingerprint))
        print(f'[eval] {tile} completed in {time.time()-started:.1f}s',flush=True)
        del mods,gt,pred
    result=aggregate_tiles(tiles_dir,list(U.test_ids),fingerprint,U.N_CLASSES,U.EVAL_CLASS_IDS)
    if not result['complete']:raise RuntimeError('Refusing to publish incomplete test result')
    result.update(dataset=U.DATASET,seed=spec['recipe']['seed'],condition=spec['recipe']['condition'],
                  checkpoint=str(checkpoint),config_fingerprint=spec['fingerprint'],
                  evaluation_fingerprint=fingerprint,test_stride=spec['recipe']['test_stride'],
                  evaluation_directory=str(directory))
    if selection:result['checkpoint_selection']=selection
    origin=output_root/'legacy_checkpoint_origin.json'
    if origin.exists():
        result['training_origin']=json.loads(origin.read_text())
    result['per_class_iou']={name:value for name,value in zip(U.LABELS,result['per_class_iou'])}
    result['per_class_f1']={name:value for name,value in zip(U.LABELS,result['per_class_f1'])}
    atomic_json(directory/'test_result.json',result)
    atomic_json(output_root/'test_result.json',result)
    print(json.dumps({k:result[k] for k in ['condition','test_mIoU','test_OA','n_test_tiles']}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--pretrained',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--selection-metadata')
    args=p.parse_args()
    evaluate(args.config,args.checkpoint,args.pretrained,args.output,args.batch_size,args.selection_metadata)
