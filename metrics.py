"""Complete, provenance-checked confusion-matrix aggregation."""
from pathlib import Path
import json,hashlib
import numpy as np

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def metrics(cm,eval_ids):
    cm=np.asarray(cm,dtype=np.int64)
    if cm.ndim!=2 or cm.shape[0]!=cm.shape[1] or np.any(cm<0):raise ValueError('Invalid confusion matrix')
    total=int(cm.sum())
    if not total:raise ValueError('No evaluated pixels')
    diag=np.diag(cm).astype(float)
    rows=cm.sum(1);cols=cm.sum(0);union=rows+cols-diag
    iou=np.divide(diag,union,out=np.full_like(diag,np.nan),where=union>0)
    f1=np.divide(2*diag,rows+cols,out=np.zeros_like(diag),where=rows+cols>0)
    expected=float(np.dot(rows.astype(float),cols.astype(float)))/(total*total)
    accuracy=float(diag.sum()/total)
    return {'test_mIoU':float(np.nanmean(iou[list(eval_ids)])),
            'test_OA':accuracy*100,'kappa':(accuracy-expected)/(1-expected) if expected<1 else 0.,
            'per_class_iou':[float(x) if np.isfinite(x) else None for x in iou],
            'per_class_f1':f1.tolist(),'pixels':total}

def aggregate_tiles(directory,expected_tiles,fingerprint,classes,eval_ids):
    directory=Path(directory);records=[];missing=[]
    for tile in expected_tiles:
        path=directory/f'{tile}.json'
        if not path.exists():missing.append(tile);continue
        item=json.loads(path.read_text(encoding='utf-8'))
        if item.get('evaluation_fingerprint')!=fingerprint:
            raise ValueError(f'Incompatible checkpoint/protocol cache: {path}')
        if item.get('tile')!=tile:raise ValueError(f'Tile identity mismatch: {path}')
        cm=np.asarray(item['cm'],dtype=np.int64)
        if cm.shape!=(classes,classes) or np.any(cm<0):raise ValueError(f'Invalid tile matrix: {path}')
        records.append(cm)
    if missing:return {'complete':False,'completed_tiles':len(records),'missing_tiles':missing}
    summed=sum(records,np.zeros((classes,classes),dtype=np.int64))
    return {'complete':True,'n_test_tiles':len(expected_tiles),'cm':summed.tolist(),**metrics(summed,eval_ids)}
