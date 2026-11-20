import sys, os, csv, time, json, itertools
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(*a, **k): pass
from utils import *
from MedSAM.UNetFormer_MMSAM import UNetFormer as MFNet

# Seed before model initialization and random crop sampling.
set_seed(SEED)


net = MFNet(
    num_classes=N_CLASSES,
    n_modalities=N_MODALITIES if 'N_MODALITIES' in globals() else 2,
    encoder_type=ENCODER_TYPE if 'ENCODER_TYPE' in globals() else 'dinov3',
    encoder_size=ENCODER_SIZE if 'ENCODER_SIZE' in globals() else 'large',
    decoder_type=DECODER_TYPE if 'DECODER_TYPE' in globals() else 'unetformer',
    dataset=DATASET if 'DATASET' in globals() else None,
).cuda()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_runtime import configure_model, capture_rng_state, restore_rng_state
from checkpoint_selection import preserve_selection, restore_best_selection
configure_model(net)


RESUME = RESUME_PATH if 'RESUME_PATH' in globals() else None

start_epoch = 1
start_iter = 0
best_mIoU = 0.0
losses = np.zeros(1000000)
mean_losses = np.zeros(100000000)

if RESUME and os.path.isfile(RESUME):
    print(f"Loading: {RESUME}")
    ckpt = torch.load(RESUME, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        start_iter = ckpt.get('iter', 0)
        best_mIoU = ckpt.get('best_mIoU', 0.0)
        if 'losses' in ckpt:
            losses[:len(ckpt['losses'])] = ckpt['losses']
        if 'mean_losses' in ckpt:
            mean_losses[:len(ckpt['mean_losses'])] = ckpt['mean_losses']
        optimizer_state = ckpt.get('optimizer_state_dict')
        scheduler_state = ckpt.get('scheduler_state_dict')
        print(f"  Full checkpoint → epoch {start_epoch}, best mIoU {best_mIoU:.4f}")
    else:
        missing, unexpected = net.load_state_dict(ckpt, strict=False)
        if missing:
            print(f"  {len(missing)} new keys (random init)")
        if unexpected:
            print(f"  {len(unexpected)} obsolete keys (ignored)")
        import re
        m = re.search(r'epoch(\d+)', RESUME)
        if m:
            start_epoch = int(m.group(1)) + 1
            print(f"  Weights-only → epoch {start_epoch} (from filename)")
        else:
            start_epoch = 1
            print(f"  Weights-only → epoch 1")
        start_iter = 0
        best_mIoU = 0.0
        optimizer_state = None
        scheduler_state = None
    print(f"  iter {start_iter}")
else:
    print("Training from scratch")
    optimizer_state = None
    scheduler_state = None


def count_trainable_params(net):
    """Return (total, frozen, trainable, trainable_detail_dict)."""
    total, frozen, trainable = 0, 0, 0
    detail = {}
    for n, p in net.named_parameters():
        n_ele = p.nelement()
        total += n_ele
        if p.requires_grad:
            trainable += n_ele
            if 'lora_' in n:
                cat = 'LoRA'
            elif 'adapter' in n or 'deep_adapter' in n:
                cat = 'Adapter'
            elif 'log_alpha' in n or 'tau' in n:
                cat = 'Potential'
            elif 'decoder' in n or 'fpn' in n or 'fusion' in n or 'segmentation' in n:
                cat = 'Decoder'
            else:
                cat = 'Other_trainable'
            detail[cat] = detail.get(cat, 0) + n_ele
        else:
            frozen += n_ele
    return total, frozen, trainable, detail

total, frozen, trainable, detail = count_trainable_params(net)
print(f"\n{'='*50}")
print(f"  Total params:     {total:>12,}")
print(f"  Frozen:           {frozen:>12,}")
print(f"  Trainable:        {trainable:>12,}")
for cat, n in sorted(detail.items()):
    print(f"    {cat:<20} {n:>10,}")
print(f"{'='*50}\n")

print(f"Training tiles:  {len(train_ids)}")
print(f"Testing tiles:   {len(test_ids)}")

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)


base_lr = float(os.environ.get('MFNET_LR', '0.01'))
if os.environ.get('MFNET_OPT', 'sgd') == 'adamw':
    base_lr = float(os.environ.get('MFNET_LR', '1e-4'))
    optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.01)
else:
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

if optimizer_state is not None:
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    print("Optimizer + scheduler restored.")
    restore_rng_state(ckpt.get('r16_rng_state'))


CKPT_DIR = RUN_DIR
os.makedirs(CKPT_DIR, exist_ok=True)

def save_full_checkpoint(epoch, iter_, best_mIoU, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'iter': iter_,
        'best_mIoU': best_mIoU,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'r16_rng_state': capture_rng_state(),
        'losses': losses[:iter_],
        'mean_losses': mean_losses[:iter_],
    }, path)
    print(f"  Checkpoint → {path}")



LOG_CSV = os.path.join(CKPT_DIR, f'{MODEL}_training_log.csv')

def init_log_csv():
    """Create the per-epoch log CSV with a header row.

    A fresh run (start_epoch <= 1) rewrites the file; a resumed run appends.
    """
    os.makedirs(os.path.dirname(LOG_CSV) or '.', exist_ok=True)
    if start_epoch <= 1 or not os.path.exists(LOG_CSV):
        with open(LOG_CSV, 'w') as f:
            f.write('epoch,lr,train_loss,train_time_s,val_mIoU,val_OA,best_mIoU\n')


def log_epoch(epoch, lr, train_loss, train_time, val_mIoU, val_OA, best_mIoU):
    """Append one epoch's metrics to the CSV log."""
    val_str = f'{val_mIoU:.4f}' if val_mIoU is not None else ''
    oa_str = f'{val_OA:.4f}' if val_OA is not None else ''
    with open(LOG_CSV, 'a') as f:
        f.write(f'{epoch},{lr:.6f},{train_loss:.6f},{train_time:.1f},{val_str},{oa_str},{best_mIoU:.4f}\n')





def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=None, window_size=WINDOW_SIZE):
    n_mod = N_MODALITIES if 'N_MODALITIES' in globals() else 2
    if batch_size is None:
        batch_size = VAL_BATCH_SIZE if 'VAL_BATCH_SIZE' in globals() else BATCH_SIZE * 2

    _resolve = lambda f, id: f(id) if callable(f) else f.format(id)
    test_labels = [np.asarray(io.imread(_resolve(LABEL_FOLDER, id)), dtype='uint8') for id in test_ids]
    eroded_labels = [load_eroded_label(id) for id in test_ids]

    all_preds, all_gts = [], []
    cm_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for idx, gt_e in enumerate(tqdm(eroded_labels, total=len(test_ids), leave=False, file=sys.stderr)):
            mods = [load_test_modality(test_ids[idx], m) for m in range(n_mod)]
            img = mods[0]
            pred = np.zeros(img.shape[:2] + (N_CLASSES,), dtype=np.float32)
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for coords in tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                              total=total, leave=False, file=sys.stderr):
                mod_patches_list = []
                for mod_idx in range(n_mod):
                    mod_data = mods[mod_idx]
                    if mod_data.ndim == 3:
                        patches = [np.copy(mod_data[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    else:
                        patches = [np.copy(mod_data[x:x + w, y:y + h]) for x, y, w, h in coords]
                    mod_patches_list.append(torch.from_numpy(np.asarray(patches)).cuda())
                outs = net(mod_patches_list, mode='Test').cpu().numpy()
                # Taper overlapping window predictions to reduce boundary artifacts.
                pw, ph = window_size
                wy, wx = np.hanning(ph), np.hanning(pw)
                weight = (wy[:, None] * wx[None, :]).astype(np.float32)
                for out, (x, y, w_n, h_n) in zip(outs, coords):
                    pred[x:x + w_n, y:y + h_n] += out.transpose((1, 2, 0)) * weight[:, :, None]
            pred_argmax = np.argmax(pred, axis=-1)
            add_to_confusion_matrix(cm_total, pred_argmax, gt_e, N_CLASSES)
            if all:
                all_preds.append(pred_argmax)
                all_gts.append(gt_e)
            try:
                clear_output(wait=False)
            except Exception:
                pass

    eval_ids = list(range(N_CLASSES)) if DATASET == 'Hunan' else EVAL_CLASS_IDS
    MIoU, OA = metrics_from_cm(cm_total, LABELS, eval_ids)
    if all:
        return MIoU, OA, all_preds, all_gts, cm_total
    return MIoU, OA



def write_summary(stop_epoch, best_epoch, best_path, train_time_s, val_time_s,
                  early_stopped, history):
    """Write one JSON summary for the run."""
    summary = dict(
        dataset=DATASET, seed=SEED, model=MODEL,
        epochs_budget=epochs, stop_epoch=stop_epoch, best_epoch=best_epoch,
        best_val_mIoU=round(best_mIoU, 6),
        early_stopped=early_stopped,
        patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA,
        train_time_s=round(train_time_s, 1), val_time_s=round(val_time_s, 1),
        test_stride=TEST_STRIDE, val_stride=VAL_STRIDE,
        batch_size=BATCH_SIZE, window_size=list(WINDOW_SIZE),
        best_ckpt=os.path.basename(best_path) if best_path else None,
        val_history=history,
    )
    path = os.path.join(CKPT_DIR, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {path}")


def _set_alloc_active(net, active):
    """Enable/disable the second-order allocation gates (warmup support)."""
    n = 0
    for m in net.modules():
        if getattr(m, 'alloc_coupling', False):
            m.alloc_active = active
            n += 1
    return n


def train(epochs, save_epoch=1):
    global best_mIoU
    weights = WEIGHTS.cuda()
    iter_ = start_iter
    init_log_csv()

    best_epoch, best_path, epochs_since_improve = restore_best_selection(CKPT_DIR, start_epoch, MODEL)
    total_train_time = 0.0
    total_val_time = 0.0
    history = []
    early_stopped = False
    # Restore the validation history when resuming from a checkpoint.
    if start_epoch > 1 and os.path.isfile(LOG_CSV):
        with open(LOG_CSV) as f:
            for row in csv.DictReader(f):
                if row.get('val_mIoU') and int(row['epoch']) < start_epoch:
                    history.append([int(row['epoch']), float(row['lr']),
                                    float(row['train_loss']), float(row['val_mIoU']),
                                    float(row['val_OA'] or 0)])

    for e in range(start_epoch, epochs + 1):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        warmup_ep = int(os.environ.get('MFNET_WARMUP_EPOCHS', '0'))
        if warmup_ep and (e - start_epoch + 1) <= warmup_ep:
            ramp = max(e - start_epoch + 1, 1) / warmup_ep
            for pg in optimizer.param_groups:
                pg['lr'] = lr * ramp
            lr = optimizer.param_groups[0]['lr']
        if ALLOC_WARMUP:
            _set_alloc_active(net, active=(e > ALLOC_WARMUP))
        net.train()
        t0 = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0

        for batch_idx, batch in enumerate(train_loader):
            *modalities, target = batch
            modalities = [Variable(m.cuda()) for m in modalities]
            target = Variable(target.cuda())
            optimizer.zero_grad()

            output = net(modalities, mode='Train')
            dice_w = DICE_WEIGHT if 'DICE_WEIGHT' in globals() else 0.5
            seg_loss = loss_calc_combined(output, target, weights, dice_weight=dice_w)
            total_loss = seg_loss

            # Supervise each modality head regardless of which output receives the main loss.
            if os.environ.get('MFNET_XGA_PHL') == '1':
                lg = getattr(net, '_xga_lg', None)
                if lg is not None:
                    for lm in lg:
                        lm_r = F.interpolate(lm, size=target.shape[-2:],
                                             mode='bilinear', align_corners=False)
                        total_loss = total_loss + loss_calc_combined(
                            lm_r, target, weights, dice_weight=dice_w)

            # At the selected encoder depths, supervise the shared class readout.
            if os.environ.get('MFNET_EVD') == '1' and \
                    os.environ.get('MFNET_EVD_EDL') == '1' and \
                    os.environ.get('MFNET_EVD_NOSUP') != '1':
                for (e0t, e1t) in getattr(net, '_evd_e_taps', None) or []:
                    if e0t.dim() == 4:
                        e0t = e0t.flatten(1, 2)
                        e1t = e1t.flatten(1, 2)
                    if e0t.dim() != 3:
                        continue
                    n_tok = e0t.shape[1]
                    g_ = int(n_tok ** 0.5)
                    npf = n_tok - g_ * g_
                    s_ = g_
                    t_tok = F.interpolate(target.float().unsqueeze(1),
                                          size=(s_, s_),
                                          mode='nearest').squeeze(1).long()
                    v = (t_tok < e0t.shape[-1]) & (t_tok >= 0)
                    for em in (e0t, e1t):
                        em2 = em[:, npf:, :].reshape(-1, s_, s_,
                                                     em.shape[-1]).permute(0, 3, 1, 2).contiguous()
                        # Compute cross-entropy from normalized class concentrations.
                        alpha = em2 + 1.0
                        S_a = alpha.sum(1, keepdim=True)
                        ce = torch.log(S_a.squeeze(1)) - torch.log(
                            alpha.gather(1, t_tok.clone().clamp(min=0)[:, None]).squeeze(1))
                        total_loss = total_loss + \
                            ((ce * v).sum() / v.sum().clamp_min(1)) * float(os.environ.get('MFNET_READOUT_AUX_WEIGHT', '0.25'))
                lg = getattr(net, '_xga_lg', None)
                ag = getattr(net, '_xga_ag', None)
                if ag is not None and lg is not None \
                        and os.environ.get('MFNET_ALPHA_FIX') != '1':
                    l0, l1 = lg
                    hs = l0.shape[-2:]
                    p0 = F.interpolate(l0, size=hs, mode='bilinear').softmax(1)
                    p1 = F.interpolate(l1, size=hs, mode='bilinear').softmax(1)
                    t_h = F.interpolate(target.float().unsqueeze(1),
                                        size=hs, mode='nearest').squeeze(1).long()
                    v = (t_h < p0.shape[1]) & (t_h >= 0)
                    lp0 = F.interpolate(l0, size=hs, mode='bilinear'
                                        ).log_softmax(1).gather(
                                            1, t_h.clamp(min=0)[:, None])
                    lp1 = F.interpolate(l1, size=hs, mode='bilinear'
                                        ).log_softmax(1).gather(
                                            1, t_h.clamp(min=0)[:, None])
                    conf = (p0 - p1).abs().sum(1, keepdim=True)        # [B,1,h,w]
                    thr = torch.quantile(conf.flatten(), 0.75)
                    m_ = (conf >= thr) & v[:, None]
                    a_star = torch.sigmoid(4.0 * (lp1 - lp0))
                    if m_.any():
                        total_loss = total_loss + (
                            (ag - a_star).abs() * m_).sum() / m_.sum()

            aux_scale = AUX_SCALE if 'AUX_SCALE' in globals() else 0.0
            kl_weight = KL_WEIGHT if 'KL_WEIGHT' in globals() else 0.01
            orth_val = torch.tensor(0.0)
            kl_val   = torch.tensor(0.0)

            if hasattr(net, 'get_orthogonal_loss'):
                ov = net.get_orthogonal_loss()
                if ov is not None and ov.numel() > 0:
                    orth_val = ov
                    total_loss = total_loss + aux_scale * ov

            if hasattr(net, 'get_kl_loss'):
                kv = net.get_kl_loss()
                if kv is not None and kv.numel() > 0:
                    kl_val = kv
                    total_loss = total_loss + kl_weight * kv

            total_loss.backward()
            optimizer.step()

            losses[iter_] = total_loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            epoch_loss_sum += total_loss.item()
            epoch_loss_n += 1

            if iter_ % 100 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt   = target.data.cpu().numpy()[0]
                phi_str = ''
                with torch.no_grad():
                    phis = []
                    for blk in net.image_encoder.blocks:
                        p = blk.get_phi() if hasattr(blk, 'get_phi') else None
                        if p is not None:
                            phis.append(p.flatten())
                    if phis:
                        phi_all = torch.cat(phis)
                        phi_str = (f'Phi[mean={phi_all.mean():.3f} '
                                   f'std={phi_all.std():.3f} '
                                   f'p10={phi_all.quantile(0.1):.3f} '
                                   f'p90={phi_all.quantile(0.9):.3f}]')
                print(f'Train (epoch {e}/{epochs}) [{batch_idx}/{len(train_loader)} '
                      f'({100.*batch_idx/len(train_loader):.0f}%)]\t'
                      f'Loss: {total_loss.data:.6f}\t'
                      f'KL: {kl_val.item():.4f}\tOrth: {orth_val.item():.4f}\t'
                      f'Acc: {accuracy(pred, gt)}\t{phi_str}')
            iter_ += 1
            del modalities, target, total_loss

        train_time = time.time() - t0
        avg_loss = epoch_loss_sum / max(epoch_loss_n, 1)
        total_train_time += train_time

        val_mIoU = None
        val_OA = None
        if e % save_epoch == 0:
            print(f"Epoch {e} training: {train_time:.1f}s")

            net.eval()

            if hasattr(net, 'per_class_modality_contribution'):
                try:
                    sample = next(iter(train_loader))
                    *mods_s, tgt_s = sample
                    with torch.no_grad():
                        net([Variable(m.cuda()) for m in mods_s], mode='Train')
                    contrib = net.per_class_modality_contribution(tgt_s, N_CLASSES)
                    if contrib:
                        parts = [f"{LABELS[c]}: {[f'{v:.3f}' for v in contrib.get(c, [])]}"
                                 for c in range(N_CLASSES)]
                        print(f"  Modality contrib: {', '.join(parts)}")
                except:
                    pass

            val_stride = VAL_STRIDE if 'VAL_STRIDE' in globals() else Stride_Size
            t1 = time.time()
            _val_ids = val_ids if ('val_ids' in globals()) else test_ids
            MIoU, val_OA = test(net, _val_ids, all=False, stride=val_stride)
            val_mIoU = MIoU
            val_time = time.time() - t1
            total_val_time += val_time
            print(f"  Validation: {val_time:.1f}s")

            history.append([e, round(lr, 6), round(avg_loss, 6), round(MIoU, 6), round(val_OA, 6)])

            if MIoU > best_mIoU + EARLY_STOP_MIN_DELTA:
                improve = MIoU - best_mIoU
                best_mIoU = MIoU
                best_epoch = e
                epochs_since_improve = 0
                best_path = f'{CKPT_DIR}/{MODEL}_best_epoch{e}_mIoU{MIoU:.4f}.pth'
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(net.state_dict(), best_path)
                for old in os.listdir(CKPT_DIR):
                    if old.startswith(f'{MODEL}_best_') and old != os.path.basename(best_path):
                        os.remove(os.path.join(CKPT_DIR, old))
                print(f"  ★ New best (epoch {e}, +{improve:.4f}) → {best_path}")
            else:
                epochs_since_improve += 1
                print(f"  No improvement for {epochs_since_improve}/{EARLY_STOP_PATIENCE} "
                      f"validation checks (best {best_mIoU:.4f} @ epoch {best_epoch})")

            preserve_selection(CKPT_DIR, e, best_epoch, best_mIoU, best_path)

            resume_path = f'{CKPT_DIR}/{MODEL}_checkpoint_latest.pth'
            save_full_checkpoint(e, iter_, best_mIoU, resume_path)

        log_epoch(e, lr, avg_loss, train_time, val_mIoU, val_OA, best_mIoU)

        if e % save_epoch == 0 and epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f"\n⏹ Early stopping at epoch {e}: no >{EARLY_STOP_MIN_DELTA} mIoU "
                  f"improvement in {EARLY_STOP_PATIENCE} validation checks. "
                  f"Best {best_mIoU:.4f} @ epoch {best_epoch}.")
            early_stopped = True
            break

    if best_path is None:
        best_path = f'{CKPT_DIR}/{MODEL}_checkpoint_latest.pth'
        best_epoch = e
    write_summary(stop_epoch=e, best_epoch=best_epoch, best_path=best_path,
                  train_time_s=total_train_time, val_time_s=total_val_time,
                  early_stopped=early_stopped, history=history)
    print(f'\nBest mIoU: {best_mIoU:.4f} (epoch {best_epoch}, '
          f'{"early-stopped" if early_stopped else "budget exhausted"} @ {e}/{epochs})')



if MODE == 'Train':
    train(epochs, save_epoch=save_epoch)

elif MODE == 'Test':
    test_ckpt = None
    if 'TEST_CKPT' in globals() and TEST_CKPT:
        test_ckpt = TEST_CKPT

    if test_ckpt and os.path.isfile(test_ckpt):
        print(f"Loading test checkpoint: {test_ckpt}")
        net.load_state_dict(torch.load(test_ckpt, weights_only=False), strict=False)
    net.eval()

    out_dir = CKPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, s):
            for f in self.files:
                f.write(s)
        def flush(self):
            for f in self.files:
                f.flush()

    result_path = os.path.join(out_dir, f'{MODEL}_test_result.txt')
    with open(result_path, 'w', encoding='utf-8') as rf:
        rf.write(f'Model: {MODEL}\n')
        rf.write(f'Dataset: {DATASET}\n')
        rf.write(f'Seed: {SEED}\n')
        rf.write(f'Checkpoint: {test_ckpt if test_ckpt else "current net"}\n')
        rf.write(f'Test tiles: {len(test_ids)}\n')
        rf.write(f'Test stride: {TEST_STRIDE} (unified paper protocol)\n')
        rf.write('=' * 50 + '\n')

        tee = Tee(sys.stdout, rf)
        import contextlib
        with contextlib.redirect_stdout(tee):
            MIoU, OA, all_preds, all_gts, cm = test(net, test_ids, all=True, stride=TEST_STRIDE)
            miou, oa, iou_pc, f1_pc, kappa = cm_metrics(cm)
            print(f'mIoU: {MIoU:.4f}')
            print(f'OA: {OA:.4f}')

        rf.write('=' * 50 + '\n')

    result = dict(
        dataset=DATASET, seed=SEED, model=MODEL,
        checkpoint=test_ckpt if test_ckpt else 'current net',
        test_stride=TEST_STRIDE, n_test_tiles=len(test_ids),
        test_mIoU=round(float(miou), 6), test_OA=round(float(oa), 6),
        kappa=round(float(kappa), 6),
        per_class_iou={LABELS[c]: round(float(iou_pc[c]), 6) for c in range(N_CLASSES)},
        per_class_f1={LABELS[c]: round(float(f1_pc[c]), 6) for c in range(N_CLASSES)},
    )
    with open(os.path.join(out_dir, 'test_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    for p, id_ in zip(all_preds, test_ids):
        io.imsave(f'{out_dir}/inference_tile_{id_}.png', convert_to_color(p))
    print(f"Inference results saved to {out_dir}/")
    print(f"Test report saved to {result_path}")
