import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import itertools
from skimage import io
import os
from glob import glob

WINDOW_SIZE = (256, 256)

FOLDER = "/root/autodl-fs/"
BATCH_SIZE = 10

MODEL = 'UNetformer'
MODE = 'Test'
TEST_CKPT='autodl-tmp/resultsv/2/UNetformer_best_epoch14_mIoU0.8448.pth'

DATASET = 'Vaihingen'

if os.environ.get('MFNET_DATASET'):
    DATASET = os.environ['MFNET_DATASET']

ENCODER_TYPE = os.environ.get('MFNET_ENCODER_TYPE', 'dinov3')
ENCODER_SIZE = os.environ.get('MFNET_ENCODER_SIZE', 'large')
PRIVATE_LORA_RANK = 4
SHARED_LORA_RANK = 4
DECODER_TYPE = 'unetformer'

AUX_SCALE = 0.01
DICE_WEIGHT = 0.0
KL_WEIGHT = 0.01

BC_COUPLING = True

TTA_ENABLE = False
VAL_BATCH_SIZE = 16
VAL_STRIDE = 128

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True

palette = {0 : (255, 255, 255),
           1 : (0, 0, 255),
           2 : (0, 255, 255),
           3 : (0, 255, 0),
           4 : (255, 255, 0),
           5 : (255, 0, 0),
           6 : (0, 0, 0)}

invert_palette = {v: k for k, v in palette.items()}

if DATASET == 'Vaihingen':
    N_MODALITIES = 2
    MODALITY_NAMES = ['rgb', 'dsm']
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    Stride_Size = 32
    epochs = 50
    save_epoch = 1
    EVAL_CLASS_IDS = list(range(5))
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
    MODALITY_FOLDERS = [DATA_FOLDER, DSM_FOLDER]
elif DATASET == 'Potsdam':
    N_MODALITIES = 2
    MODALITY_NAMES = ['rgbir', 'dsm']
    train_ids = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
                '4_12', '6_8', '6_12', '6_7', '4_11']
    test_ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
    Stride_Size = 128
    VAL_STRIDE = 256
    epochs = 50
    save_epoch = 1
    EVAL_CLASS_IDS = list(range(5))
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    _DSM_FOLDER = MAIN_FOLDER + '1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg'
    def _pad_id(id):
        parts = id.split('_')
        return '{:02d}_{:02d}'.format(int(parts[0]), int(parts[1]))
    DSM_FOLDER = lambda id: _DSM_FOLDER.format(_pad_id(id))
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
    MODALITY_FOLDERS = [DATA_FOLDER, DSM_FOLDER]
elif DATASET == 'Hunan':
    N_MODALITIES = 2
    MODALITY_NAMES = ['rgb', 'dsm']
    train_ids = ['10434' ,'11524' ,'11607' ,'11724' ,'11854' ,'11856' ,'11919' ,'12152' ,'12350' ,'12563' ,'12669' ,'12813' ,'1302' ,'13258' ,'13383' ,'13524' ,'13565' ,'13932' ,'14477' ,'14694' ,'15001' ,'15023' ,'15201' ,'15230' ,'15548' ,'15603' ,'15686' ,'1599' ,'15998' ,'16090' ,'16217' ,'16541' ,'16582' ,'16703' ,'16709' ,'17092' ,'17269' ,'18186' ,'18950' ,'1899' ,'1906' ,'19098' ,'19680' ,'19915' ,'20175' ,'20386' ,'20561' ,'20734' ,'20752' ,'20759' ,'21562' ,'21565' ,'21738' ,'21820' ,'22000' ,'2232' ,'22547' ,'22729' ,'2431' ,'24718' ,'24733' ,'25069' ,'2584' ,'2617' ,'26622' ,'27163' ,'27308' ,'27312' ,'2791' ,'29393' ,'29886' ,'30560' ,'30719' ,'31175' ,'31188' ,'31586' ,'31797' ,'3817' ,'4529' ,'4530' ,'4889' ,'5223' ,'6213' ,'6597' ,'6600' ,'6768' ,'7329' ,'8232' ,'830' ,'8931' ,'944' ,'9956' ,'2087' ,'17721' ,'13990' ,'13622' ,'13563' ,'18009' ,'12148' ,'16888' ,'14758' ,'1773' ,'16516' ,'20408' ,'2070' ,'10062' ,'17637' ,'14942' ,'13931' ,'13410' ,'11959' ,'15150' ,'17582' ,'17820' ,'21545' ,'21563' ,'21592' ,'21922' ,'2255' ,'26628' ,'28533' ,'28801' ,'29621' ,'29796' ,'30482' ,'31302' ,'31355' ,'4171' ,'4887' ,'5994' ,'6167' ,'6777' ,'7421' ,'833' ,'9064' ,'9662' ,'14936' ,'15493' ,'2097' ,'25709' ,'30301' ,'18117' ,'11766' ,'3994' ,'5830' ,'14786' ,'1774' ,'16032' ,'2597' ,'18164' ,'8976' ,'2427' ,'418' ,'23961' ,'1165' ,'6383' ,'22906' ,'26032' ,'18371' ,'6156' ,'7167' ,'20736' ,'16880' ,'29145' ,'21211' ,'7473' ,'29172' ,'22077' ,'14755' ,'2428' ,'16922' ,'15144' ,'5232' ,'25777' ,'21736' ,'14290' ,'15275' ,'1025' ,'11173' ,'12040' ,'12779' ,'14126' ,'15695' ,'16214' ,'16577' ,'18079' ,'1930' ,'21804' ,'22154' ,'25699' ,'29675' ,'298' ,'31653' ,'5042' ,'637' ,'6581' ,'708' ,'7679' ,'1031' ,'11272' ,'14463' ,'16745' ,'12244' ,'1775' ,'1752' ,'16744' ,'17095' ,'20910' ,'13742' ,'16702' ,'13925' ,'17800' ,'17040' ,'2062' ,'16912' ,'19149' ,'11371' ,'21601' ,'21610' ,'21737' ,'21747' ,'21982' ,'23232' ,'2606' ,'27860' ,'28532' ,'28933' ,'29028' ,'29284' ,'29288' ,'30597' ,'3122' ,'31242' ,'4362' ,'6405' ,'6770' ,'726' ,'7330' ,'7331' ,'8234' ,'2402' ,'646' ,'13718' ,'12363' ,'13995' ,'13807' ,'1471' ,'27168' ,'18298' ,'16093' ,'15763' ,'12042' ,'29020' ,'8831' ,'11375' ,'23772' ,'12728' ,'13448' ,'27960' ,'14467' ,'14763' ,'19866' ,'13766' ,'24296' ,'1436' ,'5236' ,'28796' ,'10258' ,'28736' ,'2100' ,'1451' ,'12918' ,'14155' ,'15184' ,'19471' ,'21822' ,'22728' ,'22837' ,'2408' ,'3100' ,'5450' ,'6186' ,'7181' ,'9258' ,'11625' ,'11644' ,'12098' ,'12154' ,'12241' ,'13248' ,'13522' ,'13564' ,'1383' ,'13927' ,'14287' ,'14822' ,'15075' ,'15520' ,'15864' ,'16028' ,'1619' ,'16699' ,'16742' ,'16866' ,'17369' ,'17934' ,'18183' ,'18185' ,'18732' ,'1928' ,'19688' ,'20314' ,'20464' ,'20737' ,'21111' ,'21561' ,'21938' ,'22555' ,'23080' ,'23588' ,'23701' ,'2604' ,'27282' ,'27718' ,'28073' ,'28189' ,'29171' ,'29286' ,'29844' ,'30148' ,'30399' ,'30425' ,'30606' ,'31014' ,'31583' ,'31621' ,'4363' ,'5043' ,'5221' ,'5238' ,'6379' ,'6407' ,'6601' ,'9174' ,'2748' ,'29829' ,'10970' ,'20926' ,'6795' ,'24149' ,'18121' ,'20935' ,'942' ,'29391' ,'29638' ,'20054' ,'3161' ,'6772' ,'17933' ,'13535' ,'5412' ,'20599' ,'299' ,'19609' ,'452' ,'28191' ,'11659' ,'1450' ,'13019' ,'11838' ,'29892' ,'12151' ,'13933' ,'11568' ,'11233' ,'12153' ,'13433' ,'13436' ,'14105' ,'14169' ,'16724' ,'18895' ,'19853' ,'21981' ,'2246' ,'22907' ,'29654' ,'30669' ,'3534' ,'6794' ,'1057' ,'11271' ,'11603' ,'11678' ,'11957' ,'1262' ,'12863' ,'13109' ,'13299' ,'13415' ,'14603' ,'14873' ,'15398' ,'15596' ,'1605' ,'16089' ,'16530' ,'16870' ,'17799' ,'17943' ,'1932' ,'21740' ,'21967' ,'23342' ,'2396' ,'2397' ,'2429' ,'2542' ,'25778' ,'2618' ,'26776' ,'28038' ,'29104' ,'29567' ,'29733' ,'29779' ,'30276' ,'31708' ,'3936' ,'419' ,'5066' ,'7898' ,'20933' ,'15597' ,'18118' ,'16356' ,'14937' ,'10503' ,'13437' ,'14247' ,'21566' ,'22099' ,'22731' ,'27437' ,'28078' ,'29394' ,'29571' ,'30130' ,'6771' ,'940']
    test_ids = ['11767' ,'11816' ,'1239' ,'12626' ,'12815' ,'1290' ,'1303' ,'13254' ,'13257' ,'13515' ,'13765' ,'14108' ,'14293' ,'15426' ,'1625' ,'16373' ,'16750' ,'16890' ,'17039' ,'17055' ,'17107' ,'17455' ,'17821' ,'17980' ,'18958' ,'1908' ,'1923' ,'20028' ,'2106' ,'21385' ,'21423' ,'22312' ,'2256' ,'2265' ,'2442' ,'2603' ,'27264' ,'28965' ,'29287' ,'30275' ,'3280' ,'4017' ,'4166' ,'4886' ,'5215' ,'5233' ,'5410' ,'639' ,'7176' ,'11624']
    Stride_Size = 256
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = "/media/lscsc/nas/xianping/ISPRS_dataset/Hunan/"
    LABELS = ["cropland", "forest", "grassland", "wetland", "water", "unused land", "built-up area"]
    N_CLASSES = len(LABELS)
    EVAL_CLASS_IDS = list(range(N_CLASSES))
    WEIGHTS = torch.ones(N_CLASSES)
    DATA_FOLDER = MAIN_FOLDER + 'images_png/{}.png'
    DSM_FOLDER = MAIN_FOLDER + 'dsm_pngs/{}.png'
    LABEL_FOLDER = MAIN_FOLDER + 'masks_png/{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'masks_png/{}.tif'
    MODALITY_FOLDERS = [DATA_FOLDER, DSM_FOLDER]
    palette = {
        0: (196, 90, 17),
        1: (51, 129, 88),
        2: (177, 205, 61),
        3: (228, 84, 96),
        4: (91, 154, 214),
        5: (225, 174, 110),
        6: (239, 159, 2)}
    invert_palette = {v: k for k, v in palette.items()}
elif DATASET == 'WHU-OPT-SAR':
    N_MODALITIES = 2
    MODALITY_NAMES = ['optical', 'sar']
    LABELS = ["background", "cropland", "urban", "rural",
              "water", "forest", "road", "other"]
    N_CLASSES = len(LABELS)
    EVAL_CLASS_IDS = list(range(1, N_CLASSES))
    WEIGHTS = torch.ones(N_CLASSES)
    def _parse_division(path):
        train, test, section = [], [], None
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s == 'train:':
                    section = 'train'
                elif s == 'test:':
                    section = 'test'
                elif section == 'train':
                    train.append(s[:-4] if s.endswith('.tif') else s)
                elif section == 'test':
                    test.append(s[:-4] if s.endswith('.tif') else s)
        return train, test

    _div_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'division.txt')
    train_ids, test_ids = _parse_division(_div_path)
    Stride_Size = 256
    VAL_STRIDE = 256
    epochs = 50
    save_epoch = 1
    MAIN_FOLDER = '/root/autodl-tmp/whu-opt-sar/'
    OPT_FOLDER = MAIN_FOLDER + 'optical/{}.tif'
    SAR_FOLDER = MAIN_FOLDER + 'sar/{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'lbl/{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'lbl/{}.tif'
    MODALITY_FOLDERS = [OPT_FOLDER, SAR_FOLDER]
    palette = {
        0: (0, 0, 0),
        1: (204, 102, 0),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (0, 0, 255),
        5: (85, 167, 0),
        6: (0, 255, 255),
        7: (153, 102, 153),
    }
    invert_palette = {v: k for k, v in palette.items()}
    LABEL_VALUE_MAP = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}

print(MODEL + ', ' + MODE + ', ' + DATASET + ', WINDOW_SIZE: ', WINDOW_SIZE,
      ', BATCH_SIZE: ' + str(BATCH_SIZE), ', Stride_Size: ', str(Stride_Size),
      ', epochs: ' + str(epochs), ', save_epoch: ', str(save_epoch),)


TEST_STRIDE = 32

def _env_int(name, default=None):
    v = os.environ.get(name)
    return int(v) if v not in (None, '') else default

def _env_float(name, default=None):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else default

MODE = os.environ.get('MFNET_MODE', MODE)
if os.environ.get('MFNET_TEST_CKPT'):
    TEST_CKPT = os.environ['MFNET_TEST_CKPT']
epochs = _env_int('MFNET_EPOCHS', epochs)
VAL_STRIDE = _env_int('MFNET_VAL_STRIDE', VAL_STRIDE)

SEED = _env_int('MFNET_SEED', 42)
RUN_DIR = os.environ.get('MFNET_RUN_DIR') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'runs', DATASET, f'seed{SEED}')

EARLY_STOP_PATIENCE = _env_int('MFNET_PATIENCE', 20)
EARLY_STOP_MIN_DELTA = _env_float('MFNET_MIN_DELTA', 1e-4)

RESUME_PATH = os.environ.get('MFNET_RESUME')

ALLOC_WARMUP = _env_int('MFNET_ALLOC_WARMUP', 0)

def set_seed(seed=SEED):
    """Seed every RNG that influences training: decoder/LoRA init (torch),
    random crops & flips (python random), and any numpy draws."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print(f'  SEED: {SEED}  RUN_DIR: {RUN_DIR}  TEST_STRIDE: {TEST_STRIDE}  '
      f'EARLY_STOP: patience={EARLY_STOP_PATIENCE} min_delta={EARLY_STOP_MIN_DELTA}')

_BACKBONE_NORM = os.environ.get('MFNET_BACKBONE_NORM', 'none')
_NORM_STATS = {
    'sat': {'mean': (0.430, 0.411, 0.296), 'std': (0.213, 0.156, 0.143)},
    'web': {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
}
_NORM_ROLE_IDX = {
    'Vaihingen': {'R': 1, 'G': 2, 'B': None},
    'Potsdam':   {'R': 0, 'G': 1, 'B': 2},
    'WHU-OPT-SAR': {'R': 2, 'G': 1, 'B': 0},
}

# Apply encoder statistics in the dataset's native optical channel order.
def apply_backbone_norm(chw_float01):
    """[C,H,W] in [0,1] (native channel order) → standardized with the
    selected backbone stats; no-op when the switch is off."""
    if _BACKBONE_NORM not in _NORM_STATS or DATASET not in _NORM_ROLE_IDX:
        return chw_float01
    st = _NORM_STATS[_BACKBONE_NORM]
    roles = _NORM_ROLE_IDX[DATASET]
    out = chw_float01.copy()
    for slot, role in enumerate(('R', 'G', 'B')):
        idx = roles.get(role)
        if idx is not None:
            out[idx] = (out[idx] - st['mean'][slot]) / st['std'][slot]
    return out


def remap_label_values(arr, mapping):
    """Remap discrete label values (e.g. 0/10/20/.../70) to contiguous 0..N-1."""
    out = np.zeros(arr.shape, dtype=np.int64)
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=None, label_files=None,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache
        self.n_modalities = N_MODALITIES if 'N_MODALITIES' in globals() else 2

        if 'MODALITY_FOLDERS' in globals():
            self.modality_files = []
            for i in range(self.n_modalities):
                folder = MODALITY_FOLDERS[i]
                if callable(folder):
                    self.modality_files.append([folder(id) for id in ids])
                else:
                    self.modality_files.append([folder.format(id) for id in ids])
        else:
            def _resolve(folder, id):
                return folder(id) if callable(folder) else folder.format(id)
            self.modality_files = [
                [_resolve(DATA_FOLDER, id) for id in ids],
                [_resolve(DSM_FOLDER, id) for id in ids],
            ]

        self.label_files = [(LABEL_FOLDER(id) if callable(LABEL_FOLDER) else LABEL_FOLDER.format(id)) for id in ids]

        all_files = []
        for mf in self.modality_files:
            all_files.extend(mf)
        all_files.extend(self.label_files)
        for f in all_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        self.modality_cache_ = [{} for _ in range(self.n_modalities)]
        self.label_cache_ = {}

    def __len__(self):
        iters = os.environ.get('MFNET_ITERS')
        if iters:
            return BATCH_SIZE * int(iters)
        if DATASET == 'Potsdam' or DATASET == 'Vaihingen':
            return BATCH_SIZE * 1000
        elif DATASET == 'Hunan':
            return BATCH_SIZE * 500
        elif DATASET == 'WHU-OPT-SAR':
            return BATCH_SIZE * 1000
        else:
            return None

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.label_files) - 1)

        modality_tensors = []
        for mod_idx in range(self.n_modalities):
            if random_idx in self.modality_cache_[mod_idx]:
                mod_data = self.modality_cache_[mod_idx][random_idx]
            else:
                mod_path = self.modality_files[mod_idx][random_idx]
                mod_data_raw = io.imread(mod_path)

                if mod_data_raw.ndim == 3:
                    mod_data = mod_data_raw.transpose((2, 0, 1))
                    mod_data = 1 / 255 * np.asarray(mod_data, dtype='float32')
                    mod_data = apply_backbone_norm(mod_data)
                else:
                    mod_data = np.asarray(mod_data_raw, dtype='float32')
                    if DATASET == 'WHU-OPT-SAR':
                        mod_data = mod_data / 255.0
                    else:
                        vmin, vmax = np.min(mod_data), np.max(mod_data)
                        mod_data = (mod_data - vmin) / (vmax - vmin + 1e-8)

                if self.cache:
                    self.modality_cache_[mod_idx][random_idx] = mod_data

            modality_tensors.append(mod_data)

        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            if DATASET == 'Hunan':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64')
            elif DATASET == 'WHU-OPT-SAR':
                raw = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64')
                label = remap_label_values(raw, LABEL_VALUE_MAP)
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        if DATASET == 'Hunan':
            mod_patches = modality_tensors
            label_p = label
        else:
            x1, x2, y1, y2 = get_random_pos(modality_tensors[0], WINDOW_SIZE)
            mod_patches = []
            for mod in modality_tensors:
                if mod.ndim == 3:
                    mod_patches.append(mod[:, x1:x2, y1:y2])
                else:
                    mod_patches.append(mod[x1:x2, y1:y2])
            label_p = label[x1:x2, y1:y2]

        aug_inputs = mod_patches + [label_p]
        aug_results = self.data_augmentation(*aug_inputs)
        mod_patches = list(aug_results[:-1])
        label_p = aug_results[-1]

        return tuple([torch.from_numpy(mp) for mp in mod_patches] + [torch.from_numpy(label_p)])

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss

def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()

    return criterion(pred, label, weights)


def dice_loss(pred, target, smooth=1.0):
    """Soft Dice loss for multi-class segmentation."""
    pred_soft = F.softmax(pred, dim=1)
    target_oh = F.one_hot(target.long(), num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred_soft * target_oh).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def loss_calc_combined(pred, label, weights, dice_weight=0.5):
    """Combined segmentation loss: (1 - dice_weight) * CE + dice_weight * Dice."""
    ce = loss_calc(pred, label, weights)
    dice = dice_loss(pred, label)
    return (1.0 - dice_weight) * ce + dice_weight * dice


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics_from_cm(cm, label_values=LABELS, eval_ids=None):
    """Compute Acc/F1/Kappa/mIoU from an (already accumulated) confusion matrix.

    eval_ids: which class indices to average for mean F1 / mean mIoU.
              Defaults to all classes.
    """
    if eval_ids is None:
        eval_ids = list(range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    total = int(cm.sum())
    accuracy = float(np.trace(cm)) * 100.0 / max(total, 1)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        denom = np.sum(cm[i, :]) + np.sum(cm[:, i])
        F1Score[i] = 2. * cm[i, i] / denom if denom > 0 else 0.0
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[eval_ids])))
    print("---")

    total = int(cm.sum())
    pa = np.trace(cm) / float(total) if total > 0 else 0.0
    pe = (np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)) if total > 0 else 0.0
    kappa = (pa - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 0.0
    print("Kappa: %.4f" % (kappa))

    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[eval_ids])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU, accuracy  # (mIoU, OA)


def load_test_modality(id, mod_idx):
    """Read + normalize one modality of one test tile (streaming, no preload).

    Shared by training and evaluation so training-time and paper-table
    inference always see identical preprocessing.
    """
    folder = MODALITY_FOLDERS[mod_idx] if 'MODALITY_FOLDERS' in globals() else (DATA_FOLDER if mod_idx == 0 else DSM_FOLDER)
    path = folder(id) if callable(folder) else folder.format(id)
    raw = io.imread(path)
    if raw.ndim == 3:
        data = 1 / 255 * np.asarray(raw, dtype='float32')
        data = apply_backbone_norm(data.transpose(2, 0, 1)).transpose(1, 2, 0)
    else:
        data = np.asarray(raw, dtype='float32')
    if data.ndim == 2:
        if DATASET == 'WHU-OPT-SAR':
            data = data / 255.0
        else:
            vmin, vmax = np.min(data), np.max(data)
            data = (data - vmin) / (vmax - vmin + 1e-8)
    return data


def load_eroded_label(id):
    """Load the boundary-eroded ground truth of one tile as an index map."""
    _resolve = lambda f, id_: f(id_) if callable(f) else f.format(id_)
    if DATASET == 'Hunan':
        return np.asarray(io.imread(_resolve(ERODED_FOLDER, id)), dtype='int64')
    elif DATASET == 'WHU-OPT-SAR':
        return remap_label_values(np.asarray(io.imread(_resolve(ERODED_FOLDER, id)), dtype='int64'), LABEL_VALUE_MAP)
    else:
        return convert_from_color(io.imread(_resolve(ERODED_FOLDER, id)))


def cm_metrics(cm, eval_ids=None):
    """Summary metrics from a confusion matrix: (mIoU, OA, per-class IoU, per-class F1, kappa)."""
    if eval_ids is None:
        eval_ids = EVAL_CLASS_IDS if 'EVAL_CLASS_IDS' in globals() else list(range(len(LABELS)))
    total = int(cm.sum())
    oa = float(np.trace(cm)) * 100.0 / max(total, 1)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    miou = float(np.nanmean(iou[eval_ids]))
    f1 = np.zeros(cm.shape[0])
    for i in range(cm.shape[0]):
        denom = cm[i, :].sum() + cm[:, i].sum()
        f1[i] = 2. * cm[i, i] / denom if denom > 0 else 0.0
    pa = np.trace(cm) / float(total) if total > 0 else 0.0
    pe = (np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / float(total * total)) if total > 0 else 0.0
    kappa = (pa - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else 0.0
    return miou, oa, iou, f1, kappa


def add_to_confusion_matrix(cm, pred, gt, n_classes):
    """Accumulate one tile's predictions into cm (avoids concatenating all pixels).

    cm is mutated in place. pred/gt are 2D arrays of the same shape.
    """
    cm += confusion_matrix(gt.ravel(), pred.ravel(), labels=range(n_classes))
    return cm
