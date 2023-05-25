# PyFastL2LiR_mtanaka

[PyFastL2LiR](https://github.com/KamitaniLab/PyFastL2LiR) より下記の点が変更されている．

- 各 unit に対して選択された voxel の index を保存する．
- 選択された voxel のみを用いて sample 単位の spatial normalization が実行可能．
- (branch relu_unitについて) 各 unit に対して使用する training sample の絞り込みが可能．

使用時には `bdpy_mtanaka` リポジトリの bdpy を使用する必要がある．（データ保存形式が通常のbdpyと違うので必須）

`bdpy_mtanaka`: https://github.com/KamitaniLab/bdpy_mtanaka


## License
[![GitHub license](https://img.shields.io/github/license/KamitaniLab/PyFastL2LiR)](https://github.com/KamitaniLab/PyFastL2LiR/blob/master/LICENSE)


## Requiarements
- Python >= 3.5
- bdpy_mtanaka
- threadpoolctl

## Usage

```
import sys
sys.path.insert(0, "./to_PyFasL2LiR_mtanaka")
sys.path.insert(0, "./to_bdpy_mtanaka")

import bdpy
print(bdpy.__file__)
import fastl2lir
print(fastl2lir.__file__)

from bdpy.ml import ModelTraining
from fastl2lir import FastL2LiR

# Training 
makedir_ifnot('./tmp')
distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

model = FastL2LiR()
model_param = {
    'alpha':  alpha,
    'n_feat': num_voxel[roi],
    'saveMemory': True, # 選択された voxel index を保存する
    'sample_norm': 'norm2mean0', # sample に spatial normalization を適用する
    'sample_selection': 'remove_nan', # training に使用する sample を基準に基づき選択する
}
train = ModelTraining(model, x, y)
train.id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
train.model_parameters = model_param
train.X_normalize = {'mean': x_mean,
                        'std': x_norm}
train.Y_normalize = {'mean': y_mean,
                        'std': y_norm}
train.Y_sort = {'index': y_index}

train.dtype = np.float32
train.chunk_axis = chunk_axis
train.save_format = 'bdmodel'
train.save_path = results_dir
train.distcomp = distcomp

train.run()

# Test
test = ModelTest(model, x)
test.model_format = 'bdmodel'
test.model_path = model_dir
test.model_parameters = {
    'sample_norm': 'norm2mean0',
    'use_feature_selector': True,
    'saveMemory': True, # Training と同じ値を選択
    'sample_norm': 'norm2mean0', # Training と同じ値を選択
    #'sample_selection': 'remove_nan', # sample_selectionは不要 
}
test.dtype = np.float32
test.chunk_axis = chunk_axis

y_pred = test.run()
```

