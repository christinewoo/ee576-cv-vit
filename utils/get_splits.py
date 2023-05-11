import os
import os.path as osp

eval_txt_path = '/home/xtreme/runs/data/celeba/eval/list_eval_partition.txt'

with open(eval_txt_path) as f:
    lines = f.readlines() #000001.jpg 0
    f.close()

train = []
val = []
test = []

for frame in lines:
    fname, id = frame[:-1].split(' ')
    if id == '0':
        train.append(fname)
    elif id == '1':
        val.append(fname)
    else:
        test.append(fname)

save_path = '/home/xtreme/runs/ee576-cv-vit/splits'
with open(osp.join(save_path, 'train.txt'), 'w') as ftr:
    for i in train:
        ftr.write(i + '\n')
    ftr.close()
with open(osp.join(save_path, 'val.txt'), 'w') as fv:
    for i in val:
        fv.write(i + '\n')
    fv.close()
with open(osp.join(save_path, 'test.txt'), 'w') as ft:
    for i in test:
        ft.write(i + '\n')
    ft.close()


print(f'Train: {len(train)}')
print(f'Val: {len(val)}')
print(f'Test: {len(test)}')