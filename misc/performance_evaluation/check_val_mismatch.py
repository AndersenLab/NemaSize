import os, re

old_path = r'C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\RF_name_mapping\name_mapping.txt'
new_path = r'C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\RF_name_mapping_new\image_key.tsv'
val_dir = r'C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyDetection.v10i.coco-segmentation\valid\images'

old_map, new_map = {}, {}
with open(old_path) as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            old_map[int(os.path.splitext(p[0])[0])] = os.path.splitext(p[1])[0]
with open(new_path) as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            new_map[int(os.path.splitext(p[1])[0])] = os.path.splitext(p[0])[0]

val_ids = []
for fn in os.listdir(val_dir):
    m = re.match(r'(\d+)_png', fn)
    if m:
        val_ids.append(int(m.group(1)))

mismatched = [i for i in val_ids if i in old_map and i in new_map and old_map[i] != new_map[i]]
print(f'Validation images: {len(val_ids)}')
print(f'Mismatched in validation: {len(mismatched)}')
print()
print(f'{"rf_id":>6}  {"OLD mapping":<42}  NEW mapping')
print('-' * 95)
for i in sorted(mismatched):
    print(f'{i:>6}  {old_map[i]:<42}  {new_map[i]}')
