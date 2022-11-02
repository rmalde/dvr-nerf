import argparse
import json 
import numpy as np
from os.path import join, dirname

TRANSFORMS_TRAIN = "transforms_train.json"
TRANSFORMS_TEST = "transforms_test.json"
TRANSFORMS_VAL = "transforms_val.json"

def loadJSON(f):
    with open(f) as f:
        return json.load(f)

def saveJSON(data, f):
    with open(f, 'w+') as f:
        json.dump(data, f)

# hold out for validation and test every $ and $ + 1 images
def splitJSON(data, split=10):
    N = len(data['frames'])
    all_ids = np.arange(N)
    print(all_ids)

    test_ids = all_ids[::split]
    print(test_ids)
    val_ids = all_ids[1::split]
    print(val_ids)
    train_ids = np.array([i for i in all_ids if i not in np.concatenate((test_ids, val_ids))])
    print(train_ids)

    frames = data['frames']
    # Make two dictionaries with empty frames
    data_train = data.copy()
    data_train['frames'] = [f for i, f in enumerate(frames) if i in train_ids]
    data_val = data.copy()
    data_val['frames'] = [f for i, f in enumerate(frames) if i in val_ids]
    data_test = data.copy()
    data_test['frames'] = [f for i, f in enumerate(frames) if i in test_ids]
    return data_train, data_val, data_test

def splitAndSaveTransforms(transformsDir, split=10):
    f_dir = dirname(transformsDir)
    f_train, f_val, f_test = join(f_dir, TRANSFORMS_TRAIN), join(f_dir, TRANSFORMS_VAL), join(f_dir, TRANSFORMS_TEST)
    data = loadJSON(transformsDir)
    data_train, data_val, data_test = splitJSON(data, split)
    saveJSON(data_train, f_train), saveJSON(data_val, f_val), saveJSON(data_test, f_test)


def parse_args():
    parser = argparse.ArgumentParser(description="Loads a single transforms.json and splits it into three files")

    parser.add_argument("--dir", default="data/BB86/colmap/transforms.json", help="input path of transforms.json")
    parser.add_argument('--hold', type=int, default=10, help="hold out for validation and test every $ and $ + 1 images")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    splitAndSaveTransforms(args.dir, args.hold)
