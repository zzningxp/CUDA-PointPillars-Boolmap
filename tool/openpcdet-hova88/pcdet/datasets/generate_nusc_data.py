import pickle
from nbformat import read
import numpy
import random


path0 = '/home/ssd1T0/daodao/nuscenes/v1.0-trainval-ori/'
path = '/home/ssd1T0/daodao/nucustom/v1.0-trainval/'

def toOneClass(pkl):
    new_infos = []
    sub_dict_keys = ['lidar_path', 'gt_boxes', 'gt_names', 'token', 'sweeps', 'num_lidar_pts']
    replaceclass = ['truck', 'construction_vehicle', 'trailer', 'bus']
    f = open(path + pkl, 'rb')
    nusc_infos = pickle.load(f)
    print(f, type(nusc_infos), len(nusc_infos))
    cnt = 0
    truck_cnt = 0
    car_cnt = 0

    for info in nusc_infos[:]:
        gtl = info['gt_names']
        # print(gtl)
        if len(gtl) == 0:
            continue
        gtl = ['truck' if i in replaceclass else 'ignore' for i in gtl]
        gtl = numpy.asarray(gtl)
        # print('after', gtl)
        if 'truck' not in gtl:
            continue

        # print(info['lidar_path'])
        tmp_info = {key: value for key, value in info.items() if key in sub_dict_keys} 
        # tmp_info['token'] = ''
        tmp_info['gt_names'] = gtl
        # tmp_info['sweeps'] = []
        new_infos.append(tmp_info)

        cnt += 1
        tc = (gtl == 'truck').sum()
        truck_cnt += tc
        car_cnt += (gtl == 'car').sum()
    print(cnt, truck_cnt, car_cnt)
    print(len(new_infos))
    return new_infos

def cdtruck(tokens = []):
    subsets = ['cd1', 'cd2', 'cd3', 'cd1027_XT06_01', 'cd1027_XT06_02', 'cd1027_XT06_03', 'cd1027_XT06_04']
    new_infos = []
    ti = 0
    for i in subsets:
        f = open(path + 'samples-cdi/' + i + '.pkl', 'rb')
        nusc_infos = pickle.load(f)
        # print(len(nusc_infos))
        for info in nusc_infos[:]:
            # info['token'] = tokens[ti]
            info['lidar_path'] = 'samples-cdi/' + info['lidar_path']
            info['token'] = 'sample_token_' + info['lidar_path'].replace('/', '_').replace('-', '').replace('.', '_')
            # for j in range(len(info['gt_boxes'])):
            #     info['gt_boxes'][j][6] = 0
            # print(info['lidar_path'])
            # if len(info['gt_names']) == 0:
            #     continue
            ti += 1
            new_infos.append(info)
    print(len(new_infos))
    return new_infos

def readtokens():
    f = open(path0 + 'nuscenes_infos_10sweeps_val_ori.pkl', 'rb')
    nusc_infos = pickle.load(f)
    tokens = []
    for info in nusc_infos:
        # if 'fd8420396768425eabec9bdddf7e64b6' == info['token']:
        #     print(info)
        tokens.append(info['token'])
    random.shuffle(tokens)
    return tokens

def readinfos(pkl):
    f = open(path + pkl, 'rb')
    nusc_infos = pickle.load(f)
    random.shuffle(nusc_infos)
    print(len(nusc_infos))
    for info in nusc_infos[:2]:
        print(info)

# tk = readtokens()
wr_name = 'nuscenes_infos_.pkl'
infos_livox = toOneClass('samples-cdi/livox_simu.pkl')
infos_2 = cdtruck()
new_infos = infos_livox + infos_2 + infos_2
random.shuffle(new_infos)
print(len(new_infos))

f = open(path + wr_name , 'wb')
pickle.dump(new_infos, f)
print(path + wr_name + ' written')

readinfos(wr_name)

# readinfos('livoxsimu_3xcd_infos_1sweeps_train_c1_20089.pkl')