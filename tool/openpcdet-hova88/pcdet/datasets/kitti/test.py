import pickle

path = '/home/ssd1T0/daodao/kitti/'
# with open(path + 'kitti_infos_train.pkl', 'rb') as f:
#     infos = pickle.load(f)
#     print(len(infos))

# with open(path + 'kitti_infos_trainval.pkl', 'rb') as f:
#     infos = pickle.load(f)
#     print(len(infos))

with open(path + 'kitti_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)
    print(len(infos))
    print(infos[10])

# with open(path + 'kitti_infos_test.pkl', 'rb') as f:
#     infos = pickle.load(f)
#     print(len(infos))
#     # print(infos[0])

# with open(path + 'kitti_dbinfos_train.pkl', 'rb') as f:
#     infos = pickle.load(f)
#     for k in infos.keys():
#         print(k, '\t', len(infos[k]))
#     # print(infos['bicycle'][:5])
