import sys
import re

f = open(sys.argv[1]).readlines()
p0 = re.compile(r'EPOCH (\d+) EVALUATION')
p1 = re.compile(r'(\w+ \w+@0.\d+, 0.\d+, 0.\d+)')
p2 = re.compile(r'bbox AP:(\d+.\d+), (\d+.\d+), (\d+.\d+)')
p3 = re.compile(r'bev  AP:(\d+.\d+), (\d+.\d+), (\d+.\d+)')

epoch_num = -1
class_num = ''

bboxap = {}
bevap = {}
for l in f:
    m = p0.search(l)
    if m:
        epoch_num = m.group(1)
    m = p1.search(l)
    if m:
        class_num = m.group(1)
    m = p2.search(l)
    if m:
        if epoch_num not in bboxap:
            bboxap[epoch_num] = {}
        bboxap[epoch_num][class_num] = float(m.group(1))
    m = p3.search(l)
    if m:
        if epoch_num not in bevap:
            bevap[epoch_num] = {}
        bevap[epoch_num][class_num] = float(m.group(1))

if len(sys.argv) == 2 or sys.argv[2] == 'bbox':
    print('\tCar AP@0.7,0.7,0.7\tPed AP@0.5,0.5,0.5\tTruck AP@0.7,0.7,0.7')
    for e in bboxap.keys():
        v = bboxap[e]
        if 'Car AP@0.70, 0.70, 0.70' in v:
            x0 = v['Car AP@0.70, 0.70, 0.70']
        else:
            x0 = 0
        if 'Pedestrian AP@0.50, 0.50, 0.50' in v:
            x1 = v['Pedestrian AP@0.50, 0.50, 0.50']
        else:
            x1 = 0
        if 'Truck AP@0.70, 0.70, 0.70' in v:
            x2 = v['Truck AP@0.70, 0.70, 0.70']
        else:
            x2 = 0
        print('bbox\t%s\t%f\t%f\t%f\t%f' % (e, x0, x1, x2, x0+x1+x2))

if len(sys.argv) > 2 and sys.argv[2] == 'bev':
    print('\tCar AP@0.7,0.7,0.7\tPed AP@0.5,0.5,0.5\tTruck AP@0.7,0.7,0.7')
    for e in bevap.keys():
        v = bevap[e]
        x0 = v['Car AP@0.70, 0.70, 0.70']
        x1 = v['Pedestrian AP@0.50, 0.50, 0.50']
        x2 = v['Truck AP@0.70, 0.70, 0.70']
        print('bev\t%s\t%f\t%f\t%f\t%f' % (e, x0, x1, x2, x0+x1+x2))