import sys
import re
import glob
import json

fs = glob.glob(sys.argv[1] + '/**/eval/**/metrics_summary.json', recursive=True)
# print(fs)
for i, f in enumerate(sorted(fs)):
    e = int(re.compile(r'epoch_(\d+)').search(f).group(1))
    x = json.load(open(f, 'r'))
    aps = x["mean_dist_aps"]
    if i == 0:
        print("epoch\tnd_s\tmap", end="")
        for c in sorted(aps.keys()):
            print("\t%s" % c[:6], end="") 
        print()
    print("%d\t%.3f\t%.3f" % (e, float(x["nd_score"]), float(x["mean_ap"])), end="")
    for c in sorted(aps.keys()):
        print("\t%.3f" % (float(aps[c])), end="")
    print()