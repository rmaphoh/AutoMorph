#!/usr/bin/env python3

from numpy import arange

values = {}

for r2t in arange(0.4, 0.99, 0.01):
    window_values = {}
    for w in range(8, 121, 8):
        window_values[str(w) + "min"] = 200
        window_values[str(w) + "max"] = 0
    values["{:.2f}".format(r2t)] = window_values


content = open("linear_complete.csv")
for line in content:
    image_id, w, _, _, r2t, tor = line.split(',')
    w = int(w)
    w = str(w)
    tor = float(tor)
    if values[r2t][w + "min"] > tor:
        values[r2t][w + "min"] = tor
        values[r2t][w + "min_id"] = image_id
    if values[r2t][w + "max"] < tor:
        values[r2t][w + "max"] = tor
        values[r2t][w + "max_id"] = image_id

def get_r2t(type, thres):
    if values["0.98"][str(w)+type] == thres:
        if values['0.97'][str(w)+type] == thres:
            if values['0.96'][str(w)+type] == thres:
                if values['0.95'][str(w)+type] == thres:
                    return 0
                return '0.95'
            return '0.96'
        return '0.97'
    return '0.98'

print("##  MIN")
for w in range(8, 121, 8):
    r2t = get_r2t("min", 200)
    w_str = str(w)
    if r2t:
        print("{},{},{},{:.2f}".format(values[r2t][w_str + "min_id"], w_str, r2t, values[r2t][w_str + "min"]))
    else:
        print("r2t not found: {}".format(r2t))
print("## MAX")
for w in range(8, 121, 8):
    r2t = get_r2t("max", 0)
    w_str = str(w)
    if r2t:
        print("{},{},{},{:.2f}".format(values[r2t][w_str + "max_id"], w_str, r2t, values[r2t][w_str + "max"]))
    else:
        print("r2t not found: {}".format(r2t))
