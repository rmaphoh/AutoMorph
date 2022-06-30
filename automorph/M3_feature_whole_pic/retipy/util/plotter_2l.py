#!/usr/bin/env python3

import argparse
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f1", "--file1", help="csv file to read")
parser.add_argument("-f2", "--file2", help="csv file to read")
parser.add_argument("-c1", help="column 1")
parser.add_argument("-c2", help="column 2")
parser.add_argument("-c3", help="column 3")

args = parser.parse_args()

def figsize(scale):
    fig_width_pt = 469.755                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.8),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

DATA1 = np.genfromtxt(args.file1, delimiter=',', names=['1', '2'])
DATA2 = np.genfromtxt(args.file2, delimiter=',', names=['1', '2'])

FIG = plt.figure()

ax1 = FIG.add_subplot(111)
ax1.grid(True)

ax1.plot(DATA1['1'], DATA1['2'], label="Eroded")
ax1.plot(DATA2['1'], DATA2['2'], label="Original")
ax1.legend(loc='best')

ax1.set_xlabel('Image')
ax1.set_ylabel('Fractal Dimension')
ax1.set_xticks(range(1, 21, 1))

plt.show()
