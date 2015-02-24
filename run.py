# -*- coding: utf-8 -*-
"""
This script generates all the relevant figures from the experiment.

"""

from Modules.processing import *
from Modules.plotting import *

save = True
savetype = ".pdf"

def main():
    plot_perf_curves(save=save, savetype=savetype)
    plot_perf_re_dep(save=save, savetype=savetype, errorbars=True, 
                     dual_xaxes=True)
    PerfCurve(1.0).plotcp(save=save, savetype=savetype)

if __name__ == "__main__":
    if not os.path.isdir("Figures"):
        os.mkdir("Figures")
    main()