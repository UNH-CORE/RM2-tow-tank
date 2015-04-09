# -*- coding: utf-8 -*-
"""
This script generates all the relevant figures from the experiment.

"""

from Modules.processing import *
from Modules.plotting import *

set_sns()
save = True
savetype = ".pdf"
show = True

def main():
    plot_perf_curves(save=save, savetype=savetype)
    plot_perf_re_dep(save=save, savetype=savetype, errorbars=True, 
                     dual_xaxes=True)
    PerfCurve(1.0).plotcp(save=save, savetype=savetype, show=False)
    wm = WakeMap()
    wm.plot_meancontquiv(save=save, savetype=savetype)
    wm.plot_k(save=save, savetype=savetype)
    plot_strut_torque(covers=False, save=save, savetype=savetype)
    plot_strut_torque(covers=True, save=save, savetype=savetype)
    plot_cp_no_blades(covers=False, save=save, savetype=savetype)
    plot_cp_no_blades(covers=True, save=save, savetype=savetype)
    plot_cp_covers(save=save, savetype=savetype, add_strut_torque=False)
    plot_cp_covers(save=save, savetype=savetype, add_strut_torque=True)
    if show:
        plt.show()

if __name__ == "__main__":
    if not os.path.isdir("Figures"):
        os.mkdir("Figures")
    main()