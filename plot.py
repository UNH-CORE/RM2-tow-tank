#!/usr/bin/env python
"""This script generates all the relevant figures from the experiment."""

from pyrm2tt.processing import *
from pyrm2tt.plotting import *
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create figures from the "
                                     "UNH RM2 tow tank experiment.")
    parser.add_argument("plots", nargs="*", help="Which plots to create",
                        choices=["perf_curves", "perf_re_dep", "cp_re_0",
                                 "meancontquiv", "kcont", "K_bar_chart",
                                 "perf_no_blades", "cp_covers", "perf_covers",
                                 "mom_bar_chart", "none"],
                        default="none")
    parser.add_argument("--all", "-a", action="store_true", default=False,
                        help="Plot all figures used in publication")
    parser.add_argument("--subplots", action="store_true", default=False,
                        help="Use subplots for performance curves")
    parser.add_argument("--no-errorbars", action="store_true",
                        help="Don't plot error bars on Re-dependence figure")
    parser.add_argument("--style", nargs=1, help="matplotlib stylesheet")
    parser.add_argument("--save", "-s", action="store_true", default=False,
                        help="Save figures to local directory")
    parser.add_argument("--savetype", "-t", help="Figure file format",
                        type=str, default=".eps")
    parser.add_argument("--no-show", action="store_true", default=False,
                        help="Do not show figures")
    args = parser.parse_args()
    save = args.save
    savetype = args.savetype
    errorbars = not args.no_errorbars

    if args.plots == "none" and not args.all:
        print("No plots selected")
        parser.print_help()
        sys.exit(2)

    if save:
        if not os.path.isdir("Figures"):
            os.mkdir("Figures")
        if savetype[0] != ".":
            savetype = "." + savetype

    if args.style is not None:
        plt.style.use(args.style)
    else:
        from pxl.styleplot import set_sns
        set_sns()
        plt.rcParams["axes.formatter.use_mathtext"] = True

    if "perf_curves" in args.plots or args.all:
        plot_perf_curves(subplots=args.subplots, save=save, savetype=savetype)
    if "perf_re_dep" in args.plots or args.all:
        plot_perf_re_dep(save=save, savetype=savetype, errorbars=errorbars,
                         dual_xaxes=True)
    if "cp_re_0" in args.plots:
        PerfCurve(1.0).plotcp(save=save, savetype=savetype)
    if ("meancontquiv" in args.plots or
        "kcont" in args.plots or
        "mom_bar_chart" in args.plots or
        "K_bar_chart" in args.plots or
        args.all):
        wm = WakeMap()
        if "meancontquiv" in args.plots or args.all:
            wm.plot_meancontquiv(save=save, savetype=savetype)
        if "kcont" in args.plots or args.all:
            wm.plot_k(save=save, savetype=savetype)
        if "mom_bar_chart" in args.plots or args.all:
            wm.make_mom_bar_graph(save=save, savetype=savetype)
        if "K_bar_chart" in args.plots or args.all:
            wm.make_K_bar_graph(save=save, savetype=savetype)
    if "perf_no_blades" in args.plots or args.all:
        plot_no_blades_all(save=save, savetype=savetype)
    if "cp_covers" in args.plots:
        plot_cp_covers(save=save, savetype=savetype, add_strut_torque=False)
        plot_cp_covers(save=save, savetype=savetype, add_strut_torque=True)
    if "perf_covers" in args.plots or args.all:
        plot_perf_covers(save=save, savetype=savetype, subplots=True)


    if not args.no_show:
        plt.show()
