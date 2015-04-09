# -*- coding: utf-8 -*-
"""
This module contains classes and functions for plotting data.

"""
from __future__ import division, print_function
from Modules.processing import *
import seaborn as sns
import os

def set_sns():
    sns.set(style="white", context="paper", font_scale=1.75,
            rc={"lines.markersize": 9, "lines.markeredgewidth": 1.25,
            "legend.fontsize": "small", "font.size": 14})

ylabels = {"mean_u" : r"$U/U_\infty$",
           "std_u" : r"$\sigma_u/U_\infty$",
           "mean_v" : r"$V/U_\infty$",
           "mean_w" : r"$W/U_\infty$",
           "mean_upvp" : r"$\overline{u'v'}/U_\infty^2$",
           "mean_u_diff" : r"$\Delta U$ (\%)",
           "mean_v_diff" : r"$\Delta V$ (\%)",
           "mean_w_diff" : r"$\Delta W$ (\%)"}
           
           
class PerfCurve(object):
    """Object that represents a performance curve."""
    def __init__(self, tow_speed):
        self.tow_speed = tow_speed
        self.Re_D = tow_speed*D/nu
        self.section = "Perf-{}".format(tow_speed)
        self.raw_data_dir = os.path.join("Data", "Raw", self.section)
        fpath = os.path.join("Data", "Processed", self.section+".csv")
        fpath_b = os.path.join("Data", "Processed", self.section+"-b.csv")
        self.df = pd.read_csv(fpath)
        try:
            self.df = self.df.append(pd.read_csv(fpath_b), ignore_index=True)
            self.df = self.df.groupby("tsr_nom").mean()
        except IOError:
            pass
        fpath_tp = os.path.join("Config", "Test plan", self.section+".csv")
        self.testplan = pd.read_csv(fpath_tp) 
        self.df = self.df[self.df.std_tow_speed < 0.009]
        self.df = self.df[self.df.mean_tsr <= 5.1]
        self.label = r"$Re_D = {:.1f} \times 10^6$".format(self.Re_D/1e6)
        
    def plotcp(self, newfig=True, show=True, save=False, savedir="Figures",
               savetype=".pdf", splinefit=False, marker="o"):
        """Generates power coefficient curve plot."""
        self.tsr = self.df.mean_tsr
        self.cp = self.df.mean_cp
        if newfig:
            plt.figure()
        if splinefit and not True in np.isnan(self.tsr):
            plt.plot(self.tsr, self.cp, marker+"k", markerfacecolor="None", 
                     label=self.label)
            plt.hold(True)
            tsr_fit = np.linspace(np.min(self.tsr), np.max(self.tsr), 200)
            tck = interpolate.splrep(self.tsr[::-1], self.cp[::-1], s=1e-3)
            cp_fit = interpolate.splev(tsr_fit, tck)
            plt.plot(tsr_fit, cp_fit, "k")
        else:
            if splinefit:
                print("Cannot fit spline. NaN present in array.")
            plt.plot(self.tsr, self.cp, "-"+marker+"k", markerfacecolor="None",
                     label=self.label)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$C_P$")
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(savedir, 
                    "cp_vs_tsr_{}".format(self.tow_speed) + savetype))
        if show:
            plt.show()
            
    def plotcd(self, newfig=True, show=True, save=False, savedir="Figures",
               savetype=".pdf", splinefit=False, marker="o"):
        """Generates power coefficient curve plot."""
        self.tsr = self.df.mean_tsr
        self.cd = self.df.mean_cd
        if newfig:
            plt.figure()
        if splinefit and not True in np.isnan(self.tsr):
            plt.plot(self.tsr, self.cd, marker+"k", markerfacecolor="None", 
                     label=self.label)
            plt.hold(True)
            tsr_fit = np.linspace(np.min(self.tsr), np.max(self.tsr), 200)
            tck = interpolate.splrep(self.tsr[::-1], self.cd[::-1], s=1e-3)
            cd_fit = interpolate.splev(tsr_fit, tck)
            plt.plot(tsr_fit, cd_fit, "k")
        else:
            if splinefit:
                print("Cannot fit spline. NaN present in array.")
            plt.plot(self.tsr, self.cd, "-"+marker+"k", markerfacecolor="None",
                     label=self.label)
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"$C_D$")
        plt.ylim((0, 1.2))
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(savedir, 
                    "cd_vs_tsr_{}".format(self.tow_speed) + savetype))
        if show:
            plt.show()
        
class WakeProfile(object):
    def __init__(self, tow_speed, z_H, orientation="horizontal"):
        self.tow_speed = tow_speed
        self.z_H = z_H
        self.section = "Wake-{}-{}".format(tow_speed, z_H)
        fpath = os.path.join("Config", "Test plan", self.section+".csv")
        self.testplan = pd.read_csv(fpath)
        self.runs = self.testplan.run
        fpath = os.path.join("Data", "Processed", self.section+".csv")
        self.df = pd.read_csv(fpath)
        self.y_R = self.df["y_R"].copy()
        self.Re_D = self.tow_speed*D/nu
        self.Re_label = r"$Re_D = {:.1f} \times 10^6$".format(self.Re_D/1e6)
        
    def plot(self, quantity, fmt="", newfig=True, show=True, save=False, 
             savedir="Figures", savetype=".pdf", preliminary=False,
             legend=False):
        """Plots some quantity"""
        y_R = self.df["y_R"].copy()
        q = self.df[quantity].copy()
        loc = 1
        if quantity == "mean_u":
            q /= self.df.mean_tow_speed
            ylab = r"$U/U_\infty$"
            loc = 3
        elif quantity == "mean_v":
            q /= self.df.mean_tow_speed
            ylab = r"$V/U_\infty$"
            loc=4
        elif quantity == "mean_w":
            q /= self.df.mean_tow_speed
            ylab = r"$U/U_\infty$"
            loc = 4
        elif quantity == "std_u":
            q /= self.df.mean_tow_speed
            ylab = r"$\sigma_u/U_\infty$"
        elif quantity is "mean_upvp":
            q /= (self.df.mean_tow_speed**2)
            ylab = r"$\overline{u'v'}/U_\infty^2$" 
        if newfig:
            plt.figure()
            plt.ylabel(ylab)
            plt.xlabel(r"$y/R$")
        plt.plot(y_R, q, fmt, label=self.Re_label)
        if legend:
            plt.legend(loc=loc)
        plt.tight_layout()
        if preliminary:
            watermark()
        if show:
            plt.show()
        if save:
            plt.savefig(savedir+quantity+"_Re_dep_exp"+savetype)
            
    
class WakeMap(object):
    def __init__(self):
        self.U_infty = 1.0
        self.z_H = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
        self.loaded = False
        self.load()
        
    def load(self):
        self.y_R = WakeProfile(self.U_infty, 0.0, "mean_u").y_R
        self.mean_u = np.zeros((len(self.z_H), len(self.y_R)))
        self.mean_v = self.mean_u.copy()
        self.mean_w = self.mean_u.copy()
        self.k = np.zeros((len(self.z_H), len(self.y_R)))
        for z_H in self.z_H:
            wp = WakeProfile(self.U_infty, z_H, "mean_u")
            self.mean_u[self.z_H.index(z_H)] = wp.df.mean_u
            self.mean_v[self.z_H.index(z_H)] = wp.df.mean_v
            self.mean_w[self.z_H.index(z_H)] = wp.df.mean_w
            self.k[self.z_H.index(z_H)] = wp.df.k
        self.loaded = True
        
    def turb_lines(self, linestyles="solid", linewidth=3, color="gray"):
        plt.hlines(0.5, -1, 1, linestyles=linestyles, colors=color,
                   linewidth=linewidth)
        plt.vlines(-1, -0.2, 0.5, linestyles=linestyles, colors=color,
                   linewidth=linewidth)
        plt.vlines(1, -0.2, 0.5, linestyles=linestyles, colors=color,
                   linewidth=linewidth)
        
    def plot_mean_u(self, save=False, show=False, savedir="Figures", 
                    savetype=".pdf"):
        """Plot contours of mean streamwise velocity."""
        plt.figure(figsize=(10,5))
        cs = plt.contourf(self.y_R, self.z_H, self.mean_u, 20,
                          cmap=plt.cm.coolwarm)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        cb = plt.colorbar(cs, shrink=1, extend="both", 
                          orientation="horizontal", pad=0.3)
        cb.set_label(r"$U/U_{\infty}$")
        self.turb_lines()
        ax = plt.axes()
        ax.set_aspect(2)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        plt.tight_layout()
        if save:
            plt.savefig(savedir+"/mean_u_cont"+savetype)
        if show:
            self.show()
    
    def plot_meancontquiv(self, save=False, show=False, savedir="Figures",
                          savetype=".pdf", cb_orientation="vertical"):
        """
        Plot contours of mean velocity and vector arrows showing mean
        cross-stream and vertical velocity.
        """
        plt.figure(figsize=(10, 2.75))
        # Add contours of mean velocity
        cs = plt.contourf(self.y_R, self.z_H, self.mean_u/self.U_infty, 20, 
                          cmap=plt.cm.coolwarm)
        if cb_orientation == "horizontal":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="horizontal", pad=0.14)
        elif cb_orientation == "vertical":
            cb = plt.colorbar(cs, shrink=0.88, extend="both", 
                              orientation="vertical", pad=0.02)
        cb.set_label(r"$U/U_{\infty}$")
        plt.hold(True)
        # Make quiver plot of v and w velocities
        Q = plt.quiver(self.y_R, self.z_H, self.mean_v/self.U_infty, 
                       self.mean_w/self.U_infty, width=0.0022,
                       edgecolor="none", scale=3)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        plt.ylim(-0.15, 0.85)
        plt.xlim(-1.68/R, 1.68/R)
        if cb_orientation == "horizontal":
            plt.quiverkey(Q, 0.65, 0.26, 0.1, r"$0.1 U_\infty$",
                          labelpos="E",
                          coordinates="figure",
                          fontproperties={"size": "small"})
        elif cb_orientation == "vertical":
            plt.quiverkey(Q, 0.65, 0.075, 0.1, r"$0.1 U_\infty$",
                          labelpos="E",
                          coordinates="figure",
                          fontproperties={"size": "small"})
        self.turb_lines()
        ax = plt.axes()
        ax.set_aspect(H/R)
        plt.yticks([0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75])
        plt.grid(True)
        plt.tight_layout()
        if show:
            self.show()
        if save:
            plt.savefig(savedir+"/meancontquiv"+savetype)
    
    def plot_xvorticity(self):
        pass
    
    def plot_diff(self, quantity="mean_u", U_infty_diff=1.0, save=False, 
                  show=False, savedir="Figures", savetype=""):
        wm_diff = WakeMap(U_infty_diff)
        q_ref, q_diff = None, None
        if quantity in ["mean_u", "mean_v", "mean_w"]:
            exec("q_ref = self." + quantity)
            exec("q_diff = wm_diff." + quantity)
            print(q_ref)
        else:
            print("Not a valid quantity")
            return None
        a_diff = (q_ref/self.U_infty - \
                  q_diff/wm_diff.U_infty)#/q_ref/self.U_infty*100
        plt.figure(figsize=(12,3.75))
        cs = plt.contourf(self.y_R, self.z_H, a_diff, 20,
                          cmap=plt.cm.coolwarm)
        cb = plt.colorbar(cs, shrink=1, fraction=0.15,
                          orientation="vertical", pad=0.05)
        cb.set_label(ylabels[quantity+"_diff"])
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        plt.axes().set_aspect(2)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        plt.tight_layout()
        if show:
            self.show()
        if save:
            if savedir: savedir += "/"
            plt.savefig(savedir+"/"+quantity+"_diff"+savetype)
    
    def plot_meancontquiv_diff(self, U_infty_diff, save=False, show=False,
                               savedir="Figures", savetype="", percent=True):
        wm_diff = WakeMap(U_infty_diff)
        mean_u_diff = (self.mean_u/self.U_infty - \
                wm_diff.mean_u/wm_diff.U_infty)
        mean_v_diff = (self.mean_v/self.U_infty - \
                wm_diff.mean_v/wm_diff.U_infty)
        mean_w_diff = (self.mean_w/self.U_infty - \
                wm_diff.mean_w/wm_diff.U_infty)
        if percent:
            mean_u_diff = mean_u_diff/self.mean_u/self.U_infty*100
            mean_v_diff = mean_v_diff/self.mean_v/self.U_infty*100
            mean_w_diff = mean_w_diff/self.mean_w/self.U_infty*100
        plt.figure(figsize=(12,4))
        cs = plt.contourf(self.y_R, self.z_H, mean_u_diff, 20,
                          cmap=plt.cm.coolwarm)
        cb = plt.colorbar(cs, shrink=1, fraction=0.15,
                          orientation="vertical", pad=0.05)
        cb.set_label(r"$\Delta U$ (\%)")
        plt.hold(True)
        # Make quiver plot of v and w velocities
        Q = plt.quiver(self.y_R, self.z_H, mean_v_diff, 
                       mean_w_diff, width=0.0022)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        plt.ylim(-0.2, 0.78)
        plt.xlim(-3.2, 3.2)
        if percent:
            keylen = 100
        else:
            keylen = 0.05
        plt.quiverkey(Q, 0.75, 0.05, keylen, str(keylen),
                      labelpos="E",
                      coordinates="figure",
                      fontproperties={"size": "small"})
        plt.axes().set_aspect(2)
        plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
        plt.tight_layout()
        if show:
            self.show()
        if save:
            if savedir: savedir += "/"
            plt.savefig(savedir+"/meancontquiv_diff"+savetype)
            
    def plot_mean_u_diff_std(self):
        u_ref = 1.0
        mean_u_ref = WakeMap(u_ref).mean_u/u_ref
        std = []
        u_array = np.arange(0.4, 1.4, 0.2)
        for u in u_array:
            wm = WakeMap(u)
            mean_u = wm.mean_u/wm.U_infty
            std.append(np.std((mean_u - mean_u_ref)/mean_u_ref))
        std = np.asarray(std)
        plt.figure()
        plt.plot(u_array, std)
        plt.show()
        
    def plot_k(self, fmt="", save=False, savetype = ".pdf", show=False,
               cb_orientation="vertical"):
        """Plot contours of turbulence kinetic energy."""
        plt.figure(figsize=(10, 2.5))
        cs = plt.contourf(self.y_R, self.z_H, self.k/(1/2*self.U_infty**2), 20,
                          cmap=plt.cm.coolwarm)
        if cb_orientation == "horizontal":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="horizontal", pad=0.14)
        elif cb_orientation == "vertical":
            cb = plt.colorbar(cs, shrink=1, extend="both", 
                              orientation="vertical", pad=0.02)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        cb.set_label(r"$k/\frac{1}{2}U_{\infty}^2$")
        self.turb_lines(color="black")
        ax = plt.axes()
        ax.set_aspect(H/R)
        plt.ylim(0, 0.75)
        plt.yticks(np.round(np.arange(0, 0.751, 0.125), decimals=2))
        plt.tight_layout()
        if save:
            plt.savefig("Figures/k_contours" + savetype)
        if show:
            self.show()
        
    def show(self):
        plt.show()

           
def plot_trans_wake_profile(quantity, U_infty=0.4, z_H=0.0, save=False, savedir="Figures", 
                            savetype=".pdf", newfig=True, marker="-ok",
                            fill="none", oldwake=False, figsize=(10, 5)):
    """Plots the transverse wake profile of some quantity. These can be
      * mean_u
      * mean_v
      * mean_w
      * std_u
    """
    Re_D = U_infty*D/nu
    label = "{:.1f}e6".format(Re_D/1e6)
    section = "Wake-" + str(U_infty)
    df = pd.read_csv(os.path.join("Data", "Processed", section+".csv"))
    df = df[df.z_H==z_H]
    q = df[quantity]
    y_R = df.y_R
    if newfig:
        plt.figure(figsize=figsize)
    if oldwake:
        plot_old_wake(quantity, y_R)
    if quantity in ["mean_upvp"]:
        unorm = U_infty**2
    else:
        unorm = U_infty
    plt.plot(y_R, q/unorm, marker, markerfacecolor=fill, label=label)
    plt.xlabel(r"$y/R$")
    plt.ylabel(ylabels[quantity])
    plt.tight_layout()
    
def plot_perf_re_dep(save=False, savedir="Figures", savetype=".pdf", 
                     errorbars=False, normalize_by=1.0, dual_xaxes=False, 
                     show=False, preliminary=False):
    """
    Plots Reynolds number dependence of power and drag coefficient. Note
    that if `errorbars=True`, the error bar values are the averages of all the
    individual run uncertainties.
    """
    df = pd.read_csv("Data/Processed/Perf-tsr_0.csv")
    df = df.append(pd.read_csv("Data/Processed/Perf-tsr_0-b.csv"), ignore_index=True)
    df = df[df.tow_speed_nom > 0.21]    
    df = df.groupby("tow_speed_nom").mean()    
    Re_D = df.mean_tow_speed*D/nu
    if normalize_by == "default":
        norm_cp = df.mean_cp[1.0]
        norm_cd = df.mean_cd[1.0]
    else:
        norm_cp = normalize_by
        norm_cd = normalize_by
    plt.figure()
    if errorbars:    
        plt.errorbar(Re_D, df.mean_cp/norm_cp, yerr=df.exp_unc_cp/norm_cp, 
                     fmt="-ok", markerfacecolor="none")
    else:
        plt.plot(Re_D, df.mean_cp/norm_cp, '-ok', markerfacecolor="none")
    plt.xlabel(r"$Re_D$")
    if normalize_by == "default":
        plt.ylabel(r"$C_P/C_{P0}$")
    else:
        plt.ylabel(r"$C_P$")
    plt.grid(True)
    ax = plt.gca()
    if dual_xaxes:
        plt.text(1.335e6, 0.46, "1e5")
        ax2 = ax.twiny()
        ax.xaxis.get_majorticklocs()
        ticklabs = np.arange(0.2e6, 1.6e6, 0.2e6)
        ticklabs = ticklabs/D*df.mean_tsr.mean()*chord/1e5
        ticklabs = [str(np.round(ticklab, decimals=1)) for ticklab in ticklabs]
        ax2.set_xticks(ax.xaxis.get_ticklocs())
        ax2.set_xlim((0.2e6, 1.4e6))
        ax2.set_xticklabels(ticklabs)
        ax2.set_xlabel(r"$Re_{c, \mathrm{ave}}$")
    ax.xaxis.major.formatter.set_powerlimits((0,0)) 
    plt.tight_layout()
    if preliminary:
        watermark()
    if save:
        plt.savefig(savedir + "/re_dep_cp" + savetype)
    plt.figure()
    if errorbars:
        plt.errorbar(Re_D, df.mean_cd/norm_cd, yerr=df.exp_unc_cd/norm_cd, 
                     fmt="-ok", markerfacecolor="none")
    else:
        plt.plot(Re_D, df.mean_cd/norm_cd, '-ok', markerfacecolor="none")
    plt.xlabel(r"$Re_D$")
    if normalize_by == "default":
        plt.ylabel(r"$C_D/C_{D0}$")
    else:
        plt.ylabel(r"$C_D$")
    ax = plt.gca()
    ax.xaxis.major.formatter.set_powerlimits((0,0))
    plt.grid(True)
    plt.tight_layout()
    if preliminary:
        watermark()
    if save:
        plt.savefig(savedir + "/re_dep_cd" + savetype)
    if show:
        plt.show()
    
def plot_old_wake(quantity, y_R):
    plt.hold(True)
    runs = range(32, 77)
    ind = [run-1 for run in runs]
    f = "../2013.03 VAT/Processed/"+quantity+".npy"
    q = np.load(f)[ind]
    plt.plot(y_R, q, 'xr', label=r"$Re_D=1.0 \times 10^6$", 
             markerfacecolor="none")
             
def plot_cfd_perf(quantity="cp", normalize_by="CFD"):
    Re_D = np.load(cfd_path + "/processed/Re_D.npy")
    q = np.load(cfd_path + "/processed/" + quantity + ".npy")
    if normalize_by=="CFD":
        normval = q[-3]
    else:
        normval = normalize_by
    plt.plot(Re_D, q/normval, "--^k", label="Simulation")
    
def plot_tare_drag():
    df = pd.read_csv("Data/Processed/Tare drag.csv")
    plt.figure()
    plt.plot(df.tow_speed, df.tare_drag, "-ok")
    plt.xlabel("Tow speed (m/s)")
    plt.ylabel("Tare drag (N)")
    plt.show()
    
def plot_settling(nrun, smooth_window=800, tol=1e-2, std=False, show=False):
    """Plot data from the settling experiments."""
    run = Run("Settling", nrun)
    tow_speed = run.tow_speed_nom
    stop_times = {0.3 : 132,
                  0.4 : 110,
                  0.5 : 98,
                  0.6 : 90,
                  0.7 : 82,
                  0.9 : 75,
                  1.0 : 68,
                  1.1 : 64,
                  1.2 : 62,
                  1.3 : 62}
    tstop = stop_times[tow_speed]
    u = run.u_all
    t = run.time_vec_all
    if std:
        uf = u.copy()
        uf[t>tstop] = ts.sigmafilter(uf[t>tstop], 4, 1)
        t_std, u_std = ts.runningstd(t, uf, 1000)
    u = ts.smooth(u, smooth_window)
    tseg = t[t>tstop]
    useg = u[t>tstop]
    zero_crossings = np.where(np.diff(np.sign(useg)))[0]
    zero_crossing = int((tseg[zero_crossings][0] - tstop))
    settling_time = int(tseg[np.where(np.abs(useg) < tol)[0][0]])
    print("Tow speed:", tow_speed, "m/s")
    print("First zero crossing:", zero_crossing, "s")
    print("Settling time based on threshold:", settling_time, "s")
    plt.figure()
    plt.plot(t - tstop, u, "k")
    plt.xlabel("t (s)")
    plt.ylabel("$u$ (m/s)")
    plt.tight_layout()
    if std:
        plt.figure()
        plt.plot(t_std - tstop, u_std)
        plt.xlabel("t (s)")
        plt.ylabel(r"$\sigma_u$")
        plt.tight_layout()
    if show:
        plt.show()
    
def plot_cp_curve(u_infty, save=False, show=False, savedir="Figures",
                  savetype=".pdf"):
    pc = PerfCurve(u_infty)
    pc.plotcp(save=False, show=False)
    if save:
        savepath = os.path.join(savedir, "cp_vs_tsr_{}".format(u_infty) + savetype)
        plt.savefig(savepath)
    if show:
        plt.show()
    
def plot_perf_curves(subplots=True, save=False, savedir="Figures", 
                     show=False, savetype=".pdf", preliminary=False):
    """Plots all performance curves."""
    if subplots:
        plt.figure(figsize=(12,5))
        plt.subplot(121)
    PerfCurve(0.4).plotcp(newfig=not subplots, show=False, marker=">")
    PerfCurve(0.6).plotcp(newfig=False, show=False, marker="s")
    PerfCurve(0.8).plotcp(newfig=False, show=False, marker="<")
    PerfCurve(1.0).plotcp(newfig=False, show=False, marker="o")
    PerfCurve(1.2).plotcp(newfig=False, show=False, marker="^")
    plt.legend(loc="lower left", ncol=2)
    if preliminary:
        watermark()
    if subplots:
        plt.subplot(122)
    PerfCurve(0.4).plotcd(newfig=not subplots, show=False, marker=">")
    PerfCurve(0.6).plotcd(newfig=False, show=False, marker="s")
    PerfCurve(0.8).plotcd(newfig=False, show=False, marker="<")
    PerfCurve(1.0).plotcd(newfig=False, show=False, marker="o")
    PerfCurve(1.2).plotcd(newfig=False, show=False, marker="^")
    plt.legend(loc="lower right")
    if preliminary:
        watermark()
    if save:
        plt.savefig(os.path.join(savedir, "perf_curves" + savetype))
    if show:
        plt.show()
    
def plot_wake_profiles(z_H=0.25, save=False, show=False, savedir="Figures", 
                       savetype=".pdf"):
    """Plots all wake profiles of interest."""
    legendlocs = {"mean_u" : 4,
                  "std_u" : 1,
                  "mean_upvp" : 1}
    for q in ["mean_u", "std_u", "mean_upvp"]:
        plot_trans_wake_profile(q, U_infty=0.4, z_H=z_H, newfig=True, marker="--vb",
                                fill="blue")
        plot_trans_wake_profile(q, U_infty=0.6, z_H=z_H, newfig=False, marker="sk",
                                fill="lightblue")
        plot_trans_wake_profile(q, U_infty=0.8, z_H=z_H, newfig=False, marker="<k",
                                fill="gray")
        plot_trans_wake_profile(q, U_infty=1.0, z_H=z_H, newfig=False, marker="-ok",
                                fill="orange")
        plot_trans_wake_profile(q, U_infty=1.2, z_H=z_H, newfig=False, marker="^k",
                                fill="red")
        plt.legend(loc=legendlocs[q])
        if q == "mean_upvp":
            plt.ylim((-0.015, 0.025))
        if save:
            plt.savefig(os.path.join(savedir, q+savetype))
    if show:
        plt.show()
    
def plot_meancontquiv(show=False, cb_orientation="vertical",
                      save=False, savedir="Figures", savetype=".pdf"):
    wm = WakeMap()
    wm.plot_meancontquiv(show=show, cb_orientation=cb_orientation)
    if save:
        p = os.path.join(savedir, "meancontquiv" + savetype)
        plt.savefig(p)
    if show:
        plt.show()
        
def plot_strut_torque(covers=False, save=False, savetype=".pdf", show=False,
                      newfig=True):
    section = "Strut-torque"
    figname = "strut_torque"
    if covers:
        section += "-covers"
        figname += "_covers"
    df = pd.read_csv("Data/Processed/" + section + ".csv")
    if newfig:
        plt.figure()
    plt.plot(df.tsr_ref, df.cp, "-ok")
    plt.xlabel(r"$\lambda_{\mathrm{ref}}$")
    plt.ylabel(r"$C_{P, \mathrm{ref}}$")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figures/" + figname + savetype)
    if show:
        plt.show()
        
def plot_cp_covers(save=False, savetype=".pdf", show=False, newfig=True,
                   add_strut_torque=False):
    """Plots the performance curve with covers."""
    df = pd.read_csv("Data/Processed/Perf-1.0-covers.csv")
    if newfig:
        plt.figure()
    if add_strut_torque:
        df2 = pd.read_csv("Data/Processed/Perf-1.0-no-blades-covers.csv")
        df.mean_cp -= df2.mean_cp
    plt.plot(df.mean_tsr, df.mean_cp, "-ok", label="Covers")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$C_P$")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figures/cp_curve_covers" + savetype)
    if show:
        plt.show()
        
def plot_cp_no_blades(covers=False, save=False, savetype=".pdf", show=False):
    """Plots the power coefficient curve with no blades."""
    section = "Perf-1.0-no-blades"
    figname = "cp_no_blades"
    if covers:
        section += "-covers"
        figname += "_covers"
    df = pd.read_csv("Data/Processed/" + section + ".csv")
    plt.figure()
    plt.plot(df.mean_tsr, df.mean_cp, "-ok")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$C_P$")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figures/" + figname + savetype)
    if show:
        plt.show()
    
    
def watermark():
    """Creates a "preliminary" watermark on plots."""
    ax = plt.gca()    
    plt.text(0.5, 0.5,"PRELIMINARY\nDO NOT PUBLISH",
             horizontalalignment="center",
             verticalalignment="center",
             transform=ax.transAxes,
             alpha=0.2,
             fontsize=32,
             zorder=10)

if __name__ == "__main__":
    pass