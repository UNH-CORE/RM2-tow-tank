# -*- coding: utf-8 -*-
"""This module contains classes and functions for plotting data."""

from __future__ import division, print_function
from .processing import *
from scipy.optimize import curve_fit
import os


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
        self.z_H = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75])
        self.loaded = False
        self.load()
        self.calc_transport()

    def load(self):
        self.df = pd.DataFrame()
        self.y_R = WakeProfile(self.U_infty, 0.0, "mean_u").y_R.values
        for z_H in self.z_H:
            wp = WakeProfile(self.U_infty, z_H, "mean_u")
            self.df = self.df.append(wp.df, ignore_index=True)
        self.mean_u = self.df.mean_u
        self.mean_v = self.df.mean_v
        self.mean_w = self.df.mean_w
        self.df["mean_k"] = \
                0.5*(self.df.mean_u**2 + self.df.mean_v**2 + self.df.mean_w**2)
        self.df["mean_upup"] = self.df.std_u**2
        self.df["mean_vpvp"] = self.df.std_v**2
        self.df["mean_wpwp"] = self.df.std_w**2
        self.grdims = (len(self.z_H), len(self.y_R))
        self.df = self.df.pivot(index="z_H", columns="y_R")

    def turb_lines(self, linestyles="solid", linewidth=2, color="gray"):
        plt.hlines(0.5, -1, 1, linestyles=linestyles, colors=color,
                   linewidth=linewidth)
        plt.vlines(-1, -0.2, 0.5, linestyles=linestyles, colors=color,
                   linewidth=linewidth)
        plt.vlines(1, -0.2, 0.5, linestyles=linestyles, colors=color,
                   linewidth=linewidth)

    def calc_transport(self):
        """
        Calculates wake tranport terms similar to Bachant and Wosnik (2015)
        "Characterising the near wake of a cross-flow turbine."
        """
        self.calc_mom_transport()
        self.calc_mean_k_grad()
        self.calc_k_prod_mean_diss()
        self.calc_mean_k_turb_trans()

    def calc_mean_k_turb_trans(self):
        """Calculates the transport of $K$ by turbulent fluctuations."""
        y, z  = self.y_R*R, self.z_H*H
        self.ddy_uvU = np.zeros(self.grdims)
        self.ddz_uwU = np.zeros(self.grdims)
        self.ddy_vvV = np.zeros(self.grdims)
        self.ddz_vwV = np.zeros(self.grdims)
        self.ddy_vwW = np.zeros(self.grdims)
        self.ddz_wwW = np.zeros(self.grdims)
        for n in range(len(z)):
            self.ddy_uvU[n,:] = \
                fdiff.second_order_diff((self.df.mean_upvp*self.df.mean_u)\
                .iloc[n,:], y)
            self.ddy_vvV[n,:] = \
                fdiff.second_order_diff((self.df.mean_vpvp*self.df.mean_v)\
                .iloc[n,:], y)
            self.ddy_vwW[n,:] = \
                fdiff.second_order_diff((self.df.mean_vpwp*self.df.mean_w)\
                .iloc[n,:], y)
        for n in range(len(y)):
            self.ddz_uwU[:,n] = \
                fdiff.second_order_diff((self.df.mean_upwp*self.df.mean_u)\
                .iloc[:,n], z)
            self.ddz_vwV[:,n] = \
                fdiff.second_order_diff((self.df.mean_vpwp*self.df.mean_v)\
                .iloc[:,n], z)
            self.ddz_wwW[:,n] = \
                fdiff.second_order_diff((self.df.mean_wpwp*self.df.mean_w)\
                .iloc[:,n], z)
        self.mean_k_turb_trans = -0.5*(self.ddy_uvU + \
                                       self.ddz_uwU + \
                                       self.ddy_vvV + \
                                       self.ddz_vwV + \
                                       self.ddy_vwW + \
                                       self.ddz_wwW)
        self.mean_k_turb_trans_y = -0.5*(self.ddy_uvU + \
                                         self.ddy_vvV + \
                                         self.ddy_vwW) # Only ddy terms
        self.mean_k_turb_trans_z = -0.5*(self.ddz_uwU + \
                                         self.ddz_vwV + \
                                         self.ddz_wwW) # Only ddz terms

    def calc_k_prod_mean_diss(self):
        """
        Calculates the production of turbulent kinetic energy and dissipation
        from mean shear. Note that the mean streamwise velocity derivatives
        have already been calculated by this point.
        """
        y, z = self.y_R*R, self.z_H*H
        self.dVdy = np.zeros(self.grdims)
        self.dVdz = np.zeros(self.grdims)
        self.dWdy = np.zeros(self.grdims)
        self.dWdz = np.zeros(self.grdims)
        for n in range(len(z)):
            self.dVdy[n,:] = \
                fdiff.second_order_diff(self.df.mean_v.iloc[n,:], y)
            self.dWdy[n,:] = \
                fdiff.second_order_diff(self.df.mean_w.iloc[n,:], y)
        for n in range(len(y)):
            self.dVdz[:,n] = \
                fdiff.second_order_diff(self.df.mean_v.iloc[:,n], z)
            self.dWdz[:,n] = \
                fdiff.second_order_diff(self.df.mean_w.iloc[:,n], z)
        self.k_prod = self.df.mean_upvp*self.dUdy + \
                      self.df.mean_upwp*self.dUdz + \
                      self.df.mean_vpwp*self.dVdz + \
                      self.df.mean_vpwp*self.dWdy + \
                      self.df.mean_vpvp*self.dVdy + \
                      self.df.mean_wpwp*self.dWdz
        self.mean_diss = -2.0*nu*(self.dUdy**2 + self.dUdz**2 + self.dVdy**2 +\
                                  self.dVdz**2 + self.dWdy**2 + self.dWdz**2)

    def calc_mean_k_grad(self):
        """Calulates $y$- and $z$-derivatives of $K$."""
        z = self.z_H*H
        y = self.y_R*R
        self.dKdy = np.zeros(self.grdims)
        self.dKdz = np.zeros(self.grdims)
        for n in range(len(z)):
            self.dKdy[n,:] = \
                fdiff.second_order_diff(self.df.mean_k.iloc[n,:], y)
        for n in range(len(y)):
            self.dKdz[:,n] = \
                fdiff.second_order_diff(self.df.mean_k.iloc[:,n], z)

    def calc_mom_transport(self):
        """
        Calculates relevant (and available) momentum transport terms in the
        RANS equations.
        """
        y = self.y_R*R
        z = self.z_H*H
        self.ddy_upvp = np.zeros(self.grdims)
        self.ddz_upwp = np.zeros(self.grdims)
        self.d2Udy2 = np.zeros(self.grdims)
        self.d2Udz2 = np.zeros(self.grdims)
        self.dUdy = np.zeros(self.grdims)
        self.dUdz = np.zeros(self.grdims)
        for n in range(len(z)):
            self.ddy_upvp[n, :] = \
                fdiff.second_order_diff(self.df.mean_upvp.iloc[n, :], y)
            self.dUdy[n, :] = \
                fdiff.second_order_diff(self.df.mean_u.iloc[n, :], y)
            self.d2Udy2[n, :] = fdiff.second_order_diff(self.dUdy[n, :], y)
        for n in range(len(y)):
            self.ddz_upwp[:, n] = \
                fdiff.second_order_diff(self.df.mean_upwp.iloc[:, n], z)
            self.dUdz[:, n] = \
                fdiff.second_order_diff(self.df.mean_u.iloc[:, n], z)
            self.d2Udz2[:, n] = fdiff.second_order_diff(self.dUdz[:, n], z)

    def plot_contours(self, quantity, label="", cb_orientation="vertical",
                      newfig=True, levels=None):
        """Plots contours of given quantity."""
        if newfig:
            plt.figure(figsize=(10, 2.5))
        cs = plt.contourf(self.y_R, self.z_H, quantity, 20,
                          cmap=plt.cm.coolwarm, levels=levels)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        if cb_orientation == "horizontal":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="horizontal", pad=0.3)
        elif cb_orientation == "vertical":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="vertical", pad=0.02)
        cb.set_label(label)
        self.turb_lines(color="black")
        plt.ylim((0, 0.75))
        ax = plt.axes()
        ax.set_aspect(H/R)
        plt.yticks([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75])
        plt.tight_layout()

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
        ax.set_aspect(H/R)
        plt.yticks([0.0, 0.13, 0.25, 0.38, 0.5, 0.63, 0.75])
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
        scale = 7.5/10.0
        plt.figure(figsize=(10*scale, 3*scale))
        # Add contours of mean velocity
        cs = plt.contourf(self.y_R, self.z_H, self.df.mean_u/self.U_infty, 20,
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
        Q = plt.quiver(self.y_R, self.z_H, self.df.mean_v/self.U_infty,
                       self.df.mean_w/self.U_infty, width=0.0022,
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
            plt.quiverkey(Q, 0.65, 0.09, 0.1, r"$0.1 U_\infty$",
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
        scale = 7.5/10.0
        plt.figure(figsize=(10*scale, 2.5*scale))
        cs = plt.contourf(self.y_R, self.z_H, self.df.k/(self.U_infty**2),
                          20, cmap=plt.cm.coolwarm)
        if cb_orientation == "horizontal":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="horizontal", pad=0.14)
        elif cb_orientation == "vertical":
            cb = plt.colorbar(cs, shrink=1, extend="both",
                              orientation="vertical", pad=0.02)
        plt.xlabel(r"$y/R$")
        plt.ylabel(r"$z/H$")
        cb.set_label(r"$k/U_{\infty}^2$")
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

    def make_K_bar_graph(self, save=False, savetype=".pdf",
                         print_analysis=True):
        """Make a bar graph of terms contributing to dK/dx:
          * Cross-stream advection
          * Vertical advection
          * Transport by turbulent fluctuations
          * Production of TKE
          * Mean dissipation
        """
        meanu, meanv, meanw = self.df.mean_u, self.df.mean_v, self.df.mean_w
        tty, ttz = self.mean_k_turb_trans_y, self.mean_k_turb_trans_z
        kprod, meandiss = self.k_prod, self.mean_diss
        dKdy, dKdz = self.dKdy, self.dKdz
        U = 1.0
        y_R, z_H = self.y_R, self.z_H
        plt.figure(figsize=(7, 3))
        names = [r"$y$-adv.", r"$z$-adv.", r"$y$-turb.", r"$z$-turb.",
                 r"$k$-prod.", "Mean diss."]
        quantities = [ts.average_over_area(-2*meanv/meanu*dKdy/(0.5*U**2)/D, y_R, z_H),
                      ts.average_over_area(-2*meanw/meanu*dKdz/(0.5*U**2)/D, y_R, z_H),
                      ts.average_over_area(2*tty/meanu/(0.5*U**2)/D, y_R, z_H),
                      ts.average_over_area(2*ttz/meanu/(0.5*U**2)/D, y_R, z_H),
                      ts.average_over_area(2*kprod/meanu/(0.5*U**2)/D, y_R, z_H),
                      ts.average_over_area(2*meandiss/meanu/(0.5*U**2)/D, y_R, z_H)]
        ax = plt.gca()
        ax.bar(range(len(names)), quantities, color="gray",
               edgecolor="black", width=0.5)
        ax.set_xticks(np.arange(len(names))+0.25)
        ax.set_xticklabels(names)
        plt.hlines(0, 0, len(names), color="black")
        plt.ylabel(r"$\frac{K \, \mathrm{ transport}}{UK_\infty D^{-1}}$")
        plt.tight_layout()
        if print_analysis:
            print("K recovery rate (%/D) =",
                  2*np.sum(quantities)/(0.5*U**2)/D*100)
        if save:
            plt.savefig("Figures/K_trans_bar_graph"+savetype)

    def show(self):
        plt.show()


def plot_trans_wake_profile(quantity, U_infty=0.4, z_H=0.0, save=False,
                            savedir="Figures", savetype=".pdf", newfig=True,
                            marker="-ok", fill="none", oldwake=False,
                            figsize=(10, 5)):
    """Plot the transverse wake profile of some quantity. These can be
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


def plot_perf_re_dep(subplots=True, save=False, savedir="Figures",
                     savetype=".pdf", errorbars=False, normalize_by=1.0,
                     dual_xaxes=False, show=False, preliminary=False):
    """Plot Reynolds number dependence of power and drag coefficient. Note
    that if `errorbars=True`, the error bar values are the averages of all the
    individual run uncertainties.
    """
    df = pd.read_csv("Data/Processed/Perf-tsr_0.csv")
    df = df.append(pd.read_csv("Data/Processed/Perf-tsr_0-b.csv"),
                               ignore_index=True)
    df = df[df.tow_speed_nom > 0.21]
    df = df.groupby("tow_speed_nom").mean()
    Re_D = df.mean_tow_speed*D/nu
    if normalize_by == "default":
        norm_cp = df.mean_cp[1.0]
        norm_cd = df.mean_cd[1.0]
    else:
        norm_cp = normalize_by
        norm_cd = normalize_by
    if subplots:
        plt.figure(figsize=(7.5, 3.5))
        plt.subplot(1, 2, 1)
    else:
        plt.figure()
    if errorbars:
        plt.errorbar(Re_D, df.mean_cp/norm_cp, yerr=df.exp_unc_cp/norm_cp,
                     fmt="-ok", markerfacecolor="none")
    else:
        plt.plot(Re_D, df.mean_cp/norm_cp, '-ok', markerfacecolor="none")
    plt.xlabel(r"$Re_D$")
    if normalize_by == "default":
        plt.ylabel(r"$C_P/C_{P_0}$")
    else:
        plt.ylabel(r"$C_P$")
    plt.grid(True)
    ax = plt.gca()
    if dual_xaxes:
        if subplots:
            x, y = 1.295e6, 0.465
        else:
            x, y = 1.345e6, 0.445
        plt.text(x, y, "1e5")
        ax2 = ax.twiny()
        ax.xaxis.get_majorticklocs()
        ticklabs = np.arange(0.2e6, 1.6e6, 0.2e6)
        ticklabs = ticklabs/D*df.mean_tsr.mean()*chord/1e5
        ticklabs = [str(np.round(ticklab, decimals=1)) for ticklab in ticklabs]
        ax2.set_xticks(ax.xaxis.get_ticklocs())
        ax2.set_xlim((0.2e6, 1.4e6))
        ax2.set_xticklabels(ticklabs)
        ax2.set_xlabel(r"$Re_{c, \mathrm{ave}}$")
    plt.tight_layout()
    if preliminary:
        watermark()
    if save and not subplots:
        plt.savefig(savedir + "/re_dep_cp" + savetype)
    if subplots:
        plt.subplot(1, 2, 2)
    else:
        plt.figure()
    if errorbars:
        plt.errorbar(Re_D, df.mean_cd/norm_cd, yerr=df.exp_unc_cd/norm_cd,
                     fmt="-ok", markerfacecolor="none")
    else:
        plt.plot(Re_D, df.mean_cd/norm_cd, '-ok', markerfacecolor="none")
    plt.xlabel(r"$Re_D$")
    if normalize_by == "default":
        plt.ylabel(r"$C_D/C_{D_0}$")
    else:
        plt.ylabel(r"$C_D$")
    plt.grid(True)
    ax = plt.gca()
    if dual_xaxes:
        if subplots:
            x, y = 1.295e6, 0.88
        else:
            x, y = 1.345e6, 0.875
        plt.text(x, y, "1e5")
        ax2 = ax.twiny()
        ax.xaxis.get_majorticklocs()
        ticklabs = np.arange(0.2e6, 1.6e6, 0.2e6)
        ticklabs = ticklabs/D*df.mean_tsr.mean()*chord/1e5
        ticklabs = [str(np.round(ticklab, decimals=1)) for ticklab in ticklabs]
        ax2.set_xticks(ax.xaxis.get_ticklocs())
        ax2.set_xlim((0.2e6, 1.4e6))
        ax2.set_xticklabels(ticklabs)
        ax2.set_xlabel(r"$Re_{c, \mathrm{ave}}$")
    plt.tight_layout()
    if preliminary:
        watermark()
    if save:
        if subplots:
            plt.savefig(savedir + "/perf_re_dep" + savetype)
        else:
            plt.savefig(savedir + "/re_dep_cd" + savetype)
    if show:
        plt.show()


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
        savepath = os.path.join(savedir,
                                "cp_vs_tsr_{}".format(u_infty) + savetype)
        plt.savefig(savepath)
    if show:
        plt.show()


def plot_perf_curves(subplots=True, save=False, savedir="Figures",
                     show=False, savetype=".pdf", preliminary=False):
    """Plot all performance curves."""
    if subplots:
        plt.figure(figsize=(11, 4.5))
        plt.subplot(121)
    PerfCurve(0.4).plotcp(newfig=not subplots, show=False, marker=">")
    PerfCurve(0.6).plotcp(newfig=False, show=False, marker="s")
    PerfCurve(0.8).plotcp(newfig=False, show=False, marker="<")
    PerfCurve(1.0).plotcp(newfig=False, show=False, marker="o")
    PerfCurve(1.2).plotcp(newfig=False, show=False, marker="^")
    if not subplots:
        plt.legend(loc="lower left", ncol=2)
    if preliminary:
        watermark()
    if save and not subplots:
        plt.savefig(os.path.join(savedir, "cp_curves" + savetype))
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
        if subplots:
            plt.savefig(os.path.join(savedir, "perf_curves" + savetype))
        else:
            plt.savefig(os.path.join(savedir, "cd_curves" + savetype))
    if show:
        plt.show()


def plot_wake_profiles(z_H=0.25, save=False, show=False, savedir="Figures",
                       savetype=".pdf"):
    """Plots all wake profiles of interest."""
    legendlocs = {"mean_u" : 4,
                  "std_u" : 1,
                  "mean_upvp" : 1}
    for q in ["mean_u", "std_u", "mean_upvp"]:
        plot_trans_wake_profile(q, U_infty=0.4, z_H=z_H, newfig=True,
                                marker="--vb", fill="blue")
        plot_trans_wake_profile(q, U_infty=0.6, z_H=z_H, newfig=False,
                                marker="sk", fill="lightblue")
        plot_trans_wake_profile(q, U_infty=0.8, z_H=z_H, newfig=False,
                                marker="<k", fill="gray")
        plot_trans_wake_profile(q, U_infty=1.0, z_H=z_H, newfig=False,
                                marker="-ok", fill="orange")
        plot_trans_wake_profile(q, U_infty=1.2, z_H=z_H, newfig=False,
                                marker="^k", fill="red")
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
    wm.plot_meancontquiv(save=save, show=show,
                         cb_orientation=cb_orientation, savetype=savetype,
                         savedir=savedir)


def plot_strut_torque(covers=False, power_law=True, cubic=False, save=False,
                      savetype=".pdf", show=False, newfig=True, fmt="-ok"):
    section = "Strut-torque"
    figname = "strut_torque"
    if covers:
        section += "-covers"
        figname += "_covers"
    df = pd.read_csv("Data/Processed/" + section + ".csv")
    if newfig:
        plt.figure()
    plt.plot(df.tsr_ref, df.cp, fmt, markerfacecolor="none", label="Stationary")
    if power_law:
        plot_power_law(df.tsr_ref, df.cp, xname=r"\lambda")
        plt.legend(loc="best")
    if cubic:
        plot_cubic(df.tsr_ref, df.cp, xname=r"\lambda")
        plt.legend(loc="best")
    plt.xlabel(r"$\lambda_{\mathrm{ref}}$")
    plt.ylabel(r"$C_{P_\mathrm{ref}}$")
    plt.xlim((0.5, 5.0))
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figures/" + figname + savetype)
    if show:
        plt.show()


def plot_cp_covers(save=False, savetype=".pdf", show=False, newfig=True,
                   add_strut_torque=False):
    """Plot the performance coefficient curve with covers."""
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


def plot_cp_no_blades(covers=False, power_law=True, cubic=False, save=False,
                      savetype=".pdf", show=False, newfig=True, fmt="-ok"):
    """Plot the power coefficient curve with no blades."""
    section = "Perf-1.0-no-blades"
    figname = "cp_no_blades"
    if covers:
        section += "-covers"
        figname += "_covers"
    df = pd.read_csv("Data/Processed/" + section + ".csv")
    if newfig:
        plt.figure()
    plt.plot(df.mean_tsr, df.mean_cp, fmt, markerfacecolor="none", label="Towed")
    if power_law:
        plot_power_law(df.mean_tsr, df.mean_cp, xname=r"\lambda")
        plt.legend(loc="best")
    if cubic:
        plot_cubic(df.mean_tsr, df.mean_cp, xname=r"\lambda")
        plt.legend(loc="best")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$C_P$")
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig("Figures/" + figname + savetype)
    if show:
        plt.show()


def plot_no_blades_all(save=False, savetype=".pdf"):
    """Plot all four cases of tests with no blades."""
    plt.figure(figsize=(7.5, 3.5))
    plt.subplot(1, 2, 1)
    plot_strut_torque(covers=False, newfig=False)
    plot_cp_no_blades(covers=False, newfig=False, fmt="-sk")
    plt.title("(a)")
    plt.subplot(1, 2, 2)
    plot_strut_torque(covers=True, newfig=False)
    plot_cp_no_blades(covers=True, newfig=False, fmt="-sk")
    plt.title("(b)")
    plt.tight_layout()
    if save:
        plt.savefig("Figures/no_blades_all" + savetype)


def plot_perf_covers(subplots=True, save=False, savetype=".pdf"):
    """Plot performance curves with strut covers installed."""
    df = PerfCurve(1.0).df
    dfc = pd.read_csv("Data/Processed/Perf-1.0-covers.csv")
    if subplots:
        plt.figure(figsize=(7.5, 3.5))
        plt.subplot(1, 2, 1)
    else:
        plt.figure()
    # Add horizontal line at zero
    plt.hlines(0, 0.5, 4.5, linewidth=1)
    plt.plot(df.mean_tsr, df.mean_cp, "-ok", markerfacecolor="none",
             label="NACA 0021")
    plt.plot(dfc.mean_tsr, dfc.mean_cp, "-sk", markerfacecolor="none",
             label="Cylindrical")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("$C_P$")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    if save and not subplots:
        plt.savefig("Figures/cp_covers" + savetype)
    if subplots:
        plt.subplot(1, 2, 2)
    else:
        plt.figure()
    plt.plot(df.mean_tsr, df.mean_cd, "-ok", markerfacecolor="none",
             label="NACA 0021")
    plt.plot(dfc.mean_tsr, dfc.mean_cd, "-sk", markerfacecolor="none",
             label="Cylindrical")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("$C_D$")
    plt.grid(True)
    if not subplots:
        plt.legend(loc="best")
    plt.tight_layout()
    if save:
        if subplots:
            plt.savefig("Figures/perf_covers" + savetype)
        else:
            plt.savefig("Figures/cd_covers" + savetype)


def plot_power_law(x, y, xname="x"):
    """Plot a power law fit for the given `x` and `y` data."""
    def func(x, a, b):
        return a*x**b
    coeffs, covar = curve_fit(func, x, y)
    a, b = coeffs[0], coeffs[1]
    xp = np.linspace(np.min(x), np.max(x), num=200)
    yp = a*xp**b
    plt.plot(xp, yp, label=r"${:.4f}{}^{{{:.4f}}}$".format(a, xname, b))


def plot_cubic(x, y, xname="x"):
    """Plots a power law fit for the given `x` and `y` data."""
    def func(x, a, b):
        return a*x**3 + b
    coeffs, covar = curve_fit(func, x, y)
    a, b = coeffs[0], coeffs[1]
    xp = np.linspace(np.min(x), np.max(x), num=200)
    yp = a*xp**3 + b
    plt.plot(xp, yp, label=r"${:.4f}{}^3 {:.4f}$".format(a, xname, b))


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
