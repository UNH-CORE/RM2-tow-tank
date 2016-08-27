# -*- coding: utf-8 -*-
"""This module contains classes and functions for processing data."""

from __future__ import division, print_function
import numpy as np
from pxl import timeseries as ts
from pxl.timeseries import calc_uncertainty, calc_exp_uncertainty
from pxl.io import loadhdf
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy.stats
from scipy.stats import nanmean, nanstd
from pxl import fdiff
import progressbar
import json
import os
import sys
import pandas as pd

if sys.version_info[0] == 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

# Dict for runs corresponding to each height
wakeruns = {0.0 : np.arange(0, 45),
            0.125 : np.arange(45, 90),
            0.25 : np.arange(90, 135),
            0.375 : np.arange(135, 180),
            0.5 : np.arange(180, 225),
            0.625 : np.arange(225, 270)}

# Read constants from turbine properties
with open("Config/turbine_properties.json") as f:
    turbine_properties = json.load(f)
turbine_properties = turbine_properties["RM2"]
H = turbine_properties["height"]
D = turbine_properties["diameter"]
A = D*H
R = D/2
rho = 1000.0
nu = 1e-6
tip_chord = 0.04
root_chord = 0.067
chord = root_chord

# Directory constants
raw_data_dir = os.path.join("Data", "Raw")
processed_data_dir = os.path.join("Data", "Processed")


def calc_b_vec(vel):
    """Calculates the systematic error of a Vectrino measurement (in m/s)
    from their published specs. Returns half the +/- value as b."""
    return 0.5*(0.005*np.abs(vel) + 0.001)


def calc_tare_torque(rpm):
    """Returns tare torque array given RPM array."""
    return 0.00104768276035*rpm - 0.848866229797


def calc_re_c(u_infty, tsr=3.1):
    """Calculates the average blade chord Reynolds number based on free stream
    velocity and tip speed ratio.
    """
    return tsr*u_infty*chord/nu


class Run(object):
    """Object that represents a single turbine tow"""
    def __init__(self, section, nrun):
        self.section = section
        self.nrun = int(nrun)
        section_raw_dir = os.path.join("Data", "Raw", section)
        self.raw_dir = os.path.join(section_raw_dir, str(self.nrun))
        self.loaded = False
        self.t2found = False
        self.not_loadable = False
        self.wake_calculated = False
        self.load()
        # Do all processing if all data was loaded successfully
        if self.loaded:
            self.subtract_tare_drag()
            self.add_tare_torque()
            self.calc_perf_instantaneous()
            self.make_trimmed()
            self.filter_wake()
            self.calc_wake_instantaneous()
            self.calc_perf_per_rev()
            self.calc_perf_stats()
            self.calc_wake_stats()
            self.calc_perf_uncertainty()
            self.calc_perf_exp_uncertainty()
        else:
            print("Cannot load Run data")

    def load(self):
        """Load the data from the run into memory."""
        self.loaded = True
        try:
            with open("Config/raw_data_urls.json") as f:
                raw_data_urls = json.load(f)
        except IOError:
            raw_data_urls = {}
        # Load metadata if it exists
        fpath_metadata = os.path.join(self.raw_dir, "metadata.json")
        if os.path.isfile(fpath_metadata):
            self.load_metadata()
        elif make_remote_name(fpath_metadata) in raw_data_urls:
            self.download_raw("metadata.json")
            self.load_metadata()
        else:
            self.loaded = False
        # Load NI data if it exists
        fpath_nidata = os.path.join(self.raw_dir, "nidata.h5")
        if os.path.isfile(fpath_nidata):
            self.load_nidata()
        elif make_remote_name(fpath_nidata) in raw_data_urls:
            self.download_raw("nidata.h5")
            self.load_nidata()
        else:
            self.loaded = False
        # Load ACS data if it exists
        fpath_acsdata = os.path.join(self.raw_dir, "acsdata.h5")
        if os.path.isfile(fpath_acsdata):
            self.load_acsdata()
        elif make_remote_name(fpath_acsdata) in raw_data_urls:
            self.download_raw("acsdata.h5")
            self.load_acsdata()
        else:
            self.loaded = False
        # Load Vectrino data if it exists
        fpath_vecdata = os.path.join(self.raw_dir, "vecdata.h5")
        if os.path.isfile(fpath_vecdata):
            self.load_vecdata()
        elif make_remote_name(fpath_vecdata) in raw_data_urls:
            self.download_raw("vecdata.h5")
            self.load_vecdata()
        else:
            self.loaded = False

    def load_metadata(self):
        """Load run metadata."""
        with open(os.path.join(self.raw_dir, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.tow_speed_nom = np.round(self.metadata["Tow speed (m/s)"], decimals=1)
        self.y_R = self.metadata["Vectrino y/R"]
        self.z_H = self.metadata["Vectrino z/H"]
        self.tsr_nom = self.metadata["Tip speed ratio"]

    def load_nidata(self):
        nidata = loadhdf(os.path.join(self.raw_dir, "nidata.h5"))
        self.time_ni = nidata["time"]
        self.sr_ni = (1.0/(self.time_ni[1] - self.time_ni[0]))
        self.carriage_pos = nidata["carriage_pos"]
        self.tow_speed_ni = fdiff.second_order_diff(self.carriage_pos, self.time_ni)
        self.tow_speed_ni = ts.smooth(self.tow_speed_ni, 100)
        self.tow_speed_ref = self.tow_speed_ni
        self.torque = nidata["torque_trans"]
        self.torque_arm = nidata["torque_arm"]
        self.drag = nidata["drag_left"] + nidata["drag_right"]
        # Remove offsets from drag, not torque
        t0 = 2
        self.drag = self.drag - np.mean(self.drag[0:self.sr_ni*t0])
        # Compute RPM and omega
        self.angle = nidata["turbine_angle"]
        self.rpm_ni = fdiff.second_order_diff(self.angle, self.time_ni)/6.0
        self.rpm_ni = ts.smooth(self.rpm_ni, 8)
        self.omega_ni = self.rpm_ni*2*np.pi/60.0
        self.omega = self.omega_ni
        self.tow_speed = self.tow_speed_ref

    def load_acsdata(self):
        fpath = os.path.join(self.raw_dir, "acsdata.h5")
        acsdata = loadhdf(fpath)
        self.tow_speed_acs = acsdata["carriage_vel"]
        self.rpm_acs = acsdata["turbine_rpm"]
        self.rpm_acs = ts.sigmafilter(self.rpm_acs, 3, 3)
        self.omega_acs = self.rpm_acs*2*np.pi/60.0
        self.time_acs = acsdata["time"]
        if len(self.time_acs) != len(self.omega_acs):
            newlen = np.min((len(self.time_acs), len(self.omega_acs)))
            self.time_acs = self.time_acs[:newlen]
            self.omega_acs = self.omega_acs[:newlen]
        self.omega_acs_interp = np.interp(self.time_ni, self.time_acs, self.omega_acs)
        self.rpm_acs_interp = self.omega_acs_interp*60.0/(2*np.pi)

    def load_vecdata(self):
        try:
            vecdata = loadhdf(os.path.join(self.raw_dir, "vecdata.h5"))
            self.sr_vec = 200.0
            self.time_vec = vecdata["time"]
            self.u = vecdata["u"]
            self.v = vecdata["v"]
            self.w = vecdata["w"]
        except IOError:
            self.vecdata = None

    def download_raw(self, name):
        download_raw(self.section, self.nrun, name)

    def subtract_tare_drag(self):
        df = pd.read_csv(os.path.join("Data", "Processed", "Tare-drag.csv"))
        self.tare_drag = \
                df.tare_drag[df.tow_speed==self.tow_speed_nom].values[0]
        self.drag = self.drag - self.tare_drag

    def add_tare_torque(self):
        rpm_ref = self.rpm_ni
        # Add tare torque
        self.tare_torque = calc_tare_torque(rpm_ref)
        self.torque += self.tare_torque

    def calc_perf_instantaneous(self):
        omega_ref = self.omega_ni
        # Compute power
        self.power = self.torque*omega_ref
        self.tsr = omega_ref*R/self.tow_speed_ref
        # Compute power, drag, and torque coefficients
        self.cp = self.power/(0.5*rho*A*self.tow_speed_ref**3)
        self.cd = self.drag/(0.5*rho*A*self.tow_speed_ref**2)
        self.ct = self.torque/(0.5*rho*A*R*self.tow_speed_ref**2)
        # Remove datapoints for coefficients where tow speed is small
        self.cp[np.abs(self.tow_speed_ref < 0.01)] = np.nan
        self.cd[np.abs(self.tow_speed_ref < 0.01)] = np.nan

    def load_vectxt(self):
        """Load Vectrino data from text (*.dat) file."""
        data = np.loadtxt(self.raw_dir + "/vecdata.dat", unpack=True)
        self.time_vec_txt = data[0]
        self.u_txt = data[3]

    def make_trimmed(self):
        """Trim all time series and replace the full run names with names with
        the '_all' suffix.
        """
        # Put in some guesses for t1 and t2
        stpath = "Config/Steady times/{}.csv".format(self.tow_speed_nom)
        s_times = pd.read_csv(stpath)
        s_times = s_times[s_times.tsr==self.tsr_nom].iloc[0]
        self.t1, self.t2 = s_times.t1, s_times.t2
        self.find_t2()
        # Trim performance quantities
        self.time_ni_all = self.time_ni
        self.time_perf_all = self.time_ni
        self.time_ni = self.time_ni_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.time_perf = self.time_ni
        self.angle_all = self.angle
        self.angle = self.angle_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.torque_all = self.torque
        self.torque = self.torque_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.torque_arm_all = self.torque_arm
        self.torque_arm = self.torque_arm_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.omega_all = self.omega
        self.omega = self.omega_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.tow_speed_all = self.tow_speed
        self.tow_speed = self.tow_speed_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.tsr_all = self.tsr
        self.tsr = self.tsr_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.cp_all = self.cp
        self.cp = self.cp_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.ct_all = self.ct
        self.ct = self.ct_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.cd_all = self.cd
        self.cd = self.cd_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.rpm_ni_all = self.rpm_ni
        self.rpm_ni = self.rpm_ni_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        self.rpm = self.rpm_ni
        self.rpm_all = self.rpm_ni_all
        self.drag_all = self.drag
        self.drag = self.drag_all[self.t1*self.sr_ni:self.t2*self.sr_ni]
        # Trim wake quantities
        self.time_vec_all = self.time_vec
        self.time_vec = self.time_vec_all[self.t1*self.sr_vec:self.t2*self.sr_vec]
        self.u_all = self.u
        self.u = self.u_all[self.t1*self.sr_vec:self.t2*self.sr_vec]
        self.v_all = self.v
        self.v = self.v_all[self.t1*self.sr_vec:self.t2*self.sr_vec]
        self.w_all = self.w
        self.w = self.w_all[self.t1*self.sr_vec:self.t2*self.sr_vec]

    def find_t2(self):
        sr = self.sr_ni
        angle1 = self.angle[sr*self.t1]
        angle2 = self.angle[sr*self.t2]
        n3rdrevs = np.floor((angle2-angle1)/120.0)
        self.n_revs = int(np.floor((angle2-angle1)/360.0))
        self.n_blade_pass = int(n3rdrevs)
        angle2 = angle1 + n3rdrevs*120
        t2i = np.where(np.round(self.angle)==np.round(angle2))[0][0]
        t2 = self.time_ni[t2i]
        self.t2 = np.round(t2, decimals=2)
        self.t2found = True
        self.t1_wake = self.t1
        self.t2_wake = self.t2

    def calc_perf_stats(self):
        """Calculate mean performance based on trimmed time series."""
        self.mean_tsr, self.std_tsr = nanmean(self.tsr), nanstd(self.tsr)
        self.mean_cp, self.std_cp = nanmean(self.cp), nanstd(self.cp)
        self.mean_cd, self.std_cd = nanmean(self.cd), nanstd(self.cd)
        self.mean_ct, self.std_ct = nanmean(self.ct), nanstd(self.ct)
        self.mean_u_enc = nanmean(self.tow_speed)
        self.std_u_enc = nanstd(self.tow_speed)

    def print_perf_stats(self):
        print("tow_speed_nom =", self.tow_speed_nom)
        print("mean_tow_speed_enc =", self.mean_u_enc)
        print("std_tow_speed_enc =", self.std_u_enc)
        print("TSR = {:.2f} +/- {:.2f}".format(self.mean_tsr, self.exp_unc_tsr))
        print("C_P = {:.2f} +/- {:.2f}".format(self.mean_cp, self.exp_unc_cp))
        print("C_D = {:.2f} +/- {:.2f}".format(self.mean_cd, self.exp_unc_cd))

    def calc_perf_uncertainty(self):
        """See uncertainty IPython notebook for equations."""
        # Systematic uncertainty estimates
        b_torque = 0.5/2
        b_angle = 3.14e-5/2
        b_car_pos = 0.5e-5/2
        b_force = 0.28/2
        # Uncertainty of C_P
        omega = self.omega.mean()
        torque = self.torque.mean()
        u_infty = np.mean(self.tow_speed)
        const = 0.5*rho*A
        b_cp = np.sqrt((omega/(const*u_infty**3))**2*b_torque**2 + \
                       (torque/(const*u_infty**3))**2*b_angle**2 + \
                       (-3*torque*omega/(const*u_infty**4))**2*b_car_pos**2)
        self.b_cp = b_cp
        self.unc_cp = calc_uncertainty(self.cp_per_rev, b_cp)
        # Drag coefficient
        drag = self.drag.mean()
        b_cd = np.sqrt((1/(const*u_infty**2))**2*b_force**2 + \
                       (1/(const*u_infty**2))**2*b_force**2 +
                       (-2*drag/(const*u_infty**3))**2*b_car_pos**2)
        self.unc_cd = calc_uncertainty(self.cd_per_rev, b_cd)
        self.b_cd = b_cd
        # Tip speed ratio
        b_tsr = np.sqrt((R/(u_infty))**2*b_angle**2 + \
                        (-omega*R/(u_infty**2))**2*b_car_pos**2)
        self.unc_tsr = calc_uncertainty(self.tsr_per_rev, b_tsr)
        self.b_tsr = b_tsr

    def calc_perf_exp_uncertainty(self):
        """Calculated expanded uncertainty for performance quantities."""
        # Power coefficient
        self.exp_unc_cp, self.dof_cp = calc_exp_uncertainty(self.n_revs,
                self.std_cp_per_rev, self.unc_cp, self.b_cp)
        # Drag coefficient
        self.exp_unc_cd, self.dof_cd = calc_exp_uncertainty(self.n_revs,
                self.std_cd_per_rev, self.unc_cd, self.b_cd)
        # Tip speed ratio
        self.exp_unc_tsr, self.dof_tsr = calc_exp_uncertainty(self.n_revs,
                self.std_tsr_per_rev, self.unc_tsr, self.b_tsr)

    def calc_wake_instantaneous(self):
        """Create fluctuating and Reynolds stress time series. Note that
        time series must be trimmed first, or else subtracting the mean makes
        no sense. Prime variables are denoted by a `p` e.g., $u'$ is `up`.
        """
        self.up = self.u - nanmean(self.u)
        self.vp = self.v - nanmean(self.v)
        self.wp = self.w - nanmean(self.w)
        self.upup = self.up**2
        self.upvp = self.up*self.vp
        self.upwp = self.up*self.wp
        self.vpvp = self.vp**2
        self.vpwp = self.vp*self.wp
        self.wpwp = self.wp**2

    def filter_wake(self, std=4, passes=1, thresh=0.95):
        """Apply filtering to wake velocity data with a standard deviation
        filter, threshold filter, or both. Renames unfiltered time series with
        the '_unf' suffix. Time series are already trimmed before they reach
        this point, so no slicing is necessary.
        """
        # Calculate means
        mean_u = self.u.mean()
        mean_v = self.v.mean()
        mean_w = self.w.mean()
        # Create new unfiltered arrays
        self.u_unf = self.u.copy()
        self.v_unf = self.v.copy()
        self.w_unf = self.w.copy()
        if std > 0:
            # Do standard deviation filters
            self.u = ts.sigmafilter(self.u, std, passes)
            self.v = ts.sigmafilter(self.v, std, passes)
            self.w = ts.sigmafilter(self.w, std, passes)
        if thresh != False:
            # Do threshold filter on u
            ibad = np.where(self.u > mean_u + thresh)[0]
            ibad = np.append(ibad, np.where(self.u < mean_u - thresh)[0])
            self.u[ibad] = np.nan
            # Do threshold filter on v
            ibad = np.where(self.v > mean_v + thresh)[0]
            ibad = np.append(ibad, np.where(self.v < mean_v - thresh)[0])
            self.v[ibad] = np.nan
            # Do threshold filter on w
            ibad = np.where(self.w > mean_w + thresh)[0]
            ibad = np.append(ibad, np.where(self.w < mean_w - thresh)[0])
            self.w[ibad] = np.nan
        # Count up bad datapoints
        self.nbadu = len(np.where(np.isnan(self.u)==True)[0])
        self.nbadv = len(np.where(np.isnan(self.v)==True)[0])
        self.nbadw = len(np.where(np.isnan(self.w)==True)[0])
        self.nbad = self.nbadu + self.nbadv + self.nbadw

    def calc_wake_stats(self):
        if not self.t2found:
            self.find_t2()
        self.mean_u, self.std_u = nanmean(self.u), nanstd(self.u)
        self.mean_v, self.std_v = nanmean(self.v), nanstd(self.v)
        self.mean_w, self.std_w = nanmean(self.w), nanstd(self.w)
        self.mean_upup, self.std_upup = nanmean(self.upup), nanstd(self.upup)
        self.mean_upvp, self.std_upvp = nanmean(self.upvp), nanstd(self.upvp)
        self.mean_upwp, self.std_upwp = nanmean(self.upwp), nanstd(self.upwp)
        self.mean_vpvp, self.std_vpvp = nanmean(self.vpvp), nanstd(self.vpvp)
        self.mean_vpwp, self.std_vpwp = nanmean(self.vpwp), nanstd(self.vpwp)
        self.mean_wpwp, self.std_wpwp = nanmean(self.wpwp), nanstd(self.wpwp)
        self.k = 0.5*(self.mean_upup + self.mean_vpvp + self.mean_wpwp)

    def print_wake_stats(self):
        ntotal = int((self.t2 - self.t1)*self.sr_vec*3)
        print("y/R =", self.y_R)
        print("z/H =", self.z_H)
        print("mean_u/tow_speed_nom =", self.mean_u/self.tow_speed_nom)
        print("std_u/tow_speed_nom =", self.std_u/self.tow_speed_nom)
        print(str(self.nbad)+"/"+str(ntotal), "data points omitted")

    def calc_wake_uncertainty(self):
        """Compute delta values for wake measurements from Vectrino accuracy
        specs, not statistical uncertainties.
        """
        self.unc_mean_u = np.nan
        self.unc_std_u = np.nan

    def calc_perf_per_rev(self):
        """Compute mean power coefficient over each revolution."""
        angle = self.angle*1
        angle -= angle[0]
        cp = np.zeros(self.n_revs)
        cd = np.zeros(self.n_revs)
        tsr = np.zeros(self.n_revs)
        torque = np.zeros(self.n_revs)
        omega = np.zeros(self.n_revs)
        start_angle = 0.0
        for n in range(self.n_revs):
            end_angle = start_angle + 360
            ind = np.logical_and(angle >= start_angle, end_angle > angle)
            cp[n] = self.cp[ind].mean()
            cd[n] = self.cd[ind].mean()
            tsr[n] = self.tsr[ind].mean()
            torque[n] = self.torque[ind].mean()
            omega[n] = self.omega[ind].mean()
            start_angle += 360
        self.cp_per_rev = cp
        self.std_cp_per_rev = cp.std()
        self.cd_per_rev = cd
        self.std_cd_per_rev = cd.std()
        self.tsr_per_rev = tsr
        self.std_tsr_per_rev = tsr.std()
        self.torque_per_rev = torque
        self.std_torque_per_rev = torque.std()

    @property
    def cp_conf_interval(self, alpha=0.95):
        self.calc_perf_per_rev()
        t_val = scipy.stats.t.interval(alpha=alpha, df=self.n_revs-1)[1]
        std = self.std_cp_per_rev
        return t_val*std/np.sqrt(self.n_revs)

    def detect_badvec(self):
        """Detect if Vectrino data is bad by looking at first 2 seconds of
        data, and checking if there are many datapoints.
        """
        nbad = len(np.where(np.abs(self.u[:400]) > 0.5)[0])
        print(nbad, "bad Vectrino datapoints in first 2 seconds")
        if nbad > 50:
            self.badvec = True
            print("Vectrino data bad")
        else:
            self.badvec = False
            print("Vectrino data okay")

    @property
    def summary(self):
        s = pd.Series()
        s["run"] = self.nrun
        if self.loaded:
            s["tow_speed_nom"] = self.tow_speed_nom
            s["tsr_nom"] = self.tsr_nom
            s["mean_tow_speed"] = self.mean_u_enc
            s["std_tow_speed"] = self.std_u_enc
            s["t1"] = self.t1
            s["t2"] = self.t2
            s["n_blade_pass"] = self.n_blade_pass
            s["n_revs"] = self.n_revs
            s["mean_tsr"] = self.mean_tsr
            s["mean_cp"] = self.mean_cp
            s["mean_cd"] = self.mean_cd
            s["std_tsr"] = self.std_tsr
            s["std_cp"] = self.std_cp
            s["std_cd"] = self.std_cd
            s["std_tsr_per_rev"] = self.std_tsr_per_rev
            s["std_cp_per_rev"] = self.std_cp_per_rev
            s["std_cd_per_rev"] = self.std_cd_per_rev
            s["sys_unc_tsr"] = self.b_tsr
            s["sys_unc_cp"]  = self.b_cp
            s["sys_unc_cd"] = self.b_cd
            s["exp_unc_tsr"] = self.exp_unc_tsr
            s["exp_unc_cp"] = self.exp_unc_cp
            s["exp_unc_cd"] = self.exp_unc_cd
            s["dof_tsr"] = self.dof_tsr
            s["dof_cp"] = self.dof_cp
            s["dof_cd"] = self.dof_cd
            s["t1_wake"] = self.t1_wake
            s["t2_wake"] = self.t2_wake
            s["y_R"] = self.y_R
            s["z_H"] = self.z_H
            s["mean_u"] = self.mean_u
            s["mean_v"] = self.mean_v
            s["mean_w"] = self.mean_w
            s["std_u"] = self.std_u
            s["std_v"] = self.std_v
            s["std_w"] = self.std_w
            s["mean_upvp"] = self.mean_upvp
            s["mean_upwp"] = self.mean_upwp
            s["mean_vpwp"] = self.mean_vpwp
            s["k"] = self.k
        else:
            s["tow_speed_nom"] = np.nan
            s["tsr_nom"] = np.nan
            s["mean_tow_speed"] = np.nan
            s["std_tow_speed"] = np.nan
            s["t1"] = np.nan
            s["t2"] = np.nan
            s["n_blade_pass"] = np.nan
            s["n_revs"] = np.nan
            s["mean_tsr"] = np.nan
            s["mean_cp"] = np.nan
            s["mean_cd"] = np.nan
            s["std_tsr"] = np.nan
            s["std_cp"] = np.nan
            s["std_cd"] = np.nan
            s["std_tsr_per_rev"] = np.nan
            s["std_cp_per_rev"] = np.nan
            s["std_cd_per_rev"] = np.nan
            s["exp_unc_tsr"] = np.nan
            s["exp_unc_cp"] = np.nan
            s["exp_unc_cd"] = np.nan
            s["dof_tsr"] = np.nan
            s["dof_cp"] = np.nan
            s["dof_cd"] = np.nan
            s["t1_wake"] = np.nan
            s["t2_wake"] = np.nan
            s["y_R"] = np.nan
            s["z_H"] = np.nan
            s["mean_u"] = np.nan
            s["mean_v"] = np.nan
            s["mean_w"] = np.nan
            s["std_u"] = np.nan
            s["std_v"] = np.nan
            s["std_w"] = np.nan
            s["mean_upvp"] = np.nan
            s["mean_upwp"] = np.nan
            s["mean_vpwp"] = np.nan
            s["k"] = np.nan
        return s

    def plot_perf(self, quantity="power coefficient"):
        """Plot the run's data."""
        if not self.loaded:
            self.load()
        if quantity == "drag":
            quantity = self.drag
            ylabel = "Drag (N)"
            ylim = None
        elif quantity == "torque":
            quantity = self.torque
            ylabel = "Torque (Nm)"
            ylim = None
        elif quantity.lower == "power coefficient" or "cp" or "c_p":
            quantity = self.cp
            ylabel = "$C_P$"
            ylim = (-1, 1)
        plt.figure()
        plt.plot(self.time_ni, quantity, 'k')
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.tight_layout()

    def plot_wake(self):
        """Plot streamwise velocity over experiment."""
        if not self.loaded:
            self.load()
        plt.figure()
        self.filter_wake()
        plt.plot(self.time_vec, self.u, 'k')
        plt.xlabel("Time (s)")
        plt.ylabel("$u$ (m/s)")

    def plot_acs(self):
        if not self.loaded:
            self.load()
        plt.figure()
        plt.plot(self.time_acs, self.rpm_acs)
        plt.hold(True)
        plt.plot(self.time_ni, self.rpm_ni)
        plt.figure()
        plt.plot(self.time_ni, self.tow_speed_ni)
        plt.hold(True)
        plt.plot(self.time_acs, self.tow_speed_acs)
        plt.show()

    def plot_carriage_vel(self):
        if not self.loaded:
            self.load()
        plt.figure()
        plt.plot(self.time_ni, self.tow_speed_ni)
        plt.tight_layout()
        plt.show()


class Section(object):
    def __init__(self, name):
        self.name = name
        self.processed_path = os.path.join(processed_data_dir, name+".csv")
        self.test_plan_path = os.path.join("Config", "Test plan", name+".csv")
        self.load()
    def load(self):
        self.test_plan = pd.read_csv(self.test_plan_path, index_col="run")
        try:
            self.data = pd.read_csv(self.processed_path, index_col="run")
        except IOError:
            self.data = pd.DataFrame()
    @property
    def mean_cp(self):
        return self.data.mean_cp
    def process(self, nproc=4, nruns="all", save=True):
        """Process an entire section of data."""
        self.process_parallel(nproc=nproc, nruns=nruns)
        self.data.index.name = "run"
        self.data = self.data.sort_index()
        if save:
            self.data.to_csv(self.processed_path, na_rep="NaN", index=True)
    def process_parallel(self, nproc=4, nruns="all"):
        s = self.name
        runs = self.test_plan.index.values
        if nruns != "all":
            if nruns == "new":
                try:
                    runs = runs[np.where(np.isnan(self.data.mean_cp))]
                    self.data = self.data.iloc[np.where(~np.isnan(self.data.mean_cp))]
                except AttributeError:
                    pass
            else:
                runs = runs[:nruns]
        if len(runs) > 0:
            pool = mp.Pool(processes=nproc)
            results = [pool.apply_async(process_run, args=(s,n)) for n in runs]
            output = [p.get() for p in results]
            pool.close()
            self.newdata = pd.DataFrame(output)
            self.newdata.set_index("run", inplace=True)
            self.newdata = self.newdata.sort_index()
        else:
            self.newdata = pd.DataFrame()
        if nruns == "all":
            self.data = self.newdata
        else:
            self.data = self.data.append(self.newdata)
            self.data = self.data.drop_duplicates()


def process_run(section, nrun):
    run = Run(section, nrun)
    return run.summary


def process_latest_run(section):
    """Automatically detect the most recently acquired run and process it,
    then print a summary to the shell.
    """
    print("Processing latest run in", section)
    raw_dir = os.path.join("Data", "Raw", section)
    dirlist = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) \
               if os.path.isdir(os.path.join(raw_dir, d))]
    dirlist = sorted(dirlist, key=os.path.getmtime, reverse=True)
    for d in dirlist:
        try:
            nrun = int(os.path.split(d)[-1])
            break
        except ValueError:
            print(d, "is not a properly formatted directory")
    print("\nSummary for {} run {}:".format(section, nrun))
    print(Run(section, nrun).summary)


def batch_process_section(name):
    s = Section(name)
    s.process()


def batch_process_all():
    """Batch process all sections."""
    sections = ["Perf-0.4", "Perf-0.4-b", "Perf-0.6", "Perf-0.6-b",
                "Perf-0.8", "Perf-0.8-b", "Perf-1.0", "Perf-1.0-b",
                "Perf-1.2", "Perf-1.2-b", "Perf-1.0-covers",
                "Perf-1.0-no-blades", "Perf-1.0-no-blades-covers",
                "Perf-tsr_0", "Perf-tsr_0-b", "Wake-1.0-0.0", "Wake-1.0-0.125",
                "Wake-1.0-0.25", "Wake-1.0-0.375", "Wake-1.0-0.5",
                "Wake-1.0-0.625", "Wake-1.0-0.75"]
    for section in sections:
        print("Processing {}".format(section))
        batch_process_section(section)


def process_tare_drag(nrun, plot=False):
    """Process a single tare drag run."""
    print("Processing tare drag run", nrun)
    times = {0.2: (15, 120),
             0.3: (10, 77),
             0.4: (10, 56),
             0.5: (8, 47),
             0.6: (10, 40),
             0.7: (8, 33),
             0.8: (5, 31),
             0.9: (8, 27),
             1.0: (6, 24),
             1.1: (9, 22),
             1.2: (8, 21),
             1.3: (7, 19),
             1.4: (6, 18)}
    rdpath = os.path.join(raw_data_dir, "Tare-drag", str(nrun))
    with open(os.path.join(rdpath, "metadata.json")) as f:
        metadata = json.load(f)
    speed = float(metadata["Tow speed (m/s)"])
    nidata = loadhdf(os.path.join(rdpath, "nidata.h5"))
    time_ni  = nidata["time"]
    drag = nidata["drag_left"] + nidata["drag_right"]
    drag = drag - np.mean(drag[:2000])
    t1, t2 = times[speed]
    meandrag, x = ts.calcstats(drag, t1, t2, 2000)
    print("Tare drag =", meandrag, "N at", speed, "m/s")
    if plot:
        plt.figure()
        plt.plot(time_ni, drag, 'k')
        plt.show()
    return speed, meandrag


def batch_process_tare_drag(plot=False):
    """Process all tare drag data."""
    runs = os.listdir("Data/Raw/Tare-drag")
    runs = sorted([int(run) for run in runs])
    speed = np.zeros(len(runs))
    taredrag = np.zeros(len(runs))
    for n in range(len(runs)):
        speed[n], taredrag[n] = process_tare_drag(runs[n])
    data = pd.DataFrame()
    data["run"] = runs
    data["tow_speed"] = speed
    data["tare_drag"] = taredrag
    data.to_csv("Data/Processed/Tare-drag.csv", index=False)
    if plot:
        plt.figure()
        plt.plot(speed, taredrag, "-ok", markerfacecolor="None")
        plt.xlabel("Tow speed (m/s)")
        plt.ylabel("Tare drag (N)")
        plt.tight_layout()
        plt.show()


def process_tare_torque(nrun, plot=False):
    """Process a single tare torque run."""
    print("Processing tare torque run", nrun)
    times = {0 : (35, 86),
             1 : (12, 52),
             2 : (11, 32),
             3 : (7, 30)}
    nidata = loadhdf("Data/Raw/Tare-torque/" + str(nrun) + "/nidata.h5")
    # Compute RPM
    time_ni  = nidata["time"]
    angle = nidata["turbine_angle"]
    rpm_ni = fdiff.second_order_diff(angle, time_ni)/6.0
    rpm_ni = ts.smooth(rpm_ni, 8)
    try:
        t1, t2 = times[nrun]
    except KeyError:
        t1, t2 = times[3]
    meanrpm, _ = ts.calcstats(rpm_ni, t1, t2, 2000)
    torque = nidata["torque_trans"]
    meantorque, _ = ts.calcstats(torque, t1, t2, 2000)
    print("Tare torque =", meantorque, "Nm at", meanrpm, "RPM")
    if plot:
        plt.figure()
        plt.plot(time_ni, torque)
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.tight_layout()
        plt.show()
    return meanrpm, -meantorque


def process_strut_torque(nrun, zero_torque=0.0, plot=False, covers=False,
                         verbose=False):
    """Process a single strut torque run."""
    testplan = pd.read_csv("Config/Test plan/Strut-torque.csv",
                           index_col="run")
    ref_speed = testplan.ref_speed.iloc[nrun]
    tsr_nom = testplan.tsr.iloc[nrun]
    revs = testplan.revs.iloc[nrun]
    rpm_nom = tsr_nom*ref_speed/R/(2*np.pi)*60
    dur = revs/rpm_nom*60
    if covers:
        if verbose:
            print("Processing strut torque with covers run", nrun)
        nidata = loadhdf("Data/Raw/Strut-torque-covers/" + str(nrun) + \
                         "/nidata.h5")
    else:
        if verbose:
            print("Processing strut torque run", nrun)
        nidata = loadhdf("Data/Raw/Strut-torque/" + str(nrun) + "/nidata.h5")
    # Compute RPM
    time_ni  = nidata["time"]
    angle = nidata["turbine_angle"]
    rpm_ni = fdiff.second_order_diff(angle, time_ni)/6.0
    rpm_ni = ts.smooth(rpm_ni, 8)
    t1, t2 = 9, dur
    meanrpm, _ = ts.calcstats(rpm_ni, t1, t2, 2000)
    torque = nidata["torque_trans"]
    torque += calc_tare_torque(rpm_ni)
    meantorque, _ = ts.calcstats(torque, t1, t2, 2000)
    tsr_ref = meanrpm/60.0*2*np.pi*R/ref_speed
    if verbose:
        print("Reference TSR =", np.round(tsr_ref, decimals=4))
        print("Strut torque =", meantorque, "Nm at", meanrpm, "RPM")
    if plot:
        plt.figure()
        plt.plot(time_ni, torque)
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.tight_layout()
        plt.show()
    meantorque -= zero_torque
    ct = meantorque/(0.5*rho*A*R*ref_speed**2)
    cp = ct*tsr_ref
    summary = pd.Series()
    summary["run"] = nrun
    summary["tsr_ref"] = tsr_ref
    summary["cp"] = cp
    summary["mean_torque"] = meantorque
    summary["mean_rpm"] = meanrpm
    return summary


def batch_process_tare_torque(plot=False):
    """Process all tare torque data."""
    runs = os.listdir("Data/Raw/Tare-torque")
    runs = sorted([int(run) for run in runs])
    rpm = np.zeros(len(runs))
    taretorque = np.zeros(len(runs))
    for n in range(len(runs)):
        rpm[n], taretorque[n] = process_tare_torque(runs[n])
    df = pd.DataFrame()
    df["run"] = runs
    df["rpm"] = rpm
    df["tare_torque"] = taretorque
    df.to_csv("Data/Processed/Tare-torque.csv", index=False)
    m, b = np.polyfit(rpm, taretorque, 1)
    print("tare_torque = "+str(m)+"*rpm +", b)
    if plot:
        plt.figure()
        plt.plot(rpm, taretorque, "-ok", markerfacecolor="None")
        plt.plot(rpm, m*rpm + b)
        plt.xlabel("RPM")
        plt.ylabel("Tare torque (Nm)")
        plt.tight_layout()
        plt.show()


def batch_process_strut_torque(covers=False):
    section = "Strut-torque"
    if covers:
        section += "-covers"
    testplan = pd.read_csv("Config/Test plan/" + section + ".csv")
    df = []
    for run in testplan.run:
        df.append(process_strut_torque(run, covers=covers))
    df = pd.DataFrame(df)
    df.to_csv("Data/Processed/" + section + ".csv", index=False)


def make_remote_name(local_path):
    """Create top level file name for uploading to figshare.

    Note: Will only work properly on Windows"""
    return "_".join(local_path.split("\\")[-3:])


def download_raw(section, nrun, name):
    """Download a run's raw data.

    `name` can be either the file name with extension, or
      * `"metadata"` -- Metadata in JSON format
      * `"nidata"` -- Data from the NI DAQ system
      * `"acsdata"` -- Data from the tow tank's motion controller
      * `"vecdata"` -- Data from the Nortek Vectrino
    """
    if name == "metadata":
        filename = "metadata.json"
    elif name in ["vecdata", "nidata", "acsdata"]:
        filename = name + ".h5"
    else:
        filename = name
    print("Downloading", filename, "from", section, "run", nrun)
    local_dir = os.path.join("Data", "Raw", section, str(nrun))
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    local_path = os.path.join(local_dir, filename)
    remote_name = make_remote_name(local_path)
    with open("Config/raw_data_urls.json") as f:
        urls = json.load(f)
    url = urls[remote_name]
    pbar = progressbar.ProgressBar()
    def download_progress(blocks_transferred, block_size, total_size):
        percent = int(blocks_transferred*block_size*100/total_size)
        try:
            pbar.update(percent)
        except ValueError:
            pass
        except AssertionError:
            pass
    pbar.start()
    urlretrieve(url, local_path, reporthook=download_progress)
    pbar.finish()
