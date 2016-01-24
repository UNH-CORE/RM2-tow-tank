# -*- coding: utf-8 -*-
"""This module contains test functions."""

import os
import time
from .processing import *
from .plotting import *
import subprocess
from pandas.util.testing import assert_frame_equal


def test_turbine_geometry():
    assert D == 1.075
    assert R == 0.5375
    assert H == 0.807
    assert A == D*H


def test_run():
    print("Testing Run class")
    run = Run("Wake-1.0-0.0", 20)
    print(run.cp_per_rev)
    print(run.std_cp_per_rev)
    print(run.cp_conf_interval)
    print(run.mean_cp)
    print(run.unc_cp)
    print(run.exp_unc_cp)
    run.print_perf_stats()


def test_section():
    print("Testing Section class")
    section = Section("Wake-1.0-0.0")


def test_batch_process_section():
    print("Testing batch_process_section")
    batch_process_section("Perf-1.0")


def test_perf_curve():
    pc = PerfCurve(0.6)


def test_wake_profile():
    wp = WakeProfile(1.0, 0.25, "horizontal")


def test_wake_map():
    wm = WakeMap()


def test_process_section_parallel():
    nproc = 2
    nruns = 8
    s = Section("Wake-1.0-0.0")
    s.process(nproc=nproc, nruns=nruns, save=False)
    df_parallel = s.data.copy()
    s.process(nproc=1, nruns=nruns, save=False)
    df_serial = s.data.copy()
    assert_frame_equal(df_parallel, df_serial)


def test_download_raw():
    """Test the `processing.download_raw` function."""
    print("Testing processing.download_raw")
    # First rename target file
    fpath = "Data/Raw/Perf-1.0/0/metadata.json"
    fpath_temp = "Data/Raw/Perf-1.0/0/metadata-temp.json"
    exists = False
    if os.path.isfile(fpath):
        exists = True
        os.rename(fpath, fpath_temp)
    try:
        download_raw("Perf-1.0", 0, "metadata")
        # Check that file contents are equal
        with open(fpath) as f:
            content_new = f.read()
        if exists:
            with open(fpath_temp) as f:
                content_old = f.read()
            assert(content_new == content_old)
    except ValueError as e:
        print(e)
    os.remove(fpath)
    if exists:
        os.rename(fpath_temp, fpath)


def test_process_tare_torque():
    process_tare_torque(2, plot=False)


def test_process_tare_drag():
    process_tare_drag(5, plot=False)


def test_plots():
    subprocess.call(["python", "plot.py", "--all", "--noshow"])
