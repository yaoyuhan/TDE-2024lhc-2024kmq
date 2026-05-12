#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:35:17 2026

@author: yuhanyao
"""
import os
import numpy as np
from copy import deepcopy

import astropy.units as u
from astropy.io import fits
import astropy.io.ascii as asci
from astropy.table import Table

from dust_extinction.parameter_averages import G23


def bin_spec(v4, y4, binning = 1, method = "sum"):
    if binning != 1:
        yy6 = deepcopy(y4)
        vv6 = deepcopy(v4)
        rest = len(yy6)%binning
        if rest!=0:
            vv6 = vv6[:(-1)*rest]
            yy6 = yy6[:(-1)*rest]
        nnew = int(len(yy6) / binning)
        yy6_new = yy6.reshape(nnew, binning)
        if method == "sum":
            yy6_new = np.sum(yy6_new, axis=1)
        elif method == "median":
            yy6_new = np.median(yy6_new, axis=1) * binning
        else:
            print ("error: method = %s not known"%method)
        y4 = yy6_new / binning
        vv6_new = vv6.reshape(nnew, binning)
        vv6_new = np.sum(vv6_new, axis=1)
        v4 = vv6_new / binning
    yy4 = np.repeat(y4, 2, axis=0)
    v4diff = np.diff(v4)
    v4diff_left = np.hstack([v4diff[0], v4diff])
    v4diff_right = np.hstack([v4diff, v4diff[-1]])
    vv4 = np.repeat(v4, 2, axis=0)
    vv4[::2] -= v4diff_left/2
    vv4[1::2] += v4diff_right/2
    return vv4, yy4


def read_spectrum_sdss(sdssdir, plate, mjd, fiberid):
    """
    sdss: 
        [0]: no data
        [1]: coadded spec
        [2]: obs info
        [3]: line flux
        [4]: B2-00027738-00027742-00027743
        [5]: B2-00027739-00027742-00027743
        [6]: B2-00027740-00027742-00027743 .... individual exposures
        
        bulk download: https://www.sdss.org/dr14/data_access/bulk/
    """
    subname = "spec-%4d-%5d-%4d.fits"%(plate, mjd, fiberid)
    subname = subname.replace(" ", "0")
    sdssfile = os.path.join(sdssdir, subname)
    
    hdus = fits.open(sdssfile)
    hdr0 = hdus[0].header
    data1 = hdus[1].data
    data2 = hdus[2].data
    flux = data1["flux"]
    wave = 10**data1["loglam"]
    or_mask = data1["or_mask"]
    and_mask = data1["and_mask"]

    # TAI     =        4507758314.18 / 1st row - Number of seconds since Nov 17 1858  
    # mjd = tai / 86400
    try:
        tai = hdr0["TAI"]
        jd = tai/86400 + 2400000.5
    except Exception:
        print ("Did not find TAI in header -- jd is not accurate")
        mjd = hdr0["MJD"]
        jd = mjd + 2400000.5
    dt = {"wave": wave,
          "flux": flux,
          "mask": and_mask,
          "subclass": Table(data2)["SUBCLASS"].data[0].replace(" ",""),
          "Z": float(data2["Z"]),
          "Z_ERR": float(data2["Z_ERR"]),
          "jd": jd,
          "ra": hdr0["PLUG_RA"],
          "dec": hdr0["PLUG_DEC"],
          "helio_rv": hdr0["HELIO_RV"]}
    return dt


def load_tde_specfiles(names = ["ZTF24aaoxmyb"], ext_corr = True,
                       skip_sdss = True, skip_hst = False):
    dt = {}
    specdir = "../spectra/"
    
    name = "ZTF24aaoxmyb"
    if name in names:
        dt[name] = {}
        
        myz = 0.2045
        dt[name]["z"] = myz
        dt[name]["ebv"] = 0.0233
        dt[name]["atname"] = "2024lhc"
        dt[name]["off"] = 1
        dt[name]["extraheight"] = 2.5
        
        i=0
        if skip_sdss==False:
            sp = read_spectrum_sdss(specdir+name+"/", plate=1175, mjd=52791, fiberid=192)
            dt[name]["sp%d"%i] = {}
            dt[name]["sp%d"%i]["date"] = "2003-06-01T10:10:30.43" # Time(sp["jd"], format = "jd").datetime64
            wave = np.array(sp["wave"], dtype = np.float64)
            flux = np.array(sp["flux"], dtype = np.float64)*1e-17
            ixno1 = (wave>5550)&(wave<5600)&(flux>1.5e-16)
            ixno2 = (wave>5550)&(wave<5600)&(flux<0.5e-16)
            ixno = ixno1 | ixno2
            dt[name]["sp%d"%i]["wave_obs"] = wave[~ixno]
            dt[name]["sp%d"%i]["flux_obs"] = flux[~ixno]
            dt[name]["sp%d"%i]["exptime"] = 0
            dt[name]["sp%d"%i]["instrument"] = "SDSS"
            dt[name]["sp%d"%i]["binning"] = 5
        
        i+=1
        sp = asci.read(specdir+name+"/desi_spectrum.dat") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2021-05-01 10:29:28.54"
        dt[name]["sp%d"%i]["wave_obs"] = sp["wave"].data
        dt[name]["sp%d"%i]["flux_obs"] = sp["flux"].data
        dt[name]["sp%d"%i]["eflux_obs"] = sp["eflux"].data
        dt[name]["sp%d"%i]["flux_model"] = sp["model"].data
        dt[name]["sp%d"%i]["mask"] = sp["mask"].data
        dt[name]["sp%d"%i]["exptime"] = 469.8
        dt[name]["sp%d"%i]["instrument"] = "DESI"
        dt[name]["sp%d"%i]["binning"] = 6
        
        i+=1
        sp = asci.read(specdir+name+"/ZTF24aaoxmyb-kast-ui-20240614.417%2Berr.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-06-14T10:00:28.8"
        wave = sp["col1"].data
        flux = sp["col2"].data
        eflux = sp["col3"].data
        dt[name]["sp%d"%i]["wave_obs"] = wave
        dt[name]["sp%d"%i]["flux_obs"] = flux
        dt[name]["sp%d"%i]["eflux_obs"] = eflux
        dt[name]["sp%d"%i]["exptime"] = 3600
        dt[name]["sp%d"%i]["instrument"] = "Kast"
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/ZTF24aaoxmyb-kast-ui-20240628.304%2Berr.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-06-28T07:17:45"
        wave = sp["col1"].data
        flux = sp["col2"].data
        eflux = sp["col3"].data
        ixno = np.isnan(flux) | np.isnan(eflux)
        dt[name]["sp%d"%i]["wave_obs"] = wave[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = flux[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = eflux[~ixno]
        dt[name]["sp%d"%i]["exptime"] = 3600
        dt[name]["sp%d"%i]["instrument"] = "Kast"
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/ZTF24aaoxmyb-lris-br-20240702.501%2Berr.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-07-02T12:01:26"
        wave = sp["col1"].data
        flux = sp["col2"].data
        eflux = sp["col3"].data
        dt[name]["sp%d"%i]["wave_obs"] = wave
        dt[name]["sp%d"%i]["flux_obs"] = flux
        dt[name]["sp%d"%i]["eflux_obs"] = eflux
        dt[name]["sp%d"%i]["exptime"] = 3600
        dt[name]["sp%d"%i]["instrument"] = "LRIS"
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/lris20240729_ZTF24aaoxmyb_o1.spec") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-07-29T09:08:38.4"
        ixno1 = np.isnan(sp["col2"].data)
        ixno2 = np.isinf(sp["col2"].data)
        ixno = ixno1 | ixno2
        dt[name]["sp%d"%i]["wave_obs"] = sp["col1"].data[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = sp["col2"].data[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col4"].data[~ixno]
        dt[name]["sp%d"%i]["exptime"] = 600
        dt[name]["sp%d"%i]["instrument"] = "LRIS"
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/ZTF24aaoxmyb-kast-ui-test-20240815.209+err.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-08-15T05:00:57"
        wave = sp["col1"].data
        flux = sp["col2"].data
        eflux = sp["col3"].data
        dt[name]["sp%d"%i]["wave_obs"] = wave
        dt[name]["sp%d"%i]["flux_obs"] = flux
        dt[name]["sp%d"%i]["eflux_obs"] = eflux
        dt[name]["sp%d"%i]["exptime"] = 3600
        dt[name]["sp%d"%i]["instrument"] = "Kast"
        dt[name]["sp%d"%i]["binning"] = 5
        
        i+=1
        sp = asci.read(specdir+name+"/ZTF24aaoxmyb-kast-test-ui-20240828.240+err.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-08-28T05:45:35"
        wave = sp["col1"].data
        flux = sp["col2"].data
        eflux = sp["col3"].data
        dt[name]["sp%d"%i]["wave_obs"] = wave
        dt[name]["sp%d"%i]["flux_obs"] = flux
        dt[name]["sp%d"%i]["eflux_obs"] = eflux
        dt[name]["sp%d"%i]["exptime"] = 3600
        dt[name]["sp%d"%i]["instrument"] = "Kast"
        dt[name]["sp%d"%i]["binning"] = 5
        
        i+=1
        sp = asci.read(specdir+name+"/at2024lhc-lris-br-20250305.602%2Berr.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2025-03-05T14:26:52"
        ixno1 = np.isnan(sp["col2"].data)
        ixno2 = np.isinf(sp["col2"].data)
        ixno = ixno1 | ixno2
        dt[name]["sp%d"%i]["wave_obs"] = sp["col1"].data[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = sp["col2"].data[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col3"].data[~ixno]
        dt[name]["sp%d"%i]["exptime"] = 1200
        dt[name]["sp%d"%i]["instrument"] = "LRIS"
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/at2024lhc-lris-br-20250601.523%2Berr.flm") 
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2025-06-01T12:33:07"
        ixno1 = np.isnan(sp["col2"].data)
        ixno2 = np.isinf(sp["col2"].data)
        ixno = ixno1 | ixno2
        dt[name]["sp%d"%i]["wave_obs"] = sp["col1"].data[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = sp["col2"].data[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col3"].data[~ixno]
        dt[name]["sp%d"%i]["exptime"] = 1200
        dt[name]["sp%d"%i]["instrument"] = "LRIS"
        dt[name]["sp%d"%i]["binning"] = 3
        
        if skip_hst is not True:
            i+=1
            filename = specdir+name+"/hst_17767_stis_at2024lhc_sg140l-sg230l_ofgu_cspec.fits"
            df = Table(fits.open(filename)[1].data)
            dt[name]["sp%d"%i] = {}
            dt[name]["sp%d"%i]["date"] = "2025-06-06T00:00:00"
            dt[name]["sp%d"%i]["wave_obs"] = df["WAVELENGTH"].data[0]
            dt[name]["sp%d"%i]["flux_obs"] = df["FLUX"].data[0]
            dt[name]["sp%d"%i]["eflux_obs"] = df["ERROR"].data[0]
            dt[name]["sp%d"%i]["snr"] = df["SNR"].data[0]
            dt[name]["sp%d"%i]["exptime"] = 1200
            dt[name]["sp%d"%i]["instrument"] = "HST"
            dt[name]["sp%d"%i]["binning"] = 10
        
        
    name = "ZTF24aapvieu"
    if name in names:
        dt[name] = {}
        
        myz = 0.2045
        dt[name]["z"] = myz
        dt[name]["ebv"] = 0.0169
        dt[name]["atname"] = "2024kmq"
        dt[name]["off"] = 1
        dt[name]["extraheight"] = 1.0
        
        i=0
        sp = asci.read(specdir+name+"/ZTF24aapvieu_20240608_GMOS-N_tellcorr.ascii")
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-06-08T07:15:42"
        ix = sp["col1"].data<9000
        dt[name]["sp%d"%i]["wave_obs"] = sp["col1"].data[ix]
        dt[name]["sp%d"%i]["flux_obs"] = sp["col2"].data[ix]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col3"].data[ix]
        dt[name]["sp%d"%i]["instrument"] = "GMOS"
        dt[name]["sp%d"%i]["exptime"] = 450
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/24kmq_20241128-29.dat")
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-11-29T00:40:13"
        dt[name]["sp%d"%i]["wave_obs"] = sp["wave"].data
        dt[name]["sp%d"%i]["flux_obs"] = sp["flux"].data
        dt[name]["sp%d"%i]["eflux_obs"] = sp["flux_err"].data
        dt[name]["sp%d"%i]["instrument"] = "Kast"
        dt[name]["sp%d"%i]["exptime"] = 3000
        dt[name]["sp%d"%i]["binning"] = 5
        
        i+=1
        sp = asci.read(specdir+name+"/20241208_ZTF24aapvieu_combine_3500.ascii")
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2024-12-08T05:25:41.21"
        wave = sp["col1"].data
        flux = sp["col2"].data
        ixno1 = wave < 3621
        ixno2 = wave > 9295
        ixno3 = (wave > 7575)&(wave < 7682)
        ixno4 = (wave > 6842)&(wave < 6922)
        ixno = ixno1 | ixno2 | ixno3 | ixno4
        dt[name]["sp%d"%i]["wave_obs"] = wave[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = flux[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col3"].data[~ixno]/10
        dt[name]["sp%d"%i]["instrument"] = "ALFOSC"
        dt[name]["sp%d"%i]["exptime"] = 3000
        dt[name]["sp%d"%i]["binning"] = 3
        
        i+=1
        sp = asci.read(specdir+name+"/at2024kmq-lris-br-20250625.305%2Berr.flm")
        dt[name]["sp%d"%i] = {}
        dt[name]["sp%d"%i]["date"] = "2025-06-25T07:19:12"
        ixno1 = np.isnan(sp["col2"].data)
        ixno2 = np.isinf(sp["col2"].data)
        ixno = ixno1 | ixno2
        dt[name]["sp%d"%i]["wave_obs"] = sp["col1"].data[~ixno]
        dt[name]["sp%d"%i]["flux_obs"] = sp["col2"].data[~ixno]
        dt[name]["sp%d"%i]["eflux_obs"] = sp["col3"].data[~ixno]
        dt[name]["sp%d"%i]["instrument"] = "LRIS"
        dt[name]["sp%d"%i]["exptime"] = 3000
        dt[name]["sp%d"%i]["binning"] = 3
        
    if ext_corr == True:
        extmod = G23(Rv=3.1)
        # perform Galactic extinction correction
        names = list(dt.keys())
        names = np.array(names)
        for i in range(len(names)):
            name = names[i]
            subdt = dt[name]
            ebv = subdt["ebv"]
            keys = list(subdt.keys())
            keys = np.array(keys)
            ix = np.array([x[:2]=="sp" for x in keys])
            keys = keys[ix]
            nsp = len(keys)
            for j in range(nsp):
                mykey = keys[j]
                mysp = subdt[mykey]
                wave = mysp["wave_obs"]
                flux = mysp["flux_obs"]
                Aextmag =  extmod(wave*u.angstrom)*3.1*ebv # extinction in magnitudes
                tau =  Aextmag / 1.086
                flux0 =  flux * np.exp(tau)
                dt[name][mykey]["flux0_obs"] = flux0
                if "eflux_obs" in mysp.keys():
                    eflux = mysp["eflux_obs"]
                    eflux0 =  eflux * np.exp(tau)
                    dt[name][mykey]["eflux0_obs"] = eflux0
                if "flux_model" in mysp.keys():
                    model = mysp["flux_model"]
                    model0 =  model * np.exp(tau)
                    dt[name][mykey]["flux_model0"] = model0
        
    return dt

