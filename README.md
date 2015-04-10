UNH RM2 tow tank experiment
===========================

This is a repository for data and processing code from the UNH RM2 tow tank 
experiment.

## Getting started
We recommend installing the 
[Anaconda Python distribution](http://continuum.io/downloads)
(Python 3.4) since it includes most dependencies. The remaining 
can be installed by executing

    pip install -r requirements.txt

After installing all dependencies, execute `python run.py` to generate
figures from the experiment. 

## Objectives
  1. Acquire a Reynolds number independent dataset characterizing the 
  performance and near-wake of the DOE RM2 cross-flow turbine, for validation of
  [CACTUS](http://energy.sandia.gov/?page_id=16734), and other prediction codes. 
  2. Measure the effects of strut drag, with the hope that this will improve
  CACTUS's performance prediction ability. 
  
## Test plan
* The [test plan document](https://drive.google.com/file/d/0BwMVIAlxIxfZZFR4cTVoRXdRNEU/view?usp=sharing) is online.
* CSV test matrices for each part of the experiment are located in `Config/Test plan`.

### Summary
  1. Build a scaled RM2 vertical axis turbine model.
  2. Measure performance curves to find a Reynolds number independent operating
  condition. 
  3. Measure the near-wake at the Reynolds number found in 2. 
  4. Measure the parasitic torque from the struts by rotating them in still water
  with the blades removed.
  5. Repeat 4 but for higher-drag struts. These will be cylindrical tubes
  slid over the regular struts.
  6. Acquire a performance curve with the higher-drag struts.
  
## Turbine model

|                | Full-scale | Model (1:6) |
| -------------  | ---------- | ----------- |
| Diameter (m)   | 6.45       |     1.075   |
| Height (m)     | 4.84       |     0.807   |
| Blade root chord (m) |  0.400  |     0.067   |
| Blade tip chord (m)  |  0.240  |     0.040   |
| Blade profile  | NACA 0021 |   NACA 0021 |
| Blade mount    | 1/2 chord |  1/2 chord  |
| Blade pitch (deg) | 0.0   |      0.0    |
| Strut profile | NACA 0021 |   NACA 0021 |
| Strut chord (m) |  0.360  |    0.060    |
| Shaft diameter (m) | 0.254 or 0.416 |   0.0635  |

### CAD files
Manufacturing drawings and 3-D models are
available from http://dx.doi.org/10.6084/m9.figshare.1373870

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by/4.0/88x31.png" />
</a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
Creative Commons Attribution 4.0 International License</a>.
