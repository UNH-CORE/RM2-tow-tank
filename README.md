UNH RM2 tow tank experiment
===========================

This is a repository for data and documentation from the UNH RM2 tow tank 
experiment. Currently the test plan is being finalized, and the turbine
model is being designed. 

## Objectives
  1. Acquire a Reynolds number independent dataset characterizing the 
  performance and near wake of the DOE RM2 cross-flow turbine, for validation of
  [CACTUS](http://energy.sandia.gov/?page_id=16734), and other prediction codes. 
  2. Measure the effects of strut drag, with the hope that this will improve
  CACTUS's performance prediction ability. 
  
## Test plan
See `Test plan/` for CSVs of test matrices for each part of the experiment. See
`Documents/Test plan/testplan.tex` for the test plan report. 

### Summary

  1. Build a scaled RM2 vertical axis turbine model, as faithfully as possible.
  2. Measure performance curves to find a Reynolds number independent operating
  condition. 
  3. Measure the near-wake at the Reynolds number found in 2. 
  4. Measure the parasitic torque from the struts by rotating them in still water
  with the blades removed.
  5. Repeat 4 but for higher-drag struts.
  5. Acquire a performance curve with the higher-drag struts.
  
## Turbine model

|              | Full-scale | Model (1:6) |
| -----------  | ---------- | ----------- |
| Diameter (m) | 6.45       |     1.075   |
| Height (m)   | 4.84       |     0.807   |
| Root chord (m) |  0.400   |     0.067   |
| Tip chord (m) |   0.240   |     0.040   |
| Blade profile | NACA 0021 |   NACA 0021 |
| Blade mount  | 1/2 chord? |  1/2 chord? |
| Blade pitch (deg) | 0.0?  |     0.0?    |
| Strut profile | NACA 0021 |   NACA 0021 |
| Strut chord (m) |  0.400? |    0.067?   |

### CAD files
CAD files of the turbine model will be uploaded to figshare once designed.

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="http://i.creativecommons.org/l/by/4.0/88x31.png" />
</a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">
Creative Commons Attribution 4.0 International License</a>.
