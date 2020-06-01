# Simulator

This folder contains the tests for the simulator.


## 4.5 Simulator validation


### ICESat-2 parameters

Photon rates (approx 1.0 for pure canopy and 1.7 for pure ground).

Footprint diameter (11 m)



## 4.5.1 Simulated RH metric comparison

Compare simulated ICESAt-2 RH metrics against simulated GEDI metrics, as a step towards determining whether a biomass model calibrated with RH metrics can be applied to a map of ICESat-2 metrics.

GEDI provides metrics over 22 m diamater footprints. ICESat-2 provides metrics over 100 m by 11 m transects. So ICESat-2 will see more trees over a transect and has a chance of larger values for the higher RH metrics. RH metrics are non-linearly related to the waveform, so that the mean of the RH metrics across a number of footprints is not necessarily equal to the RH metrics of a mean waveform.

To test this, there are a number of ways of calculating the RH metrcs from the two systems:

### ICESat-2

* With ground photons included
* Without ground photons included

The former will give a metric related to both height and cover and be more comparable to GEDI's metrics, but it relies upon the probability of ground and canopy photons for a given cover remaining stable over the area of interest. If that is not the case and it cannot be corrected, it may not be a stable metric. The latter will be a function purely of height, so would need an alternative measure of cover to predict biomass, but avoids any issues of the ground and canopy photon probabilities not being constant.


### GEDI

* Mean RH metrics across waveforms
* Mean log(RH) metrics across waveforms
* Mean exp(RH) metrics across waveforms
* Mean RH^2 metrics across waveforms
* Mean sqrt(RH) metrics across waveforms
* RH metric of mean waveforms

The former is the simplest, but is unlikely to work due to the non-linearity of RH metrics. The next four transformations will mimic the biomass model. If the biomass model is accurate, then the mean biomass of multiple footprints should give the correct biomass of an area, so linearly combining the RH metrics of the waveforms with the same non-linear transformation should give a mean RH metric that gives the correct mean biomass. The last combines all waveforms (after estimating ground elevation) and calculated the RH metrics for the mean waveform.


