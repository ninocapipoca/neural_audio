# NeuralAudio

This toolbox is an implementation of an auditory model of the cochlea as developed by Shamma et al, first developed in MATLAB by Powen Ru and colleagues (Neural Systems Laboratory, University of Maryland). Their original documentation, containing the theoretical basis of the implementation, can be found [here](http://nsl.isr.umd.edu/Files/auditory.pdf).

## Key functions
- `wav2aud` permits the user to produce a spectrogram ('audiogram') in which each row represents the average spike count carried by an auditory nerve fiber
- `aud2cor` represents auditory information along four dimensions : scale, rate, time and frequency. This outputs a 4D complex-valued matrix stored in a binary file. Currently under development.

### Important to note
 - When filter coefficients were exported from MATLAB as a CSV and loaded in directly, they resulted in filters with significant numerical instability, even when using `sosfilt`. They were then exported in `zpk` format and this seemed to fix the problem, but the reason for this is unclear.
 - The behavior of other filters beyond those produced by the work of Shamma et al has not been tested. While it is possible to load in different filters, the function may not behave as expected. Custom filters requires converting the zpk information to a numpy-specific format (npz). See documentation for details.
 - The correctness of the functions was tested against the MATLAB implementation, and numerically corresponds to the MATLAB output with an absolute tolerance of 0.01.
