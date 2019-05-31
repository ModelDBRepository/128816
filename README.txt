This code is freely available for modifcation and extension, please
cite the original sources (see below).  For discussion and assistance
please contact: m.d.humphries@shef.ac.uk or drmdhumphries@gmail.com

**********************************************************************

This MATLAB code builds the medial reticular formation (mRF) anatomy
models detailed in Humphries, Gurney & Prescott (2006) [HGP06]. The
main function (discrete_cluster1.m) constructs either form of the
"stochastic" model from HGP06. The function pruning_model.m constructs
either form of the "pruning" model from HGP06: it calls
discrete_cluster1.m to construct the first, overgrowth phase of the
model.

The main function (discrete_cluster1.m) also includes options to
simulate the constructed model as a network of leaky integrator
neurons (aka firing rate units) to examine its basic dynamics. This
simulation model is sketched in Humphries, Gurney & Prescott (2010)
[HGP10]; in that paper we looked at how the fidelity of input-output
encoding in the mRF model depended on the scale of inputs and outputs
one was considering. The MSc thesis of Donoso (2008) is a preliminary
study of the relationship between the mRF model's structure and the
oscillatory dynamics that result.

The reasons for studying the mRF as a potential brainstem action
selection system are reviewed in Humphries et al (2007) and Humphries,
Gurney & Prescott (2010)

**********************************************************************

Main Files:

discrete_cluster1.m: builds the mRF anatomy model using the
		     "stochastic" algorithm; optionally runs that
		     model as a dynamic system

pruning_model.m: builds the mRF anatomy model using the "pruning"
    algorithm example_run_of_model.m: shows how to specify all the
    parameters, and run a single instance of the model, with
    interesting dynamics.
example_script_using_parameters_from_MNAS_book_chapter: an alternative
    specification of the model, as used in HGP10
cluster_input_IO_patterns: runs the complete set of cluster input
    simulations, examining the fidelity of input-output responses
    (from HGP10, Fig5c,d) proj_input_IO_patterns: runs the complete
    set of projection-unit input simulations, examining the fidelity
    of input-output responses (from HGP10, Fig5a,b)

Support Functions (folder LI_network_toolbox, make sure this is on
your MATLAB path):

LI_network.m: called by discrete_cluster1.m to run the mRF anatomy
	      model as a network of leaky integrators; this in turn
	      calls the MEX function LI_network_C
LI_network_C.c: the source C-code for the MEX function; this is
	        supplied compiled as both 32-bit (.dll) and 64-bit
	        (.mexw64) Windows versions. It is strongly suggested
	        that this function is recompiled for your platform to
	        avoid problems: type "help mex" for information at the
	        MATLAB prompt
LI_network_ode.m: optionally called by discrete_cluster1.m to run the
		  mRF model as a network of leaky integrators, using
		  MATLAB's built-in ODE solver, rather than the custom
		  MEX file. Useful for checking the consistency of
		  results, but orders-of-magnitude slower than the MEX
		  version

Requires:
Statistics Toolbox (for normrnd and gamrnd functions: these are easily
replaced or found)

***********************************************************************
Open questions - some suggestions for ideas to pursue

Structure:
(1) How do the graph properties of the network scale with the number
of neurons? Due to computational limitations at the time, the mRF
anatomy model was built up to a maximum of 3750 nodes (75 clusters and
50 neurons per cluster).  Yet the number of neurons-per-cluster is
likely to be at least an order of magnitude bigger than this. [Any
anatomical model]

(2) What are the effects of changing the power-law exponent on
connectvity in the distance-dependent model? [Stochastic model]

(3) The pruning model has numerous parameters that remain unexplored:
	(i) Choice of starting model (spatially-uniform vs
	distance-dependent probability of connections)
	(ii) choice of initial weight distribution (width of Gaussian;
	log-normal distributions etc)
	(iii) Choice of parameters for weight updating
	(iv) Threshold for pruning

(4) How might we alter the weight update algorithm for the pruning
model?

Dynamics:

(1) What are the effects of weight choice and distribution on the
intrinsic dyna mics? (Oscillations, stability etc)
(2) How do the different construction algorithms affect the resultant
dynamics?
(3) What are the parameter regimes dividing stable from oscillatory
states?
(4) How could the intrinsic dynamics of the model support "selection
of actions" ?
(5) And, of course, what are the dynamics of the system if we using
spiking neuron models?

*****************************************************************************

References:

Humphries, M. D., Gurney, K. & Prescott, T. J. (2006) The brainstem
reticular formation is a small-world, not scale-free,
network. Proceedings of the Royal Society B. Biological Sciences, 273,
503-511. [PDF supplied with code]

Humphries, M. D., Gurney, K. & Prescott, T. J. (2007) Is there a
brainstem substrate for action selection?  Phil Trans R Soc B, 2007,
362, 1627-1639. [PDF supplied with code]

Donoso, J. R. (2008) Dynamics of the brainstem action selection
systems. MSc Thesis, University of Sheffield. Supervisor: Dr M
Humphries. [PDF supplied with code]

Humphries, M. D., Gurney, K. & Prescott, T. (2010) The medial
reticular formation: a brainstem substrate for simple action
selection? In A. K. Seth, T. J. Prescott and J. J. Bryson (Eds)
Modelling Natural Action Selection. Cambridge, UK: CUP. In press. [PDF
supplied with code]
