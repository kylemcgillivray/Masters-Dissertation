# Masters-Dissertation
Score based diffusion models.

Objective of thsi dissertation is to understand score based and diffusion models.

The benefit of score based models is that it circumvents the normalising constant of a density function via taking the logarithm and then the derivative of the unknown (to be estimated) density function. This preserves all of the information in the density as the map between the 2 is just differentiation/ integration and a bijective logarithm map.

We want to use langevin dynamics to eestimate the score funciton, but this method gives very poos approximations over low density regions of the density funciton. This is overcome by perturbing the dataset with some Gaussian random noise. We generalise this to a continuum of random noise ranging from no random noise at t=0 to asymptotically all random noise at t = T, and this is characterised through SDEs. Then we contruct BSDEs that map from a normal Gauss prior back to the underlying dataset. The drift term in the BSDE is then the score function. The objective is minimising an L2 loss function with is a weighted combination of the different perturbed noise levels. This can be efficiently evaluated via the slicing score matching method. 

Nested in the model are deep learning neural networks that are evaluated in one pass back propagation or automatic differentiation. 

To sample from the score function, we use Langevin dynamics, and when using sufficiently small timesteps, this guarantees correct sampling. 
