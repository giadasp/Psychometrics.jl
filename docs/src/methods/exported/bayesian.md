# Bayesian IRT

These functions allow to calibrate item parameters or estimate abilities under a unidimensional 2PL IRT model by using the PolyaGamma distribution approach introduced by [^Polson] and adapted to IRT models by [^JiangTemplin].

The documentation starts with an introduction on the basic tools to work on examinees and on items.
It follows a section with estimation/calibration tools for latents and item parameters separately.
At the end, an example of simultaneous estimation of item parameters and abilities is provided.

## Basic Tools for Examinees

First of all, we generate a groundtruth set of ``N = 100`` examinees with 1-dimensional latent ability sampled from a standardized Normal distribution.

```@example basic-tools-examinees, continued = true
N = 100
true_examinees = [Examinee(n, "examinee_$(n)") for n = 1 : N]
```

We can look at the values of the latent variables by printing the field `.latent.val`.

```@example basic-tools-examinees; continued = true
println(true_examinees[1].latent.val)
```

The field `.latent` is an instance of the `Latent1D` mutable struct (i.e. its subfields can be modified). It has the following subfields:

- **`val::Float64`**
- **`bounds::Vector{Float64}`**
- **`prior::Distributions.ContinuousUnivariateDistribution`**
- **`posterior::Distributions.ContinuousUnivariateDistribution`**
- **`chain::Vector{Float64}`**
- **`expected_information::Float64`**

Every latent object of such a type has structural properties such as a value and bounds, but also Bayesian properties, such as prior and posterior distributions, which are univariate in this case (thanks to `Distributions.jl`).

For example, the default factory for an examinee set a `Distributions.Normal(0,1)` both as prior and posterior of each of the examinees in `true_examinees`.

```@example basic-tools-examinees; continued = true
println(true_examinees[1].latent.prior)
println(true_examinees[1].latent.posterior)
```
## Basic Tools for Items

## Examinee Assessment

## Item Calibration

## Simultaneous Assessment and Calibration


[^Polson]: Polson, Nicholas G., Scott, James G., & Windle, J., (2013). Bayesian Inference for Logistic Models Using P olya–Gamma Latent Variables, Journal of the American Statistical Association, 108:504, 1339–1349, DOI: 10.1080/01621459.2013.

[^JiangTemplin]:  Jiang, Z., & Templin, J. (2019). Gibbs Samplers for Logistic Item Response Models via the Polya-Gamma Distribution: A Computationally Efficient Data-Augmentation Strategy.Psychometrika,84(2), 358-374.  DOI: 10.1007/s11336-018-9641-x.

```@meta
    CurrentModule = Psychometrics
```
```@docs
get_latents(examinees::Vector{<:AbstractExaminee})
empty_chain!(examinee::AbstractExaminee)
```
