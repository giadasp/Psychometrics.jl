```@meta
CurrentModule = Psychometrics
DocTestSetup = quote
    using Psychometrics
end
```

# Bayesian IRT

These functions allow to calibrate item parameters or estimate abilities under a unidimensional 2PL IRT model by using the PolyaGamma distribution approach introduced by [^Polson] and adapted to IRT models by [^JiangTemplin].

The documentation starts with an introduction on the basic tools to work on examinees and on items.
It follows a section with estimation/calibration tools for latents and item parameters, separately.
At the end, an example of simultaneous estimation of item parameters and abilities is provided.

## Basic Tools for Examinees

An examinee is a physical or virtual person who is intended to answer the questions of a test (the items) for measuring an underlying personal latent trait. The purposes of the test may be various, from ability assessment, to clinical diagnosis.
In this example, a set of virtual examinees is generated and their properties and dedicated methods are described.

First of all, we generate a groundtruth set of ``N = 100`` examinees with 1-dimensional latent ability sampled from a standardized Normal distribution. We use the default factory ``Examinee()`` to create a virtual respondent:

```@example basic-tools-examinees, continued = true
N = 100
true_examinees = [Examinee(n, "examinee_$(n)") for n = 1 : N]
```

We can look at the examinees' latent traits by analysing the field `latent`.
In this case, the field `latent` is an instance of the `Latent1D` mutable struct (i.e. its subfields can be modified). It has the following subfields:

- **`val::Float64`**
- **`bounds::Vector{Float64}`**
- **`prior::Distributions.ContinuousUnivariateDistribution`**
- **`posterior::Distributions.ContinuousUnivariateDistribution`**
- **`likelihood::Float64`**
- **`chain::Vector{Float64}`**
- **`expected_information::Float64`**

Any latent object of such a type has structural properties, such as value and bounds, but also Bayesian properties, such as prior and posterior distributions 
(which, in this case, are specified as univariate thanks to `Distributions.jl`).

The default factory for an examinee returns an examinee with a `Distributions.Normal(0,1)` as prior (and posterior) for their ability. The value of their ability is randomly extracted from the posterior distribution. In our example, the examinees in `true_examinees` are all default examinees. Thus, we can see that their latent properties have been set to the defaults:

```@example basic-tools-examinees; continued = true
println(true_examinees[1].latent.prior)
println(true_examinees[1].latent.posterior)
println(true_examinees[1].latent.val)
```

The method `get_latents()` can be used to extract the latents of all the examinees in a vector. While, `get_latent_vals()` produces a matrix with ``N`` rows and ``L`` columns, where ``L`` is the maximum latent dimension among the examinees (1 in our example, since all the examinees have a latent of type `Latent1D`). Let's look at how they work on the first 10 respondents:

```@example basic-tools-examinees; continued = true
println(get_latents(true_examinees[1:10]))
println(get_latent_vals(true_examinees[1:10]))
```

## Basic Tools for Items

## Examinee Assessment

## Item Calibration

## Simultaneous Assessment and Calibration


[^Polson]: Polson, Nicholas G., Scott, James G., & Windle, J., (2013). Bayesian Inference for Logistic Models Using P olya–Gamma Latent Variables, Journal of the American Statistical Association, 108:504, 1339–1349, DOI: 10.1080/01621459.2013.

[^JiangTemplin]:  Jiang, Z., & Templin, J. (2019). Gibbs Samplers for Logistic Item Response Models via the Polya-Gamma Distribution: A Computationally Efficient Data-Augmentation Strategy.Psychometrika,84(2), 358-374.  DOI: 10.1007/s11336-018-9641-x.

