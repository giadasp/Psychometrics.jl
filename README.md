# Psychometrics.jl

This package is a playground I built for working on my research projects.
It provides objects (Julia structs) and dedicated methods (functions) to deal with psychometric data under the item response theory (IRT) statistical paradigm.

The documentation is still in progress. Nevertheless, some functions have detailed descriptions that can be found in the documentation pages or using the Julia help `?`.

## Structs

- `Item`: an immutable containing information about an item (a question in a test), e.g. `id::String` the item identifier, `calibrated::Bool` says if it is a field item (`false`) or an operational item (`true`), and the field `parameters::AbstractParameter` which accepts a mutable item parameter object (see below).
 - `Examinee`: an immutable containing the information about the examinee (the test taker).
 The field `latent::AbstractLatent` accepts a mutable latent variable (see below).
 - `AbstractParameters`: an abstract type which, at the moment, has only the abstract `AbstractParametersBinary` as subtype.
 Thus, it works only with binary (dichotomous) responses. The available mutables for the latter are `Parameters1PL`, `Parameters2PL`, `Parameters3PL`.
 They contain the details about the item parameters under the 1-parameter logistic (1PL), 2-parameters logistic (2PL) and 3-parameter logistic (3PL) IRT models, respectively.
 For example, the object `Parameters2PL` has the fields difficulty `b::Float64` and discrimination `a::Float64`. It is possible to define the Bayesian priors and posterior by assigning multivariate distributions from the package `Distrubutions` to the fields `prior::Distributions.MultivariateDistribution` and `posterior::Distributions.MultivariateDistribution`.
 - `Latent1D` and `LatentND`: mutables describing an univariate or multivariate latent variable, respectively.
 For the univariate case, the field `val::Float64` holds the estimate of the ability of the examinee, like the item parameters, also the latent variables can have a prior or a posterior, assigning them to the fields `prior` and `posterior`, respectively.
 - `ResponseBinary`: an immutable which holds the information about a binary (correct or incorrect) response, such as the identifier of the examinee who gave the response `examinee_id::String`, the identifier of the answered item `item_idx::String`, and the answer starting and ending times `start_time::Dates.DateTime` and `end_time::Dates.DateTime`.
 
Each of the mentioned structs has a random default factory, callable by using the name of the struct followed by `()`.

_Example: `Parameters2PL()` generates a set of 2PL item parameters from the product distribution (independent bivariate)_
```
Distributions.Product([
   Distributions.LogNormal(0, 0.25),
   Distributions.Normal(0, 1)
])
```

The immutable structs `Item` and `Examinee` need the identification fields `id` and `idx` to be randomly generated. 

_Example: `Item(1, "item_1")` generates an operational item with an empty `content` description, and a default 1PL item parameter._

## Basic functions

  - Item characteristic function (ICF): `probability`.
  - Latent and item information function (IIF) (expected and observed are different for the 3PL model): `expected_information_item`,  `observed_information_item`, `information_latent`.
  - Likelihood function: `likelihood`, `log_likelihood`.
  
  
  
 




 
