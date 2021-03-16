__precompile__(true)
"""
Main module for `Psychometrics.jl` -- A Julia package that provides tools for psychometric data analysis.

# Exports

    TruncatedInverseGaussian
    PolyaGamma
    TruncatedGamma
    rand
    AbstractExaminee
    Examinee
    AbstractLatent
    Latent1D
    LatentND
    AbstractItem
    Item
    AbstractParameters
    AbstractParametersBinary
    Parameters1PL
    Parameters2PL
    Parameters3PL
    ParametersNPL
    AbstractResponse
    Response
    probability
    log_likelihood
    likelihood
    latent_information
    item_observed_information
    item_expected_information
    add_prior!
    get_item_by_id
    get_parameters
    empty_chain!
    add_response!
    answer
    get_responses_by_item_id
    get_responses_by_item_idx
    get_responses_by_examinee_id
    get_response_matrix
    get_design_matrix
    get_responses
    get_examinee_by_id
    get_items_idx_answered_by_examinee
    get_examinees_idx_who_answered_item
    generate_w
    set_val_from_chain!
    set_val_from_posterior!
    chain_append!
    mcmc_iter_pg!
    update_estimate!
    posterior
    update_posterior!
    get_latents_vals
    get_latents
    find_best_item
    find_best_examinee
    calibrate_item!
    assess!
    joint_estimation!
"""
module Psychometrics

import Distributions
using Distributed
using Dates
using LinearAlgebra
using Interpolations
using NLopt
import Base.copy
using Requires

include("utils/math/utils_math.jl")
include("distributions/distributions.jl")
include("examinee/examinee.jl")
include("examinee/examinees.jl")
include("item/item.jl")
include("item/items.jl")
include("response.jl")
include("probability.jl")
include("likelihood.jl")
include("utils/polyagamma/utils_pg.jl")
include("posterior/posterior.jl")
include("information/information.jl")
include("online/online.jl")
include("utils/mmle/utils_mmle.jl")
include("item/calibration/calibration.jl")
include("examinee/assessment/assessment.jl")
include("joint_estimation/joint_estimation.jl")
include("bootstrap/bootstrap.jl")

export 
    TruncatedInverseGaussian,
    TruncatedNormal,
    PolyaGamma,
    TruncatedGamma,
    rand,
    AbstractExaminee,
    Examinee,
    AbstractLatent,
    Latent1D,
    LatentND,
    AbstractItem,
    Item,
    AbstractParameters,
    AbstractParametersBinary,
    Parameters1PL,
    Parameters2PL,
    Parameters3PL,
    ParametersNPL,
    AbstractResponse,
    Response,
    probability,
    log_likelihood,
    likelihood,
    latent_information,
    item_observed_information,
    item_expected_information,
    add_prior!,
    get_item_by_id,
    get_parameters,
    empty_chain!,
    add_response!,
    answer,
    get_responses_by_item_id,
    get_responses_by_item_idx,
    get_responses_by_examinee_id,
    get_response_matrix,
    get_design_matrix,
    get_responses,
    get_examinee_by_id,
    get_items_idx_answered_by_examinee,
    get_examinees_idx_who_answered_item,
    generate_w,
    set_val_from_chain!,
    set_val_from_posterior!,
    chain_append!,
    get_latents,
    find_best_item,
    find_best_examinee,
    calibrate_item!,
    assess_examinee!,
    joint_estimate!,
    rescale!,
    bootstrap!
end # module
