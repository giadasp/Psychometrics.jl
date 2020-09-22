var documenterSearchIndex = {"docs":
[{"location":"item/#Items","page":"Items","title":"Items","text":"","category":"section"},{"location":"item/","page":"Items","title":"Items","text":"Modules = [Psychometrics]\nPages   = [\"item.jl\"]","category":"page"},{"location":"item/#Psychometrics.Item","page":"Items","title":"Psychometrics.Item","text":"Item <: AbstractItem\n\nDescription\n\nA generic item struct.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\ncontent::Vector{String}: A string vector containing the content features of an item.\nparameters::AbstractParameters: A generic item parameters object.\n\nInner Methods\n\nItem(idx, id, content, parameters) = new(idx, id, content, parameters)\n\nCreates a new 3PL item with custom index, id, content features and item parameters.\n\nRandom initilizers\n\nItem(idx, id, content) = new(idx, id, content, Parameters1PL())\n\nRandomly generates a new generic item with custom index, id, content features and default 1PL item parameters  (Look at (Parameters1PL)[#Psychometrics.Parameters1PL] for the defaults).\n\n\n\n\n\n","category":"type"},{"location":"item/#Psychometrics.Item1PL","page":"Items","title":"Psychometrics.Item1PL","text":"Item1PL <: AbstractItem\n\nDescription\n\nItem struct under the 1-parameter logistic model.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\ncontent::Vector{String}: A string vector containing the content features of an item.\nparameters::Parameters1PL: A Parameters1PL object.\n\nInner Methods\n\nItem1PL(idx, id, content, parameters) = new(idx, id, content, parameters)\n\nCreates a new 1PL item with custom index, id, content features and item parameters.\n\nRandom initilizers\n\nItem1PL(idx, id, content) = new(idx, id, content, Parameters1PL())\n\nRandomly generates a new 1PL item with custom index, id, content features and default 1PL item parameters  (Look at (Parameters1PL)[#Psychometrics.Parameters1PL] for the defaults).\n\n\n\n\n\n","category":"type"},{"location":"item/#Psychometrics.Item2PL","page":"Items","title":"Psychometrics.Item2PL","text":"Item2PL <: AbstractItem\n\nDescription\n\nItem struct under the 2-parameter logistic model.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\ncontent::Vector{String}: A string vector containing the content features of an item.\nparameters::Parameters2PL: A Parameters2PL object.\n\nInner Methods\n\nItem2PL(idx, id, content, parameters) = new(idx, id, content, parameters)\n\nCreates a new 2PL item with custom index, id, content features and item parameters.\n\nRandom initilizers\n\nItem2PL(idx, id, content) = new(idx, id, content, Parameters2PL())\n\nRandomly generates a new 2PL item with custom index, id, content features and default 2PL item parameters  (Look at (Parameters2PL)[#Psychometrics.Parameters2PL] for the defaults).\n\n\n\n\n\n","category":"type"},{"location":"item/#Psychometrics.Item3PL","page":"Items","title":"Psychometrics.Item3PL","text":"Item3PL <: AbstractItem\n\nDescription\n\nItem struct under the 3-parameter logistic model.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\ncontent::Vector{String}: A string vector containing the content features of an item.\nparameters::Parameters2PL: A Parameters3PL object.\n\nInner Methods\n\nItem3PL(idx, id, content, parameters) = new(idx, id, content, parameters)\n\nCreates a new 3PL item with custom index, id, content features and item parameters.\n\nRandom initilizers\n\nItem3PL(idx, id, content) = new(idx, id, content, Parameters3PL())\n\nRandomly generates a new 3PL item with custom index, id, content features and default 3PL item parameters  (Look at (Parameters3PL)[#Psychometrics.Parameters3PL] for the defaults).\n\n\n\n\n\n","category":"type"},{"location":"item/#Psychometrics.get_item_by_idx-Tuple{Int64,Array{var\"#s12\",1} where var\"#s12\"<:AbstractItem}","page":"Items","title":"Psychometrics.get_item_by_idx","text":"get_item_by_idx(item_idx::Int64, items::Vector{<:AbstractItem})\n\nIt returns the item with index item_idx from a vector of <:AbstractItem.\n\n\n\n\n\n","category":"method"},{"location":"likelihood/#Log-Likelihood","page":"Log Likelihood","title":"Log Likelihood","text":"","category":"section"},{"location":"likelihood/","page":"Log Likelihood","title":"Log Likelihood","text":"Modules = [Psychometrics]\nPages   = [\"likelihood.jl\"]","category":"page"},{"location":"likelihood/#Psychometrics.log_likelihood-Tuple{AbstractResponse,Array{Float64,1},Array{Float64,1}}","page":"Log Likelihood","title":"Psychometrics.log_likelihood","text":"log_likelihood(response::AbstractResponse, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nIt computes the log likelihood for a response response.  It updates also the gradient vectors.\n\n\n\n\n\n","category":"method"},{"location":"likelihood/#Psychometrics.log_likelihood-Tuple{AbstractResponse}","page":"Log Likelihood","title":"Psychometrics.log_likelihood","text":"log_likelihood(response::AbstractResponse, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nIt computes the log likelihood for a response response. \n\n\n\n\n\n","category":"method"},{"location":"likelihood/#Psychometrics.log_likelihood-Tuple{Array{var\"#s31\",1} where var\"#s31\"<:AbstractResponse,Array{Float64,1},Array{Float64,1}}","page":"Log Likelihood","title":"Psychometrics.log_likelihood","text":"log_likelihood(responses::Vector{<:AbstractResponse}, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nIt computes the log likelihood for a vector of responses responses.  It updates also the gradient vectors.\n\n\n\n\n\n","category":"method"},{"location":"likelihood/#Psychometrics.log_likelihood-Tuple{Latent1D,AbstractParameters,Float64,Array{Float64,1},Array{Float64,1}}","page":"Log Likelihood","title":"Psychometrics.log_likelihood","text":"log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nIt computes the log likelihood for a 1-dimensional latent variable and item parameters parameters with answer response_val.  It updates also the gradient vectors.\n\n\n\n\n\n","category":"method"},{"location":"likelihood/#Psychometrics.log_likelihood-Tuple{Latent1D,AbstractParameters,Float64}","page":"Log Likelihood","title":"Psychometrics.log_likelihood","text":"log_likelihood(latent::Latent1D, parameters::AbstractParameters, response_val::Float64)\n\nIt computes the log likelihood for a 1-dimensional latent variable and item parameters parameters with answer response_val.\n\n\n\n\n\n","category":"method"},{"location":"latent/#Latent-Variables","page":"Latent Variables","title":"Latent Variables","text":"","category":"section"},{"location":"latent/","page":"Latent Variables","title":"Latent Variables","text":"Modules = [Psychometrics]\nPages   = [\"latent.jl\"]","category":"page"},{"location":"information/#Fisher-Information","page":"Fisher Information","title":"Fisher Information","text":"","category":"section"},{"location":"information/","page":"Fisher Information","title":"Fisher Information","text":"Modules = [Psychometrics]\nPages   = [\"information.jl\"]","category":"page"},{"location":"information/#Psychometrics.expected_information_item-Tuple{Latent1D,Parameters1PL}","page":"Fisher Information","title":"Psychometrics.expected_information_item","text":"expected_information_item(latent::Latent1D, parameters::Parameters1PL)\n\nDescription\n\nIt computes the expected information (-second derivative of the likelihood) with respect to the difficulty parameter of the 1PL model.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 1-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.expected_information_item-Tuple{Latent1D,Parameters2PL}","page":"Fisher Information","title":"Psychometrics.expected_information_item","text":"expected_information_item(latent::Latent1D, parameters::Parameters2PL)\n\nDescription\n\nIt computes the expected information (-second derivative of the likelihood) with respect to the 2 parameters of the 2PL model.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 2-parameter logistic parameters object. \n\nOutput\n\nA 2 times 2 matrix of the expected informations. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.expected_information_item-Tuple{Latent1D,Parameters3PL}","page":"Fisher Information","title":"Psychometrics.expected_information_item","text":"expected_information_item(latent::Latent1D, parameters::Parameters3PL)\n\nDescription\n\nIt computes the expected information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. \n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA 3 times 3 matrix of the expected informations. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.expected_information_item-Tuple{Psychometrics.AbstractExaminee,AbstractItem}","page":"Fisher Information","title":"Psychometrics.expected_information_item","text":"expected_information_item(examinee::AbstractExaminee, item::AbstractItem)\n\nDescription\n\nAbstraction of expectedinformationitem(latent, parameters) on examinee and item.\n\nArguments\n\nexaminee::AbstractExaminee : Required. \nitem::AbstractItem : Required. \n\nOutput\n\nA matrix (or a scalar if there is only on item parameter) of the expected informations. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.information_latent-Tuple{Latent1D,Parameters1PL}","page":"Fisher Information","title":"Psychometrics.information_latent","text":"information_latent(latent::Latent1D, parameters::Parameters1PL)\n\nDescription\n\nIt computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 1PL model.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 1-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.information_latent-Tuple{Latent1D,Parameters2PL}","page":"Fisher Information","title":"Psychometrics.information_latent","text":"information_latent(latent::Latent1D, parameters::Parameters2PL)\n\nDescription\n\nIt computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 2PL model.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters2PL : Required. A 2-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.information_latent-Tuple{Latent1D,Parameters3PL}","page":"Fisher Information","title":"Psychometrics.information_latent","text":"information_latent(latent::Latent1D, parameters::Parameters3PL)\n\nDescription\n\nIt computes the information (-second derivative of the likelihood) with respect to the 1-dimensional latent variable under the 3PL model.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters3PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.information_latent-Tuple{Psychometrics.AbstractExaminee,AbstractItem}","page":"Fisher Information","title":"Psychometrics.information_latent","text":"information_latent(examinee::AbstractExaminee, item::AbstractItem)\n\nDescription\n\nAn abstraction of information_latent(latent::AbstractLatent, parameters::AbstractParameters) on examinee and item.\n\nArguments\n\nexaminee::AbstractExaminee : Required. \nitem::AbstractItem : Required. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.observed_information_item-Tuple{AbstractResponse}","page":"Fisher Information","title":"Psychometrics.observed_information_item","text":"observed_information_item(response::AbstractResponse)\n\nDescription\n\nIt computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. \n\nArguments\n\nresponse::AbstractResponse : Required. An instance of the AbstractResponse struct. \n\nOutput\n\nA matrix (or scalar if the item has only 1 parameter) of the observed informations. \n\n\n\n\n\n","category":"method"},{"location":"information/#Psychometrics.observed_information_item-Tuple{Float64,Latent1D,Parameters3PL}","page":"Fisher Information","title":"Psychometrics.observed_information_item","text":"observed_information_item(response_val::Float64, latent::Latent1D, parameters::Parameters3PL)\n\nDescription\n\nIt computes the observed information (-second derivative of the likelihood) with respect to the 3 parameters of the 3PL model. \n\nArguments\n\nresponse_val::Float64 : Required. A scalar response. \nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters3PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA 3 times 3 matrix of the observed informations. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Probability-(Item-Characteristic-Function)","page":"Probability (Item Characteristic Function)","title":"Probability (Item Characteristic Function)","text":"","category":"section"},{"location":"probability/","page":"Probability (Item Characteristic Function)","title":"Probability (Item Characteristic Function)","text":"Modules = [Psychometrics]\nPages   = [\"probability.jl\"]","category":"page"},{"location":"probability/#Psychometrics.probability-Tuple{Float64,Parameters1PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":" probability(latent_val::Float64, parameters::Parameters1PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 1PL model at latent_val point.\n\nArguments\n\nlatent_val::Float64 : Required. The point in the latent space in which compute the probability. \nparameters::Parameters1PL : Required. A 1-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Float64,Parameters2PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":" probability(latent_val::Float64, parameters::Parameters2PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 2PL model at latent_val point.\n\nArguments\n\nlatent_val::Float64 : Required. The point in the latent space in which compute the probability. \nparameters::Parameters2PL : Required. A 2-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Float64,Parameters3PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent_val::Float64, parameters::Parameters3PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 3PL model at latent_val point.\n\nArguments\n\nlatent_val::Float64 : Required. The point in the latent space in which compute the probability. \nparameters::Parameters3PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters1PL,Array{Float64,1},Array{Float64,1}}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent::Latent1D, parameters::Parameters1PL, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 1PL model at Latent1D point. It updates the gradient vectors if they are not empty.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 1-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters1PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":" probability(latent::Latent1D, parameters::Parameters1PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 1PL model at Latent1D point.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters1PL : Required. A 1-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters2PL,Array{Float64,1},Array{Float64,1}}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent::Latent1D, parameters::Parameters2PL, g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 2PL model at Latent1D point. It updates the gradient vectors if they are not empty.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters2PL : Required. A 2-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters2PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent::Latent1D, parameters::Parameters2PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 2PL model at Latent1D point.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters2PL : Required. A 2-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters3PL,Array{Float64,1},Array{Float64,1}}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent::Latent1D, parameters::Parameters3PL,  g_item::Vector{Float64}, g_latent::Vector{Float64})\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 3PL model at Latent1D point. It updates the gradient vectors if they are not empty.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters3PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"probability/#Psychometrics.probability-Tuple{Latent1D,Parameters3PL}","page":"Probability (Item Characteristic Function)","title":"Psychometrics.probability","text":"probability(latent::Latent1D, parameters::Parameters3PL)\n\nDescription\n\nIt computes the probability (ICF) of a correct response for item parameters under the 3PL model at Latent1D point.\n\nArguments\n\nlatent::Latent1D : Required. A 1-dimensional Latent1D latent variable. \nparameters::Parameters3PL : Required. A 3-parameter logistic parameters object. \n\nOutput\n\nA Float64 scalar. \n\n\n\n\n\n","category":"method"},{"location":"parameter/#Item-Parameters","page":"Item Parameters","title":"Item Parameters","text":"","category":"section"},{"location":"parameter/","page":"Item Parameters","title":"Item Parameters","text":"Modules = [Psychometrics]\nPages   = [\"parameters.jl\"]","category":"page"},{"location":"parameter/#Psychometrics.Parameters1PL","page":"Item Parameters","title":"Psychometrics.Parameters1PL","text":"Parameters1PL\n\nContains info about the difficulty of an item under the 1-parameter logistic model.\n\n\n\n\n\n","category":"type"},{"location":"parameter/#Psychometrics.add_posterior!-Tuple{AbstractParameters,Array{Distributions.Distribution{Distributions.Univariate,S} where S<:Distributions.ValueSupport,1}}","page":"Item Parameters","title":"Psychometrics.add_posterior!","text":"add_posterior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution})\n\nDescription\n\nIt transforms the vector posteriors of univariate distributions to their products and assign it to AbstractParameters instance.\n\nArguments\n\nparameters::AbstractParameters : Required. Any type of parameters object. \nposteriors::Vector{Distributions.UnivariateDistribution} : Required. A vector of probability distributions. The size of the vector must be the same as the number of parameters. \n\nExamples\n\nparameters2PL = Parameters2PL()\na_dist = Distributions.Normal(0,1)\nb_dist = Distributions.Normal(0,1)\nadd_posterior!(parameters2PL, [a_dist, b_dist])\n\n\n\n\n\n","category":"method"},{"location":"parameter/#Psychometrics.add_posterior!-Tuple{AbstractParameters,Distributions.Distribution}","page":"Item Parameters","title":"Psychometrics.add_posterior!","text":"add_posterior!(parameters::AbstractParameters, posterior::Distributions.Distribution)\n\nDescription\n\nIt assigns the <n>-variate posterior distribution to a AbstractParameters instance with <n> parameters.\n\nArguments\n\nparameters::AbstractParameters : Required. Any type of parameters object. \nposterior::Distributions.Distribution : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in parameters. \n\nExamples\n\nparameters2PL = Parameters2PL()\nbivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))\nadd_posterior!(parameters2PL, bivariate_normal)\n\n\n\n\n\n","category":"method"},{"location":"parameter/#Psychometrics.add_prior!-Tuple{AbstractParameters,Array{Distributions.Distribution{Distributions.Univariate,S} where S<:Distributions.ValueSupport,1}}","page":"Item Parameters","title":"Psychometrics.add_prior!","text":"add_prior!(parameters::AbstractParameters, priors::Vector{Distributions.UnivariateDistribution})\n\nDescription\n\nIt transforms the vector priors of univariate distributions to their products and assign it to AbstractParameters instance.\n\nArguments\n\nparameters::AbstractParameters : Required. Any type of parameters object. \npriors::Vector{Distributions.UnivariateDistribution} : Required. A vector of probability distributions. The size of the vector must be the same as the number of item parameters. \n\nExamples\n\nparameters2PL = Parameters2PL()\na_dist = Distributions.Normal(0,1)\nb_dist = Distributions.Normal(0,1)\nadd_prior!(parameters2PL, [a_dist, b_dist])\n\n\n\n\n\n","category":"method"},{"location":"parameter/#Psychometrics.add_prior!-Tuple{AbstractParameters,Distributions.Distribution{Distributions.Multivariate,S} where S<:Distributions.ValueSupport}","page":"Item Parameters","title":"Psychometrics.add_prior!","text":"add_prior!(parameters::AbstractParameters, prior::Distributions.Distribution)\n\nDescription\n\nIt assigns the prior prior to a AbstractParameters instance.\n\nArguments\n\nparameters::AbstractParameters : Required. Any type of parameters object. \nprior::Distributions.Distribution : Required. A <n>-variate probability distribution where <n> > 1 and is the numebr of item parameters in parameters. \n\nExamples\n\nparameters2PL = Parameters2PL()\nbivariate_normal = Distributions.MultivariateNormal([0,0], LinearAlgebra.I(2))\nadd_prior!(parameters2PL, bivariate_normal)\n\n\n\n\n\n","category":"method"},{"location":"","page":"Psychometrics.jl","title":"Psychometrics.jl","text":"CurrentModule = Psychometrics\nDocTestSetup = quote\n    using Psychometrics\nend","category":"page"},{"location":"#Psychometrics.jl","page":"Psychometrics.jl","title":"Psychometrics.jl","text":"","category":"section"},{"location":"","page":"Psychometrics.jl","title":"Psychometrics.jl","text":"A package for psychometric data analysis.","category":"page"},{"location":"#Documentation-Contents","page":"Psychometrics.jl","title":"Documentation Contents","text":"","category":"section"},{"location":"","page":"Psychometrics.jl","title":"Psychometrics.jl","text":"Pages = [\"index.md\", \"parameter.md\", \"item.md\", \"latent.md\", \"examinee.md\", \"response.md\", \"probability.md\",\"information.md\", \"likelihood.md\"]\nDepth = 3","category":"page"},{"location":"response/#Responses","page":"Responses","title":"Responses","text":"","category":"section"},{"location":"response/","page":"Responses","title":"Responses","text":"Modules = [Psychometrics]\nPages   = [\"response.jl\"]","category":"page"},{"location":"response/#Psychometrics.add_response!-Tuple{AbstractResponse,Array{Response,1}}","page":"Responses","title":"Psychometrics.add_response!","text":"add_response!(response::AbstractResponse, responses::Vector{Response})\n\nPush the response in the response vector responses.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.generate_response-Tuple{Latent1D,AbstractParameters}","page":"Responses","title":"Psychometrics.generate_response","text":"generate_response(latent::Latent1D, parameters::AbstractParameters)\n\nRandomly generate a response for a 1-dimensional latent variable and custom item parameters.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.generate_response-Tuple{Psychometrics.AbstractExaminee,AbstractItem}","page":"Responses","title":"Psychometrics.generate_response","text":"generate_response(examinee::AbstractExaminee, item::AbstractItem)\n\nRandomly generate a response by examinee to item.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.get_examinee_responses-Tuple{Int64,Array{Response,1}}","page":"Responses","title":"Psychometrics.get_examinee_responses","text":"get_examinee_responses(idx::Int64, responses::Vector{Response})\n\nIt returns the vector of responses given by examinee with index idx.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.get_examinees-Tuple{Int64,Array{var\"#s31\",1} where var\"#s31\"<:AbstractResponse}","page":"Responses","title":"Psychometrics.get_examinees","text":"get_examinees(item_idx::Int64, responses::Vector{<:AbstractResponse})\n\nIt returns the esaminees who answered to the item with index item_idx.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.get_item_responses-Tuple{Int64,Array{Response,1}}","page":"Responses","title":"Psychometrics.get_item_responses","text":"get_item_responses(idx::Int64, responses::Vector{Response})\n\nIt returns the vector of responses given to item with index idx.\n\n\n\n\n\n","category":"method"},{"location":"response/#Psychometrics.get_items-Tuple{Int64,Array{var\"#s31\",1} where var\"#s31\"<:AbstractResponse}","page":"Responses","title":"Psychometrics.get_items","text":"get_items(examinee_idx::Int64, responses::Vector{<:AbstractResponse})\n\nIt returns the items answered by the examinee with index examinee_idx.\n\n\n\n\n\n","category":"method"},{"location":"examinee/#Examinees","page":"Examinees","title":"Examinees","text":"","category":"section"},{"location":"examinee/","page":"Examinees","title":"Examinees","text":"Modules = [Psychometrics]\nPages   = [\"examinee.jl\"]","category":"page"},{"location":"examinee/#Psychometrics.Examinee","page":"Examinees","title":"Psychometrics.Examinee","text":"Examinee <: AbstractExaminee\n\nDescription\n\nExaminee struct with a generic latent variable.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\nlatent::Latent: A generic latent variable associated with the examinee.\n\nInner Methods\n\nExaminee1D(idx, id, latent) = new(idx, id, latent)\n\nCreates a new examinee with custom index, id and a generic latent variable.\n\n\n\n\n\n","category":"type"},{"location":"examinee/#Psychometrics.Examinee1D","page":"Examinees","title":"Psychometrics.Examinee1D","text":"Examinee1D <: AbstractExaminee\n\nDescription\n\nExaminee struct with a 1-dimensional latent variable.\n\nFields\n\nidx::Int64: An integer that identifies the examinee.\nid::String: A string that identifies the examinee.\nlatent::Latent1D: A 1-dimensional latent variable associated with the examinee.\n\nInner Methods\n\nExaminee1D(idx, id, latent) = new(idx, id, latent)\n\nCreates a new examinee with custom index, id and 1-dimensional latent variable.\n\n\n\n\n\n","category":"type"},{"location":"examinee/#Psychometrics.answer-Tuple{Array{var\"#s13\",1} where var\"#s13\"<:Psychometrics.AbstractExaminee,Array{var\"#s12\",1} where var\"#s12\"<:AbstractItem}","page":"Examinees","title":"Psychometrics.answer","text":"answer(examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})\n\nRandomly generate responses by all the examinees in examinees to items in items.\n\n\n\n\n\n","category":"method"},{"location":"examinee/#Psychometrics.answer-Tuple{Int64,Int64,Array{var\"#s31\",1} where var\"#s31\"<:Psychometrics.AbstractExaminee,Array{var\"#s32\",1} where var\"#s32\"<:AbstractItem}","page":"Examinees","title":"Psychometrics.answer","text":"answer(examinee_idx::Int64, item_idx::Int64, examinees::Vector{<:AbstractExaminee}, items::Vector{<:AbstractItem})\n\nRandomly generate a response by examinee with index examinee_idx to item with index item_idx.\n\n\n\n\n\n","category":"method"},{"location":"examinee/#Psychometrics.answer-Tuple{Psychometrics.AbstractExaminee,AbstractParameters}","page":"Examinees","title":"Psychometrics.answer","text":"answer(examinee::AbstractExaminee, item::AbstractParameters)\n\nRandomly generate a response by examinee to item.\n\n\n\n\n\n","category":"method"},{"location":"examinee/#Psychometrics.get_examinee_by_idx-Tuple{Int64,Array{var\"#s31\",1} where var\"#s31\"<:AbstractItem}","page":"Examinees","title":"Psychometrics.get_examinee_by_idx","text":"get_examinee_by_idx!(examinee_idx::Int64, examinees::Vector{<:AbstractExaminee})\n\nIt returns the examinee with index examinee_idx from a vector of ::AbstractExaminee.\n\n\n\n\n\n","category":"method"}]
}
