function calibrate_MMLE(responses::Vector{<:AbstractResponse}, examinees::Dict{Int64,<:Examinee1D}, items::Dict{Int64,<:AbstractItem}:
	K = 61,
	prior_latent = Normal(0,1),
	bounds_latent = [-6.0, 6.0],
	first_items = true
	)
	I = length(items)
	N = length(examinees)

	(X, W) = discretize(pior_latent, K = K, bounds = bounds_latent)

	nIndex= [map(r -> r.item_idx, filter(r2 -> r2.examinee_idx == key_e, responses)) for key_e in keys(examinees)]
	iIndex= [map(r -> r.examinee_idx, filter(r2 -> r2.item_idx == key_i, responses)) for key_i in keys(items)]
	
		########################################################################
		### initialize variables
		########################################################################
		startTime=time()

		###############################################################################
		### optimize (item parameters)
		###############################################################################
		BestLh=-Inf
		nloptalg=Symbol("LD_SLSQP")
		limES=0.5
		s=1
		endOfWhile=0
		latent_posterior = zeros(Float64, N, K)

		while endOfWhile<1
			
			if first_items
				################################################################
				####					MStep
				################################################################
				before_time=time()
				maxLHMMLE!(responses, items, latent_posterior, X, W)
				lhMat = [log_likelihood(filter(r -> r.item_idx == key_i, responses), examinees, Dict(key_i => i)) for (key_i, i) in items];
				println("time elapsed for Mstep intOpt ",time()-before_time)
				#################################################################
			else
				################################################################
				####					RESCALE
				################################################################
				#before_time=time()
				if mdl.eo.denType == "EH" && (mdl.eo.intW==0 && s<=mdl.eo.minMaxW[2] && s>=mdl.eo.minMaxW[1] )  && mdl.bs.BS==false# && xGap>0.5 && mdl.bs.BS==false
					posterior, newLh= compPost(posterior,lhMat,N,K,I,iIndex,newResponses,Wk,phi)
					Wk=LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
					observed=[LinearAlgebra.dot(Wk,Xk),sqrt(LinearAlgebra.dot(Wk,Xk.^2))]
					observed=[observed[1]-mdl.estimates.latents[1].metric[1],observed[2]/mdl.estimates.latents[1].metric[2]]
					#check mean
					#if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
						Xk2, Wk2=myRescale(Xk,Wk,mdl.estimates.latents[1].metric,observed)
						Wk=cubicSplineInt(Xk,Xk2,Wk2)
					#end
					newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				elseif typeof(mdl.eo.denType)==Distribution
					if s>=1
						for l=1:nLatent
							(bins,X[:,l+1])=cutR(newLatentVals[:,l+1];star=mdl.bds.minLatent[l], stop=mdl.bds.maxLatent[l], nBins=K+(toadd*2)-1, returnBreaks=false,returnMidPts=true)
							X[:,l+1]=X[(toadd+1):K+toadd,l+1]
							W[:,l+1]=pdf.(denType,X[:,l+1])
							W[:,l+1]=W[:,l+1]./sum(Wk)
						end
					end
				end
				#println("time elapsed for Rescale intOpt ",time()-before_time)
				################################################################
			end

			############################################################
			####					SAVE
			############################################################
			#lh
			oldLh[3]=oldLh[2]
			oldLh[2]=oldLh[1]
			oldLh[1]=copy(newLh)
			#pars
			deltaPars=(newPars-oldPars[1])./oldPars[1]
			oldPars[2]=oldPars[1]
			oldPars[1]=copy(newPars)
			#xGap
			xGap=maximum(abs.(deltaPars))
			bestxGap=min(xGap,xGapOld[1])
			xGapOld[6]=copy(xGapOld[5])
			xGapOld[5]=copy(xGapOld[4])
			xGapOld[4]=copy(xGapOld[3])
			xGapOld[3]=copy(xGapOld[2])
			xGapOld[2]=copy(xGapOld[1])
			xGapOld[1]=copy(bestxGap)
			oldLatentVals=copy(newLatentVals)
			############################################################

			#SECOND STEP
			if mdl.eo.first=="items"
				before_time=time()
				################################################################
				####					RESCALE
				################################################################
				if mdl.eo.denType=="EH" && (s%mdl.eo.intW==0 && s<=mdl.eo.minMaxW[2] && s>=mdl.eo.minMaxW[1])  && mdl.bs.BS==false
					posterior= compPostSimp(posterior,N,K,I,iIndex,newResponses,Wk,phi)
					Wk=LinearAlgebra.BLAS.gemv('T', one(Float64), posterior, oneoverN) #if Wk depends only on the likelihoods
					observed=[LinearAlgebra.dot(Wk,Xk),sqrt(LinearAlgebra.dot(Wk,Xk.^2))]
					observed=[observed[1]-mdl.estimates.latents[1].metric[1],observed[2]/mdl.estimates.latents[1].metric[2]]
					#check mean
					#if  (abs(observed[1])>1e-3 ) || observed[2]<0.99
						Xk2, Wk2=myRescale(Xk,Wk,mdl.estimates.latents[1].metric,observed)
						Wk=cubicSplineInt(Xk,Xk2,Wk2)
					#end
					newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				elseif typeof(mdl.eo.denType)==Distribution
					if s>=1
						for l=1:nLatent
							(bins,X[:,l+1])=cutR(newLatentVals[:,l+1];star=mdl.bds.minLatent[l], stop=mdl.bds.maxLatent[l], nBins=K+(toadd*2)-1, returnBreaks=false,returnMidPts=true)
							X[:,l+1]=X[(toadd+1):K+toadd,l+1]
							W[:,l+1]=pdf.(denType,X[:,l+1])
							W[:,l+1]=W[:,l+1]./sum(Wk)
						end
					end
				end
				println("time elapsed for rescale intOpt ",time()-before_time)
				################################################################
			else
				################################################################
				####					MStep
				################################################################
				before_time=time()
				newPars,  phi =maxLHMMLE(newPars,phi,posterior,iIndex,newDesign,X,Wk,newResponses,mdl.io,mdl.bds)
				lhMat=compLh(lhMat,N,K,iIndex,newResponses,phi)
				newLh=sum(log_c.(LinearAlgebra.BLAS.gemv('N', one(Float64), lhMat, Wk)))
				#println("time elapsed for Mstep intOpt",time()-before_time)

				################################################################
			end
			# if size(simData.θ,1)>0 && size(simData.pool,1)>0
			 #println("RMSE for pars is ",RMSE(newPars,newSd.pars))
			#println("RMSE for latents is ",RMSE(newLatentVals',newSd.latentVals'))
			#println("RMSE for θ is ",RMSE(newLatentVals,newSimθ))
			# end
			println("end of iteration  #",s)
			println("newlikelihood is ",newLh)

			newTime=time()
			####################################################################
			#                           CHECK CONVERGENCE
			####################################################################
			if (s >= mdl.eo.maxIter)
				println("maxIter reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
				# ItemPars=DataFrame(a=new_a,b=new_b)
				# Bestθ=copy(newLatentVals)
			end
			if newTime-oldTime>mdl.eo.timeLimit
				println("timeLimit reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
			end
			fGap=abs(newLh-oldLh[1])/oldLh[1]
			if fGap<mdl.eo.lTolRel && fGap>=0
				println("f ToL reached after ", newTime-oldTime," and ",Int(s)," iterations")
				endOfWhile=1
			end
			if s>3
				deltaPars=(newPars-oldPars[1])./oldPars[1]
				xGap=maximum(abs.(deltaPars))
				bestxGap=min(xGap,xGapOld[1])
				println("Max-change is ",xGap)
				if xGap<=mdl.eo.xTolRel
					println("X ToL reached after ", newTime-oldTime," and ",Int(s)," iterations")
					endOfWhile=1
				else
					if s>20
						if !(newLh>oldLh[1] && oldLh[1]>oldLh[2] && oldLh[2]>oldLh[3])
							if xGap<0.1 && xGapOld[1]==bestxGap && xGapOld[1]==xGapOld[2] && xGapOld[3]==xGapOld[2] && xGapOld[3]==xGapOld[4] && xGapOld[4]==xGapOld[5] && xGapOld[6]==xGapOld[5]
								endOfWhile=1
								println("No better result can be obtained")
							end
						end
					end
				end
			end
			if endOfWhile==1
				posterior= compPostSimp(posterior,N,K,I,iIndex,newResponses,Wk,phi)
				newLatentVals=(posterior*X)./(posterior*ones(K,nLatent+1))
			end
			s=s+1
			####################################################################

		end

		if mdl.bs.BS
			isample=findall(sum(newDesign,dims=2).>0)
			for p=1:nTotPar
				BSPar[p][isample,r+1].=newPars[p,:]
			end
			for l=1:nLatent
				BSLatentVals[l][nsample,r+1].=newLatentVals[:,l]
			end
		else
			mdl.prf.time=time()-startTime
			mdl.estimates.pars=newPars
			mdl.estimates.latentVals=newLatentVals
			mdl.estimates.latents[1].dist=Distributions.DiscreteNonParametric(Xk,Wk)
			mdl.prf.nIter=s-1
		end
		println("end of ",r, " bs replication")
	end
	if mdl.bs.BS
	JLD2.@save "pars.jld2" BSPar
	JLD2.@save "latentVals.jld2" BSLatentVals
	end

	return mdl
	# CSV.write("BS 1/estPoolGlobal.csv", ItemPars)
	# writedlm("BS 1/estAbilitiesGlobal.csv",Bestθ,'\t')
end