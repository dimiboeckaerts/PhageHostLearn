## FUNCTION TO ALIGN DATA
function create_SM(df)
    #=  INPUT:
        df =  Input dataframe of RBPs where each row has a protein sequence in column 'ProteinSeq'
        OUTPUT:
        n x n symmetric similarity matrix where n equals the number of sequences in InputFile   
    =#
        SM = Array{Float64}(undef, nrow(df),nrow(df))
        @printf("Aligning %.0f sequences. ", nrow(df) )
        k = 0
        scoremodel = AffineGapScoreModel(BLOSUM62, gap_open=-5, gap_extend=-1);
        @time begin
        Threads.@threads for i = 1:nrow(df)
        	k += 1
                localarray = zeros(i-1)
            s1 = df[i,:ProteinSeq];
            for j = i:nrow(df)
                s2 = df[j,:ProteinSeq];
                res = pairalign(LocalAlignment(), s1, s2, scoremodel)
                aln = alignment(res)
                if i!=j
                    if length(s1) < length(s2)
                        percentage = count_matches(aln)/length(s1)
                    else
                        percentage = count_matches(aln)/length(s2)
                    end
                else
                    percentage = 1
                end
                append!(localarray, percentage)
            end
            SM[:,i] = localarray
            @printf("%.0f ", k)
        end
        end
    return(SM)
    end