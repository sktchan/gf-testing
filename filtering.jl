using Arrow, DataFrames, MultivariateStats, Statistics, LinearAlgebra, UMAP, SparseArrays


### load in data

full_dataset = Arrow.Table("/home/muninn/scratch/chans/test_serena/genecorpus_30M/genecorpus_30_2048.dataset/dataset.arrow")
df_full = DataFrame(full_dataset)

#= 
loaded genecorpus_30M into oni
need to run a = [row.input_ids[1] for row in eachrow(df_full)]
then save a as a separate column
once we have the first index of each cell for the column, we can see where len = 49
thus check if 0 was set for the rest of the ~2k genes or if it genuinely just doesn't have anything in it
also, see how the dataset was filtered/preprocessed to that amount of 10k cells
=#

using Base.Threads

# for multithreading, set ENV["JULIA_NUM_THREADS"] = 64 in repl

input_ids_vec = Vector(df_full.input_ids)
target_prefix = [5280, 16689]

function find_matching_row_parallel(input_ids_vec, target_prefix)
    n = length(input_ids_vec)
    row_index = nothing 
    @threads for i in 1:n
        if input_ids_vec[i][1:length(target_prefix)] == target_prefix
            row_index = i
            return
        end
    end
    return row_index
end

row_index = find_matching_row_parallel(input_ids_vec, target_prefix)

# row index: 25410609 with len 49 !!!

len_49_vector = collect(df_full[25410609, :].input_ids)
show(stdout, "text/plain", len_49_vector)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          