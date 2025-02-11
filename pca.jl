using Arrow, DataFrames, MultivariateStats, Statistics, LinearAlgebra, UMAP, SparseArrays


### load in data

dataset_labeled = Arrow.Table("/home/muninn/scratch/chans/test_serena/Geneformer/examples/250209142055/tf_dosage_sens_test_labeled.dataset/data-00000-of-00001.arrow")
df_labeled = DataFrame(dataset_labeled)


### reformatting matrix

features = []
labels_filtered = []

for row in eachrow(df_labeled)
    input_ids = row.input_ids
    labels = row.labels

    ## this removes all -100 values from the labels and accordingly with the input_ids. 
    # valid_indices = labels .!= -100 # removing -100 values
    # filtered_input_ids = input_ids[valid_indices]
    # filtered_labels = labels[valid_indices]

    # append!(features, filtered_input_ids)
    # append!(labels_filtered, filtered_labels)

    append!(features, input_ids)
    append!(labels_filtered, labels)
end

n_samples = length(labels_filtered)
n_features = maximum(df_labeled.length)


### pca, need to pad input!!

function pad_sequences(sequences, max_len, pad_value=-1)
    return [vcat(seq, fill(pad_value, max_len - length(seq))) for seq in sequences]
end

# converting input_ids column to a matrix of 2048x43501 (rows = genes, columns = cells)
X = reduce(hcat, pad_sequences(df_labeled.input_ids, n_features))'

pca_model = fit(PCA, X; maxoutdim=2) ### doesn't work


### umap

# converting to sparse matrix?
sparse_X = sparse(X)

embedding = umap(sparse_X, 2) ### also doesn't work... 