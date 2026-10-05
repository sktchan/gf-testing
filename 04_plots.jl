# 04: figures

using Pkg
Pkg.activate(joinpath(@__DIR__, "../transformers", Sys.ARCH == :aarch64 ? "aarch64" : "x86_64"))
using CSV, DataFrames, CairoMakie, Statistics, Printf, TOML
Makie.inline!(true) 
cd(@__DIR__)

res_dir = "results"
fig_dir = "results/figures/julia"; mkpath(fig_dir)
gf_name = "Geneformer (published)"
read_csv(f) = CSV.read(f, DataFrame, stringtype=String)

function cv_metrics(auc, wt)
    w = wt ./ sum(wt); m = sum(auc .* w)
    return m, sqrt(sum(w .* (auc .- m) .^ 2))
end

model_name(m) = get(Dict("Detection count" => "Count", "MLP" => "MLP-r"), m, m)
load_roc(f) = [(name=model_name(k.model), auc=first(g.auc), sd=first(g.sd), fpr=g.fpr, tpr=g.tpr)
               for (k, g) in pairs(groupby(read_csv(f), :model, sort=false))]
roc1 = load_roc("$res_dir/fig1_baseline_setup_roc.csv")    # 01
roc2 = load_roc("$res_dir/fig2_geneformer_setup_roc.csv")   # 02

# fold AUCs
read_folds(f, auc, wt) = select(read_csv("$res_dir/$f"), :model => ByRow(model_name) => :model, :fold, auc => :auc, wt => :wt)
fold_summary(F) = Dict(k.model => cv_metrics(g.auc, g.wt) for (k, g) in pairs(groupby(F, :model)))
fold_baseline = read_folds("01_baseline_setup_per_fold.csv", :auc_gene, :tpr_wt)
fold_gf = read_folds("02_geneformer_setup_per_fold.csv", :auc_occurrence, :tpr_wt_occurrence)
summ_baseline = fold_summary(fold_baseline)
summ_gf = fold_summary(fold_gf)
pub = read_csv("$res_dir/geneformer_published.csv")
summ_gf[gf_name] = (pub.weighted_auc[1], pub.weighted_sd[1])
fold_gf = vcat(DataFrame(model=gf_name, fold=pub.fold, auc=pub.auc, wt=missing), fold_gf)

# 03
s03 = vcat([read_csv("$res_dir/03_2026_setup/seed$s.csv") for s in (42, 0, 1)]...)
s03.model = model_name.(s03.model)
pub26 = TOML.parsefile("config/published.toml")["chen_2026"]

# styling
model_order = [gf_name, "MLP-r", "SVM-r", "RF-r", "LR-r", "Count"]
mcol(m) = Makie.wong_colors()[findfirst(==(m), model_order)]
name(m) = replace(m, " (published)" => "")   # display name: data keeps "(published)", figures drop it
short(m) = name(m)


# main: ROC of both setups (a: 01, b: 02), AUC of every fold behind the ± (c, d)
begin
    fig = Figure(size=(1500, 1050))
    Label(fig[0, 1:2], "Dosage-sensitive vs -insensitive TFs (2023)", font=:bold, tellwidth=false)

    # a: 01, baseline setup
    ax_a = Axis(fig[1, 1],
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="AUROC under baseline setup")
    lines!(ax_a, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=2)   # chance
    for c in roc1
        lines!(ax_a, c.fpr, c.tpr, color=mcol(c.name), linewidth=2, label=@sprintf("%s: %.2f ± %.2f", name(c.name), c.auc, c.sd))
    end
    axislegend(ax_a, position=:rb, "AUC ± SD by 5-fold cross-validation\nBaselines: balanced, eval. per gene\nGeneformer: unbalanced, eval. per occurrence")
    Label(fig[1, 1, TopLeft()], "a", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    # b: 02, geneformer's setup
    ax_b = Axis(fig[1, 2],
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="AUROC under Geneformer's setup")
    lines!(ax_b, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=2)
    for c in roc2
        lines!(ax_b, c.fpr, c.tpr, color=mcol(c.name), linewidth=2, label=@sprintf("%s: %.2f ± %.2f", name(c.name), c.auc, c.sd))
    end
    axislegend(ax_b, position=:rb, "AUC ± SD by 5-fold cross-validation\nUnbalanced, eval. per gene occurrence")
    Label(fig[1, 2, TopLeft()], "b", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    # c: 01 folds. dots: folds; tick ± error bar: fold-weighted AUC ± SD
    models = ["MLP-r", "SVM-r", "RF-r", "LR-r"]
    ax_c = Axis(fig[2, 1],
        xticks=(eachindex(models), short.(models)),
        ylabel="AUC",
        title="AUC per fold under baseline setup") # subtitle="dots: folds; line ± error bar: fold-weighted AUC ± SD"
    hlines!(ax_c, 0.5, color=:gray, linestyle=:dash, linewidth=2)
    for (i, m) in enumerate(models)
        g = fold_baseline[fold_baseline.model .== m, :]; mu, sd = summ_baseline[m]
        scatter!(ax_c, i .+ (g.fold .- 2) .* 0.07, g.auc, color=mcol(m))
        errorbars!(ax_c, [i], [mu], [sd], color=mcol(m), linewidth=2)
        scatter!(ax_c, [i], [mu], marker=:hline, markersize=25, color=mcol(m))
    end
    Label(fig[2, 1, TopLeft()], "c", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    # d: 02 folds
    models = [gf_name, "MLP-r", "SVM-r", "RF-r", "LR-r", "Count"]
    ax_d = Axis(fig[2, 2],
        xticks=(eachindex(models), short.(models)),
        ylabel="AUC",
        title="AUC per fold under Geneformer's setup")
    hlines!(ax_d, 0.5, color=:gray, linestyle=:dash, linewidth=2)
    for (i, m) in enumerate(models)
        g = fold_gf[fold_gf.model .== m, :]; mu, sd = summ_gf[m]
        scatter!(ax_d, i .+ (g.fold .- 2) .* 0.07, g.auc, color=mcol(m))
        errorbars!(ax_d, [i], [mu], [sd], color=mcol(m), linewidth=2)
        scatter!(ax_d, [i], [mu], marker=:hline, markersize=25, color=mcol(m))
    end
    Label(fig[2, 2, TopLeft()], "d", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    linkaxes!(ax_a, ax_b)     # a / b: same ROC axes
    linkyaxes!(ax_c, ax_d)    # c / d: same AUC axis (x = different models)
    rowsize!(fig.layout, 1, Relative(0.55))
    display(fig)
end
save("$fig_dir/fig_main.png", fig, px_per_unit=2)

# 01: ROC, baseline setup
begin
    fig1 = Figure(size=(750, 550))
    ax = Axis(fig1[1, 1],
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Dosage sensitive vs. insensitive TFs (2023)")
    lines!(ax, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=2)   # chance
    for c in roc1
        lines!(ax, c.fpr, c.tpr, color=mcol(c.name), linewidth=2, label=@sprintf("%s: %.2f ± %.2f", name(c.name), c.auc, c.sd))
    end
    axislegend(ax, position=:rb, "AUC ± SD by 5-fold cross-validation\nBaselines: balanced, eval. per gene\nGeneformer: unbalanced, eval. per occurrence")
    display(fig1)
end
save("$fig_dir/01_baseline_setup.png", fig1, px_per_unit=2)

# 02: ROC, geneformer's setup
begin
    fig2 = Figure(size=(750, 550))
    ax = Axis(fig2[1, 1],
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Dosage sensitive vs. insensitive TFs (2023)")
    lines!(ax, [0, 1], [0, 1], color=:gray, linestyle=:dash, linewidth=2)   # chance
    for c in roc2
        lines!(ax, c.fpr, c.tpr, color=mcol(c.name), linewidth=2, label=@sprintf("%s: %.2f ± %.2f", name(c.name), c.auc, c.sd))
    end
    axislegend(ax, position=:rb, "AUC ± SD by 5-fold cross-validation\nUnbalanced, eval. per gene occurrence")
    display(fig2)
end
save("$fig_dir/02_geneformer_setup.png", fig2, px_per_unit=2)

# 03: boxplots over 3 seeds; macro F1 (a) and AUC (b) (one 80/20 split, 3 seeds per model)
begin
    fig3 = Figure(size=(1100, 520))
    Label(fig3[0, 1:2], "Dosage-sensitive vs -insensitive TFs (2026)",
          font=:bold, tellwidth=false)
    # Label(fig3[2, 1:2], "box: median and quartiles; dots: seeds. (2026) = published values, different cells: context only",
        #   fontsize=12, tellwidth=false)
    groups = [
        ("Geneformer", mcol(gf_name), k -> pub26["Geneformer GF-10M (published)"][k]),
        [(m, mcol(m), k -> s03[s03.model .== m, k == "auc" ? :auc_occurrence : :macro_f1_occurrence]) for m in ["MLP-r", "SVM-r", "RF-r", "LR-r"]]...]
        # [("$(m)\n(2026)", (mcol("$m-r"), 0.35), k -> pub26["$m (published)"][k]) for m in ["SVM", "RF", "LR"]]...]   # Chen 2026's published baselines

    # a: macro F1
    ax = Axis(fig3[1, 1],
        xticks=(eachindex(groups), first.(groups)),
        ylabel="Macro-F1",
        limits=(nothing, (0.5, 1)))
    hlines!(ax, 0.5, color=:gray, linestyle=:dash, linewidth=2)   # chance
    for (i, (_, c, f)) in enumerate(groups)
        v = Float64.(collect(f("macro_f1")))
        boxplot!(ax, fill(i, length(v)), v, color=c, show_outliers=false, whiskerlinewidth=2, medianlinewidth=2)
        scatter!(ax, i .+ range(-0.12, 0.12, length=length(v)), v, color=:black, markersize=6)
    end
    Label(fig3[1, 1, TopLeft()], "a", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    # b: AUC
    ax = Axis(fig3[1, 2],
        xticks=(eachindex(groups), first.(groups)),
        ylabel="AUC",
        limits=(nothing, (0.5, 1)))
    hlines!(ax, 0.5, color=:gray, linestyle=:dash, linewidth=2)   # chance
    for (i, (_, c, f)) in enumerate(groups)
        v = Float64.(collect(f("auc")))
        boxplot!(ax, fill(i, length(v)), v, color=c, show_outliers=false, whiskerlinewidth=2, medianlinewidth=2)
        scatter!(ax, i .+ range(-0.12, 0.12, length=length(v)), v, color=:black, markersize=6)
    end
    Label(fig3[1, 2, TopLeft()], "b", font=:bold, padding=(0, 10, 10, 0), halign=:right)

    display(fig3)
end
save("$fig_dir/03_2026_setup.png", fig3, px_per_unit=2)