"""
scRNA-seq utility functions for pseudobulk aggregation, pathway, and TF activity analysis.
"""

import random

import numpy as np
import pandas as pd
import decoupler as dc


def aggregate_and_filter(
    adata,
    group_key="sample",
    rep_key="sample_reps",
    replicates_per_group=3,
    num_cells_per_group=30,
):
    """
    Split cells into pseudobulk replicates per sample, dropping under-represented groups.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    group_key : str
        Column in `adata.obs` that identifies samples/groups. Default: "sample".
    rep_key : str
        Column name to write replicate labels into. Default: "sample_reps".
    replicates_per_group : int
        Number of pseudobulk replicates to create per group. Default: 3.
    num_cells_per_group : int
        Minimum number of cells a group must have to be retained. Default: 30.

    Returns
    -------
    AnnData
        Filtered AnnData with replicate labels in `adata.obs[rep_key]`.
    """
    size_by_group = adata.obs.groupby(group_key).size()
    groups_to_drop = [
        group
        for group in size_by_group.index
        if size_by_group[group] <= num_cells_per_group
    ]

    if groups_to_drop:
        print("Dropping the following samples:")
        print(groups_to_drop)

    adata.obs[rep_key] = "reps"
    adata.obs[group_key] = adata.obs[group_key].astype("category")

    groups = adata.obs[group_key].cat.categories
    for i, group in enumerate(groups):
        print(f"\tProcessing group {i + 1} out of {len(groups)}...", end="\r")
        if group not in groups_to_drop:
            adata_group = adata[adata.obs[group_key] == group]
            indices = list(adata_group.obs_names)
            random.shuffle(indices)
            splits = np.array_split(np.array(indices), replicates_per_group)
            for rep_i, rep_idx in enumerate(splits):
                adata.obs.loc[rep_idx, rep_key] = f"{group}_{rep_i}"

    adata = adata[adata.obs[rep_key] != "reps"].copy()
    print("\n")
    adata.obs[rep_key] = adata.obs[rep_key].astype("category")
    return adata


def pathway_tf_analysis(
    file,
    tables_dir="tables",
    tfs=None,
    pws=None,
):
    """
    Run TF and pathway activity scoring on DEG statistics using decoupler.

    Reads pre-computed DEG result CSVs, scores TF and pathway activity via
    univariate linear model (ULM), and writes the results back to CSV.

    Parameters
    ----------
    file : str
        Base filename (without extension) used to locate input CSVs and name outputs.
        Expects the following files under `tables_dir`:
          - ``{file}.csv``         full DEG results with a ``stat`` column
        tables_dir : str
        Directory containing the input CSVs and where outputs are written.
        Default: "tables".
    tfs : pd.DataFrame or None
        TF–target network for decoupler (e.g. CollecTRI). Must be provided.
    pws : pd.DataFrame or None
        Pathway–gene network for decoupler (e.g. MSigDB). Must be provided.
        Expected columns: ``source``, ``target``, ``collection``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(tf_df, pw_df)`` — scored TF and pathway activity DataFrames.
        Both are also saved as CSV files alongside the inputs.
    """
    if tfs is None or pws is None:
        raise ValueError("`tfs` and `pws` network DataFrames must be provided.")

    results_df = pd.read_csv(f"{tables_dir}/{file}.csv", index_col=0)
    results_df_filter_pv = results_df[results_df['padj'] < 0.05]
    results_df_filter_fc15 = results_df_filter_pv[(results_df_filter_pv['log2FoldChange'] > 0.58496250072) | (results_df_filter_pv['log2FoldChange'] < -0.58496250072)]
    results_df_filter_fc20 = results_df_filter_pv[(results_df_filter_pv['log2FoldChange'] > 1) | (results_df_filter_pv['log2FoldChange'] < -1)]
    
    data = results_df[["stat"]].T

    # --- TF activity ---
    tf_acts, tf_padj = dc.mt.ulm(data=data, net=tfs)
    tf_df = tf_acts.melt(value_name="score").merge(
        tf_padj.melt(value_name="pvalue")
        .assign(logpval=lambda x: -np.log10(x["pvalue"].clip(2.22e-4, 1)))
    )
    tf_df.to_csv(f"{tables_dir}/{file}_TF.csv", index=False)

    # --- Pathway activity ---
    pw_acts, pw_padj = dc.mt.ulm(data=data, net=pws)
    pw_df = pw_acts.melt(value_name="score").merge(
        pw_padj.melt(value_name="pvalue")
        .assign(logpval=lambda x: -np.log10(x["pvalue"].clip(2.22e-4, 1)))
    )
    pw_df = pw_df.merge(
        pws[["collection", "source"]].drop_duplicates(subset="source"),
        left_on="variable",
        right_on="source",
        how="left",
    ).drop(columns="source")

    pw_df = add_features_column(pw_df, pws, "features with 1.5 FC", results_df_filter_fc15)
    pw_df = add_features_column(pw_df, pws, "features with 2 FC", results_df_filter_fc20)
    pw_df.to_csv(f"{tables_dir}/{file}_PW.csv", index=False)

    return tf_df, pw_df


def add_features_column(pw_df, msigdb_all, feature_col, results_df):
    """
    Annotate a pathway results DataFrame with direction-matched genes.

    For each pathway, finds genes that (1) belong to the pathway, (2) are
    present in `results_df`, and (3) have a ``stat`` sign matching the
    pathway activity score.

    Parameters
    ----------
    pw_df : pd.DataFrame
        Pathway activity DataFrame with columns ``variable`` (pathway name)
        and ``score``.
    msigdb_all : pd.DataFrame
        Gene-set membership table with columns ``source`` (pathway) and
        ``target`` (gene).
    feature_col : str
        Name of the new column to add to `pw_df`.
    results_df : pd.DataFrame
        DEG results with a ``stat`` column indexed by gene name.

    Returns
    -------
    pd.DataFrame
        Copy of `pw_df` with `feature_col` added, containing lists of
        direction-matched genes per pathway.
    """
    pw_df_copy = pw_df.copy()
    pw_df_copy[feature_col] = [[] for _ in range(len(pw_df_copy))]

    pathway_to_genes = msigdb_all.groupby("source")["target"].apply(list).to_dict()
    gene_to_stat = results_df["stat"].to_dict()

    for idx, row in pw_df_copy.iterrows():
        genes_in_pathway = pathway_to_genes.get(row["variable"], [])
        score = row["score"]
        matching_genes = [
            gene
            for gene in genes_in_pathway
            if gene in gene_to_stat
            and (
                (score > 0 and gene_to_stat[gene] > 0)
                or (score < 0 and gene_to_stat[gene] < 0)
            )
        ]
        pw_df_copy.at[idx, feature_col] = matching_genes

    return pw_df_copy


def plot_tfs(file, tables_dir="tables", output_dir="pngs", title=None):
    """
    Plot the top significant TF activities as a barplot.
 
    Reads the TF activity CSV produced by `pathway_tf_analysis`, selects up to
    10 significant TFs (balancing up- and down-regulated), and saves a barplot.
 
    Parameters
    ----------
    file : str
        Base filename used to locate ``{tables_dir}/{file}_TF.csv`` and name
        the output ``{output_dir}/{file}_TFs.png``.
    tables_dir : str
        Directory containing the TF activity CSV. Default: "tables".
    output_dir : str
        Directory where the PNG is saved. Default: "pngs".
    title : str or None
        Plot title. Default: None.
    """
    tfs = pd.read_csv(f"{tables_dir}/{file}_TF.csv", index_col=0)
 
    sig = tfs[tfs["pvalue"] < 0.05]
    up = sig[sig["score"] > 0].nlargest(10, "score")
    down = sig[sig["score"] < 0].nsmallest(10, "score")
 
    n_up = min(len(up), 10 - min(len(down), 5))
    n_down = min(len(down), 10 - n_up)
 
    top_tfs = pd.concat([up.head(n_up), down.head(n_down)]).sort_values("score")
    tfs_select_score = top_tfs[["score"]].T
 
    dc.pl.barplot(
        data=tfs_select_score, name="score", top=len(top_tfs),
        figsize=(3, 3.5), dpi=300, vcenter=0,
    )
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file}_TFs.png", dpi=300, bbox_inches="tight")
    plt.show()
 
 
def plot_pws(file, tables_dir="tables", output_dir="pngs", rename_dict=None, title=None):
    """
    Plot the top significant pathway activities as a dotplot.
 
    Reads the pathway activity CSV produced by `pathway_tf_analysis`, filters
    to hallmark and KEGG collections, selects up to 16 significant pathways
    (balancing up- and down-regulated), and saves a dotplot.
 
    Parameters
    ----------
    file : str
        Base filename used to locate ``{tables_dir}/{file}_PW.csv`` and name
        the output ``{output_dir}/{file}_PW.png``.
    tables_dir : str
        Directory containing the pathway activity CSV. Default: "tables".
    output_dir : str
        Directory where the PNG is saved. Default: "pngs".
    rename_dict : dict or None
        Optional mapping to rename pathway labels before plotting. Default: None.
    title : str or None
        Plot title. Default: None.
    """
    if rename_dict is None:
        rename_dict = {}
 
    pws = pd.read_csv(f"{tables_dir}/{file}_PW.csv")
    pws = pws[pws["collection"].isin(["hallmark", "kegg_pathways"])]
 
    sig = pws[pws["pvalue"] < 0.05]
    up = sig[sig["score"] > 0].nlargest(14, "score")
    down = sig[sig["score"] < 0].nsmallest(14, "score")
 
    n_up = min(len(up), 16 - min(len(down), 8))
    n_down = min(len(down), 16 - n_up)
 
    top_pathways = pd.concat([up.head(n_up), down.head(n_down)]).sort_values("score")
    top_pathways["variable"] = top_pathways["variable"].replace(rename_dict)
 
    dc.pl.dotplot(
        df=top_pathways, x="score", y="variable", s="logpval", c="score",
        top=len(top_pathways), scale=0.6, figsize=(3, 4.8), vcenter=0,
    )
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file}_PW.png", dpi=300, bbox_inches="tight")
    plt.show()
 
