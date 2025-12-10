import anndata

def combine_adatas(adata1, adata2):
    adata_new = anndata.AnnData(
    X=adata1.X,
    obs=adata1.obs[['cell_type']].copy())
    adata_new.var_names = adata1.var_names
    combined_adata = anndata.concat([adata_new, adata2], join='inner')
    return combined_adata