import scanpy as sc
def plot_umap(adata, colormodel):
    sc.pp.neighbors(adata)
    sc.tl.umap(adata, n_components=3)
    sc.pl.umap(adata, color=colormodel, components='1,2')
    sc.pl.umap(adata, color=colormodel, components='2,3')
    sc.pl.umap(adata, color=colormodel, components='3,1')

