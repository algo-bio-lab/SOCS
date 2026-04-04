import numpy as np
import pandas as pd
import scipy
import anndata as ad
import diffxpy.api as de
import pickle
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection
from socs.utils import row_normalize
from scipy.stats import entropy
import shapely
import copy

def hillNumber_1(A):
    """
    Computes the Hill number of order 1 for the rows of the input matrix A

    Parameters
    ----------
    A: np.ndarray
        Input array
    
    Returns
    -------
    hn_1: np.ndarray
        Vector of Hill numbers of order 1 for the rows of A. hn_1[i] is the order-1 Hill number of the ith row of A
    """
    shannon_e = entropy(A,axis=1)
    hn_1 = np.exp(shannon_e)
    return hn_1

def map_vector_sampled(vec,T):
    """
    Sample from a vector of values associated with cells in the target distribution, using probabilities computed by a transport map

    Parameters
    ----------
    vec: np.ndarray
        Vector of values associated with cells in the target distribution.
    T: np.ndarray
        Transport map from a source distribution to a target distribution.
    
    Returns
    -------
    vec_mapped: list
        Vector of values sampled from vec: interpreted as the value associated with source distribution cells' descendants.
    """
    T_n = row_normalize(T)
    vec_mapped = []
    range_1 = np.arange(T_n.shape[1])
    for x in range(T_n.shape[0]):
        rx_bin = np.random.choice(range_1,p=T_n[x,:])
        vec_mapped.append(vec[rx_bin])
    return vec_mapped
    
def vec2vec(v1,v2,T):
    """
    Using map_vector_sampled, identifies the number of cells belonging to categories in the source distribution which map to
    categories in the target distribution

    Parameters
    ----------
    v1: np.ndarray
        Vector of category labels associated with cells in the source distribution.
    v2: np.ndarray
        Vector of category labels associated with cells in the target distribution.
    T: np.ndarray
        Transport map from source distribution to target distribution.
    
    Returns
    -------
    vec2vec: np.ndarray
        Array of numbers of cells belonging to each category in the source distribution mapping to each category in the target
        distribution. vec2vec[i,j] givs the number of cells in category i in the source distribution that map to category j in
        the target distribution.
    """
    v1_vals = list(np.unique(v1))
    v2_vals = list(np.unique(v2))
    v2_mapped = map_vector_sampled(v2,T)
    vec2vec = np.zeros([len(v1_vals),len(v2_vals)])
    for x in range(T.shape[0]):
        ind_x = v1_vals.index(v1[x])
        ind_x_mapped = v2_vals.index(v2_mapped[x])
        vec2vec[ind_x,ind_x_mapped]+=1
    return vec2vec

def struct_average(adata,geneName,struct_key='struct'):
    """
    Computes the average expression of a specified gene over each labeled structure in the dataset

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object representing a Spatial Transcriptomics dataset
    geneName: str
        Gene for which to compute average expression
    struct_key: str
        Column name in adata.obs in which cells' structure labels are stored

    Returns
    -------
    adata_r: anndata.AnnData
        A copy of the input adata, with an added column 'struct_{geneName}' giving the average gene expression in the structure
        associated with each cell.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    struct_ids = np.unique(adata_r.obs[struct_key])
    avg_mkr = np.zeros([len(struct_ids),])
    for x in range(len(struct_ids)):
        avg_mkr[x] = np.mean(adata_r[adata_r.obs[struct_key]==struct_ids[x],:].X[:,adata_r.var_names==geneName],axis=0)
    adata_r.uns['follicles']['avg_'+geneName] = avg_mkr
    struct_avg_mkr = [avg_mkr[list(struct_ids).index(x)] for x in adata.obs[struct_key].tolist()]
    adata_r.obs['struct_'+geneName] = struct_avg_mkr
    return adata_r


def add_struct_df(adata,struct_key='struct'):
    """
    Adds a pandas DataFrame to adata.uns, where rows are labeled structures in the dataset.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    struct_key: str
        Column in adata.obs in which structure labels are found.

    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with pandas DataFrame with a row representing each labeled structure added as adata_r.uns['structs']
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        structs_labels = np.array(adata_r.obs[struct_key].tolist())
        structs_u = np.unique(structs_labels)
        structs_df = pd.DataFrame(index=structs_u)
        adata_r.uns['structs'] = structs_df
    return adata_r



def struct_average_obs(adata,obs_name,struct_key='struct'):
    """

   Computes the average value of a column in adata.obs for cells associated with each of the labeled structures, and adds this value
    as a column in adata.uns['struct'].

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    obs_name: str
        The name of the column in adata.obs to be averaged over the cells in each structure.
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.uns['structs'] giving the average value of the specified column
        in adata.obs for cells associated with each of the labeled structures..
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata)
    struct_ids = adata_r.uns['struct'].index.tolist()
    avg_mkr = np.zeros([len(struct_ids),])
    for x in range(len(struct_ids)):
        inds_x = np.where(adata.obs[struct_key]==struct_ids[x])[0]
        avg_mkr[x] = np.mean(adata[inds_x,:].obs[obs_name],axis=0)
    adata_r.uns['structs']['avg_'+obs_name] = avg_mkr
    return adata_r

def structSize(adata,struct_key='struct'):
    """
    Gets the number of cells associated with each labeled structure, and adds this value as a column in adata.obs and adata.uns['struct']

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.obs, 'size' giving the number of cells in the structure associated with
        each cell. A column, 'size' is also added to adata_r.uns['structs'] giving the number of cells in each structure.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    struct_labels = np.array(adata_r.obs[struct_key].tolist())
    structs_u = adata_r.uns['structs'].index.tolist()
    nStructs = len(structs_u)
    struct_size = []
    for x in range(nStructs):
        struct_size.append(len(np.where(struct_labels==structs_u[x])[0]))
    adata_r.uns['structs']['size'] = struct_size
    return adata_r


def struct_diameter_sweep(xy):
    """
    Estimates the average diameter of a structure by calculating the radius of the convex hull while sweeping the radial angle

    Parameters
    ----------
    xy: np.ndarray
        spatial coordinates of the cells belonging to the structure
    
    Returns
    -------
    diam_avg: float
        Estimate of the average diameter of the structure
    """
    hull_fval = ConvexHull(xy)
    hull_vertices = list(hull_fval.vertices)#+[hull_f.vertices[0]]
    poly_hull_fval = shapely.geometry.Polygon(shell=xy[hull_fval.vertices,:])
    ctr_fval = poly_hull_fval.centroid.xy
    max_lens = np.zeros([180,])
    for x in range(180):
        slp = np.tan(np.deg2rad(x))
        pt1 = shapely.geometry.Point([ctr_fval[0][0]+100,ctr_fval[1][0]+(100*slp)])
        pt2 = shapely.geometry.Point([ctr_fval[0][0]-100,ctr_fval[1][0]-(100*slp)])
        diam_len = np.zeros([100,])
        for y in range(100):
            pt1_y = shapely.geometry.Point([pt1.xy[0][0]+(-50+y),pt1.xy[1][0]-((-50+y)*(1/(np.finfo(float).eps+slp)))])
            pt2_y = shapely.geometry.Point([pt2.xy[0][0]+(-50+y),pt2.xy[1][0]-((-50+y)*(1/(np.finfo(float).eps+slp)))])
            line_y = shapely.geometry.LineString([pt1_y,pt2_y])
            diam_y = shapely.intersection(poly_hull_fval,line_y)
            diam_len[y] = diam_y.length
        max_lens[x] = np.max(diam_len)
        max_ind = np.argmax(diam_len)
    diam_avg = np.mean(max_lens) 
    return diam_avg 

def struct_diameters_angles(adata,struct_key='struct'):
    """
    Estimates the diameter of each labeled structure, and adds this value as a column in adata.uns['struct']

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.uns['structs'] giving the diameter of each structure.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    struct_labels = np.array(adata_r.obs[struct_key].tolist())
    structs_u = adata_r.uns['structs'].index.tolist()
    nStructs = len(structs_u)
    struct_diams = np.zeros([nStructs,])
    xy = adata_r.obsm['spatial']
    for x in range(nStructs):
        inds_x = np.where(struct_labels==structs_u[x])[0]
        if(len(inds_x)>2):
            xy_x = xy[inds_x,:]
            struct_diams[x] = struct_diameter_sweep(xy_x)
    adata_r.uns['structs']['diameter'] = struct_diams
    return adata_r

def get_ctr(xy):
    """
    Gets the centroid of a list of 2-D spatial coordinates
    """
    c_x = np.sum(xy[:,0])/xy.shape[0]
    c_y = np.sum(xy[:,1])/xy.shape[0]
    return [c_x,c_y]

def struct_centroids(adata,spatial_key='spatial',struct_key='struct'):
    """
    Computes the centroid of each labeled structure, and adds this value as a column in adata.uns['struct']

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    spatial_ley: str
        Key in adata.obsm in which spatial coordinates are found.
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with columns added to adata_r.uns['structs'] giving the x-coordinate and y-coordinate of the 
        centroid  of each structure.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    struct_labels = np.array(adata_r.obs[struct_key].tolist())
    structs_u = adata_r.uns['structs'].index.tolist()
    nStructs = len(structs_u)
    struct_ctrs = np.zeros([nStructs,2])
    xy = adata_r.obsm[spatial_key]
    for x in range(nStructs):
        inds_x = np.where(struct_labels==structs_u[x])[0]
        if(len(inds_x)!=0):
            xy_x = xy[inds_x,:]
            ctr_x = get_ctr(xy_x)
            struct_ctrs[x,:] = ctr_x
    adata_r.uns['structs']['centroid_x'] = struct_ctrs[:,0]
    adata_r.uns['structs']['centroid_y'] = struct_ctrs[:,1]
    return adata_r

def follicle_radial_dist(adata,spatial_key='spatial',struct_key='struct'):
    """
    Computes the distance of each labeled structure from the centroid of the entire sample, and adds this value as a column in adata.uns['struct']

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    spatial_key: str
        Key in adata.obsm in which spatial coordinates are found.
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.uns['structs'] giving the distances of each structure to the centroid
        of the sample.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    if 'centroid_x' not in adata_r.uns['structs']:
        adata_r = struct_centroids(adata_r)
    xy_all = get_ctr(adata_r.obsm[spatial_key])
    struct_labels = np.array(adata_r.obs['struct'].tolist())
    structs_u = adata_r.uns['structs'].index.tolist()
    nStructs = len(structs_u)
    struct_rads = np.zeros([nStructs,])
    for x in range(nStructs):
        inds_x = np.where(struct_labels==structs_u[x])[0]
        if(len(inds_x)!=0):
            ctr_xx = adata.uns['structs']['centroid_x'][structs_u[x]]
            ctr_xy = adata.uns['structs']['centroid_y'][structs_u[x]]
            ctr_x = [ctr_xx,ctr_xy]
            struct_rads[x] = np.linalg.norm(np.array(ctr_x)-np.array(xy_all))
    adata_r.uns['structs']['rad_dist'] = struct_rads
    return adata_r

    
def cell_edge_dist(adata,filenames,spatial_key='spatial',struct_key='struct'):
    """
    Computes the minimum distance of each cell to a boundary defined by an ordered set of coordinates, and adds this value as a column in adata.obs
    named "edge_dist".

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    filenames: list
        list of filenames, each of which should point to a csv file containing the coordinates of a boundary
    spatial_key: str
        Key in adata.obsm in which spatial coordinates are found.
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.obs giving the distances of each cell to the boundary it is closest to.
    """
    adata_r = adata.copy()
    edge_dists = np.zeros([adata_r.shape[0],len(filenames)])
    pts = [shapely.geometry.Point(adata_r.obsm[spatial_key][g,0],adata_r.obsm[spatial_key][g,1]) for g in range(adata_r.shape[0])]
    for x in range(len(filenames)):
    #for x in range(len(regions)):
        #bounds = pd.read_csv('//broad/clearylab/Users/Peter/shalekOvary/follicleBoundaries/ovary_boundaries_{}.csv'.format(regions[x]))
        bounds = pd.read_csv(filenames[x])
        boundary_line = shapely.geometry.LineString(bounds)
        edge_dists[:,x] = shapely.distance(boundary_line,pts)
    adata_r.obs['edge_dist'] = np.min(edge_dists,axis=1)
    return adata_r

def struct_edge_dist(adata,filenames,spatial_key='spatial',struct_key='struct'):
    """
    Computes the minimum distance of the centroid of each structure to a boundary defined by an ordered set of coordinates, 
    and adds this value as a column in adata.uns['structs'] named "edge_dist".

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    filenames: list
        list of filenames, each of which should point to a csv file containing the coordinates of a boundary
    spatial_key: str
        Key in adata.obsm in which spatial coordinates are found.
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with a column added to adata_r.uns['structs'] giving the distances of each
         structure to the boundary it is closest to.
    """
    adata_r = adata.copy()
    if 'structs' not in adata_r.uns:
        adata_r = add_struct_df(adata_r)
    if 'centroid_x' not in adata_r.uns['structs']:
        adata_r = struct_centroids(adata_r,spatial_key=spatial_key,struct_key=struct_key)
    edge_dists = np.zeros([adata_r.uns['structs'].shape[0],len(filenames)])
    pts = [shapely.geometry.Point(adata_r.uns['structs']['centroid_x'].iloc[x],adata_r.uns['structs']['centroid_y'].iloc[x]) for x in range(adata_r.uns['structs'].shape[0])]
    for x in range(len(filenames)):
        #bounds = pd.read_csv('//broad/clearylab/Users/Peter/shalekOvary/follicleBoundaries/ovary_boundaries_{}.csv'.format(regions[x]))
        bounds = pd.read_csv(filenames[x])
        boundary_line = shapely.geometry.LineString(bounds)
        edge_dists[:,x] = shapely.distance(boundary_line,pts)
    adata_r.uns['structs']['edge_dist'] = np.min(edge_dists,axis=1)
    return adata_r

def structs_to_cells(adata,struct_key='struct'):
    """
    For each column in adata.uns['structs'], adds a column to adata.obs giving the value of that column for the structure associated
    with each cell.

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object with labeled structures
    struct_key: str
        Column in adata.obs in which structure labels are found.
    
    Returns
    -------
    adata_r: anndata.AnnData
        Copy of input AnnData object, with columns added to adata_r.obs giving the value of each column of adata.uns['structs'] for the
        structure associated with each cell.
    """
    adata_r = adata.copy()
    struct_columns = adata_r.uns['structs'].columns.tolist()
    for y in struct_columns:
        cell_struct_data = np.zeros([adata.shape[0],])
        for x in range(adata.shape[0]):
            cell_struct_data[x] = adata.uns['structs'].loc[adata.obs[struct_key][x]][y]
        adata_r.obs['struct_'+y] = cell_struct_data
    return adata_r


def get_deg_bool(deg_test,min_fc,min_q):
    logfc = deg_test.log2_fold_change()
    qvals = -deg_test.log10_qval_clean(log10_threshold=-30)
    inds_lfc = np.where(np.abs(logfc)<10)[0]
    logfc_l = logfc[inds_lfc]
    qvals_l = logfc[inds_lfc]
    up_s = np.logical_and(qvals_l>min_q,logfc_l.T>min_fc)
    up_s = np.where(up_s)[0]
    down_s = np.logical_and(qvals_l>min_q,(-logfc_l.T)>min_fc)
    down_s = np.where(down_s)[0]
    up_row = np.zeros([1,len(logfc)])
    down_row = np.zeros([1,len(logfc)])
    inds_up = [inds_lfc[x] for x in up_s]
    inds_down = [inds_lfc[x] for x in down_s]
    up_row[0,inds_up] = 1
    down_row[0,inds_down] = 1
    return up_row,down_row

def in_radius(xy_1,xy_2,r):
    dm_xy = np.ascontiguousarray(pairwise.pairwise_distances(xy_1,Y=xy_2,metric='euclidean',n_jobs=1))
    inds_r = []
    ds_r = []
    for x in range(xy_1.shape[0]):
        inds_x = np.where(dm_xy[x,:]<r)[0].tolist()
        ds_x = dm_xy[x,inds_x].tolist()
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        if(len(ind_i)!=0):
            del inds_x[ind_i[0]]
            del ds_x[ind_i[0]]
        inds_r.append(inds_x)
        ds_r.append(ds_x)
    return ds_r,inds_r

def neighbors_delaunay_thresholded(xy_1,xy_2,t):
    tri = Delaunay(xy_2)
    inds_d = []
    dm_xy = np.ascontiguousarray(pairwise.pairwise_distances(xy_1,Y=xy_2,metric='euclidean',n_jobs=1))
    for x in range(xy_1.shape[0]):
        ds_x = dm_xy[x,:]
        #ind_x2 = xy_2.tolist().index(xy_1[x,:].tolist())
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        tri_x = tri.simplices==ind_i
        inds_x = np.where(np.sum(tri_x,axis=1))[0]
        tri_xi = tri.simplices[inds_x,:]
        inds_xu = np.unique(tri_xi.flatten()).tolist()
        ds_xu = ds_x[inds_xu]
        inds_t = np.where(ds_xu<t)[0]
        inds_xu = [inds_xu[x] for x in inds_t]
        inds_xu.remove(ind_i)
        inds_d.append(inds_xu)
    return inds_d


def neighbors_delaunay_t_exclude(xy_1,xy_2,xy_e,t):
    inds_nbrs = neighbors_delaunay_thresholded(xy_1,xy_2,t)
    all_inds = []
    for x in inds_nbrs:
        for y in x:
            all_inds.append(y)
    all_inds_u = np.unique(all_inds)
    all_inds_u_s = set(all_inds_u.tolist())
    nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(xy_2)
    ds,inds = nbrs.kneighbors(xy_e)
    inds_match = []
    for x in range(len(ds)):
        ds_x = ds[x].tolist()
        inds_x = inds[x].tolist()
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        if(len(ind_i)!=0):
            inds_match.append(inds_x[ind_i[0]])
    inds_match_s = set(inds_match)
    inds_only_nbrs_s = all_inds_u_s.difference(inds_match_s)
    inds_only_nbrs = np.array(list(inds_only_nbrs_s))
    xy_nbrs = xy_2[inds_only_nbrs,:]
    return xy_nbrs,inds_only_nbrs


def get_match_inds(xy_1,xy_2):
    inds_m = []
    nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(xy_2)
    ds,inds = nbrs.kneighbors(xy_1)
    inds_match = []
    for x in range(len(ds)):
        ds_x = ds[x].tolist()
        inds_x = inds[x].tolist()
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        if(len(ind_i)!=0):
            inds_m.append(inds_x[ind_i[0]])
    return inds_m


def neighborhood_n_layers_e(xy_1,xy_2,xy_e,n,t):
    layers = np.ones([xy_2.shape[0],])*(n+2)
    inds_n0 = get_match_inds(xy_1,xy_2)
    layers[inds_n0] = 0
    xy_layer = copy.deepcopy(xy_1)
    xy_cat = copy.deepcopy(xy_1)
    xy_cat = np.concatenate([xy_cat,xy_e],axis=0)
    for x in range(n):
        xy_layer,inds_n = neighbors_delaunay_t_exclude(xy_layer,xy_2,xy_cat,t)
        xy_cat = np.concatenate([xy_cat,xy_layer],axis=0)
        layers[inds_n] = x+1
    return layers


def loadDE_iters(alphas,it_inds,nGenes,r_src,r_tgt,tpF,minfc,lead_str,p_str,doT,tech_str,test_str):
    nAs = len(alphas)
    nIs = len(it_inds)
    tntest_res_up = np.zeros([nAs,nGenes])
    tntest_res_down = np.zeros([nAs,nGenes])
    log_fcs = np.zeros([nAs,nIs,nGenes])
    pvals = np.zeros([nAs,nIs,nGenes])
    err = 0
    errs = np.zeros([nAs,nIs])
    if(len(tech_str)==1):
        tech_str_0 = tech_str[0]
        tech_str_1 = tech_str[0]
    else:
        tech_str_0 = tech_str[0]
        tech_str_1 = tech_str[1]
    for x in range(nAs):
        alpha = alphas[x]
        for y in range(nIs):
            it_ind = it_inds[y]
            try:
                p_t = np.load('//broad/clearylab/Users/Peter/shalekOvary/trajMap/TN_test/{}_{}{}_{}_{}_{}_{:.1f}_{}_{}{}.npy'.format(lead_str,p_str,doT,tech_str_0,r_src,r_tgt,alpha,tpF,it_ind,test_str))
                labels_t,q_t = fdrcorrection(p_t,alpha=0.05,method='i',is_sorted=False)
                logfc = np.load('//broad/clearylab/Users/Peter/shalekOvary/trajMap/TN_test/{}_logfc{}_{}_{}_{}_{:.1f}_{}_{}{}.npy'.format(lead_str,doT,tech_str_1,r_src,r_tgt,alpha,tpF,it_ind,test_str))
                up_inds_t = np.where(np.logical_and(labels_t,np.logical_and(logfc>minfc,logfc<10)))[0]
                down_inds_t = np.where(np.logical_and(labels_t,np.logical_and((logfc)<-minfc,logfc>-10)))[0]
                tntest_res_up[x,up_inds_t] = tntest_res_up[x,up_inds_t]+1
                tntest_res_down[x,down_inds_t] = tntest_res_down[x,down_inds_t]+1
                pvals[x,y,:] = p_t
                log_fcs[x,y,:] = logfc
            except:
                err+=1
                errs[x,y] = 1
    return tntest_res_up,tntest_res_down,log_fcs,pvals,errs

def de_overlapping(test_res):
    nIs = np.max(test_res).astype(np.int32)
    nAs = test_res.shape[0]
    nGenes = test_res.shape[1]
    nSets = test_res.shape[2]
    nDEs_arr = np.zeros([nAs,nIs,nSets+1])
    inds_de_a = []
    for x in range(nAs):
        for y in range(nIs):
            test_t = test_res[x,:,:]>y
            test_a = np.expand_dims(np.prod(test_t,axis=1),-1)
            test_t = np.concatenate([test_t,test_a],axis=1)
            nDEs_arr[x,y,:] = np.sum(test_t,axis=0)
            inds_de_a.append(np.where(test_t[:,-1]))
    return nDEs_arr,inds_de_a

def de_overlapping_arr(test_res):
    nIs = np.max(test_res).astype(np.int32)
    nAs = test_res.shape[0]
    nGenes = test_res.shape[1]
    nSets = test_res.shape[2]
    inds_arr = np.zeros([nAs,nIs,nGenes,nSets+1])
    for x in range(nAs):
        for y in range(nIs):
            test_t = test_res[x,:,:]>y
            test_a = np.expand_dims(np.prod(test_t,axis=1),-1)
            test_t = np.concatenate([test_t,test_a],axis=1)
            inds_arr[x,y,:,:] = test_t
    return inds_arr

def diffExpr_fn(adata_src,adata_tgt,pi):
    adata_src = getInds(adata_src,adata_tgt,pi,False)
    X_src = adata_src.X
    if(isinstance(X_src,scipy.sparse._csr.csr_matrix)):
        X_src = X_src.toarray()
    X_src_1 = X_src[adata_src.obs['transport_0']==1.0,:]
    X_src_2 = X_src[adata_src.obs['transport_0']==2.0,:]
    X_src_both = np.concatenate([X_src_1,X_src_2])
    condition_numbers = np.concatenate([np.zeros([X_src_1.shape[0],1]),np.ones([X_src_2.shape[0],1])],axis=0)
    ad_src_both = ad.AnnData(X_src_both)
    ad_src_both.obs['condition'] = condition_numbers
    pickleFile = open('//broad/clearylab/Users/Peter/shalekOvary/adata_concat_traj_gcl.pkl','rb')
    adata_ts = pickle.load(pickleFile)
    gene_names_merfish = adata_ts.var_names.tolist()
    gene_names = adata_src.var_names.tolist()
    inds_g = []
    genes_err = []
    for x in range(len(gene_names_merfish)):
        try:
            inds_g.append(gene_names.index(gene_names_merfish[x]))
        except:
            genes_err.append(gene_names_merfish[x])
    ad_src_both = ad_src_both[:,inds_g]
    inds_nan = np.where(~np.isnan(np.sum(ad_src_both.X,axis=1)))[0]
    ad_src_both = ad_src_both[inds_nan,:]
    t = de.test.wald(
        data=ad_src_both,
        formula_loc="~ 1 + condition",
        factor_loc_totest="condition"
    )
    return t

def diffExpr_nbr_fn(adata_src_1,adata_src_2,adata_tgt,pi,ver='delaunay',k=None,r=None):
    xy_src_1 = adata_src_1.obsm['spatial']
    xy_src_2 = adata_src_2.obsm['spatial']
    X_src_1 = adata_counts(adata_src_1)
    X_src_2 = adata_counts(adata_src_2)
    if(ver=='radius'):
        _,inds = in_radius(xy_src_1,xy_src_2,r)
    elif(ver=='knn'):
        _,inds = knn_thresholded(xy_src_1,xy_src_2,k,r)
    elif(ver=='delaunay'):
        inds = neighbors_delaunay(xy_src_1,xy_src_2)
    elif(ver=='delaunay_ex'):
        inds = neighbors_delaunay_ex(xy_src_1,xy_src_2)
    else:
        raise ValueError('Could not recognize neighborhood version')
    inds_ne = np.where([len(x) for x in inds])[0]
    inds = [inds[x] for x in inds_ne]
    X_src_2 = X_src_2[inds_ne,:]
    X_nbrs = average_neighbors(X_src_1,X_src_2,inds)
    adata_src_1.X = X_nbrs
    t = diffExpr_fn(adata_src_1,adata_tgt,pi)
    return 