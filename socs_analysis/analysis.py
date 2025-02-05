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
import shapely
import copy

def follicle_average(adata,geneName):
    follicle_ids = np.unique(adata.obs['follicle'])
    avg_mkr = np.zeros([len(follicle_ids),])
    for x in range(len(follicle_ids)):
        avg_mkr[x] = np.mean(adata[adata.obs['follicle']==follicle_ids[x],:].X[:,adata.var_names==geneName],axis=0)
    follicle_avg_mkr = [avg_mkr[list(follicle_ids).index(x)] for x in adata.obs['follicle'].tolist()]
    adata.obs['follicle_'+geneName] = follicle_avg_mkr
    return adata

def getInds_g(adata_src,adata_tgt,pi,clst):
    adata_src_r = adata_src.copy()
    adata_tgt_r = adata_tgt.copy()
    pi_cx = np.zeros(pi.shape)
    for x in range(pi.shape[0]):
        pi_cx[x,:] = pi[x,:]/np.sum(pi[x,:])
    clsts_bt_src = np.array(adata_src_r.obs[clst].tolist())
    clsts_bt_tgt = np.array(adata_tgt_r.obs[clst].tolist())
    nTypes_src = len(np.unique(clsts_bt_src))
    nTypes_tgt = len(np.unique(clsts_bt_tgt))
    types_src_u = np.unique(clsts_bt_src)
    types_tgt_u = np.unique(clsts_bt_tgt)
    inds_tgt_types = []
    for x in range(nTypes_tgt):
        inds_tgt_types.append(np.where(clsts_bt_tgt==types_tgt_u[x])[0].tolist())
    inds_src_types = []
    for x in range(nTypes_src):
        inds_src_types.append(np.where(clsts_bt_src==types_src_u[x])[0].tolist())
    pi_types = np.zeros([pi_cx.shape[0],nTypes_tgt])
    for x in range(nTypes_tgt):
        pi_types[:,x] = np.sum(pi_cx[:,inds_tgt_types[x]],axis=1)
    inds_samples = [[[] for y in range(nTypes_tgt)] for x in range(nTypes_src)]
    errs = 0
    for y in range(nTypes_tgt):
        for x in range(len(inds_src_types[y])):
            rx = np.random.rand()
            bins_x = [np.sum(pi_types[inds_src_types[y][x],0:z+1]) for z in range(pi_types.shape[1])]
            rx_bin = np.digitize(rx,bins_x)
            try:
                inds_samples[y][rx_bin].append(inds_src_types[y][x])
            except:
                errs+=1
    for x in range(nTypes_src):
        transport_vals = np.zeros([adata_src_r.shape[0],])
        for y in range(nTypes_tgt):
            transport_vals[inds_samples[x][y]] = y+1
        adata_src_r.obs['transport_{}'.format(x)] = transport_vals
    return adata_src_r
def getInds(adata_src,adata_tgt,pi,fBool):
    ### This function is used to identify which cells, in each cell type are transported to each cell type.
    pi_cx = np.zeros(pi.shape)
    for x in range(pi.shape[0]):
        pi_cx[x,:] = pi[x,:]/np.sum(pi[x,:])
    clsts_bt_src = np.array(adata_src.obs['clsts_bt2'].tolist())
    clsts_bt_tgt = np.array(adata_tgt.obs['clsts_bt2'].tolist())
    inds_tgt_types = [[],[],[]]
    if(fBool):
        for x in range(3):
            follicles_tgt = np.array(adata_tgt.obs['follicle'].tolist())
            follicles_u_tgt = np.unique(follicles_tgt)
            follicles_u_tgt = np.array(follicles_u_tgt,dtype=np.int32)
            nFs_2 = len(follicles_u_tgt)
            follicles_bt_u_tgt = np.zeros([nFs_2,],dtype=np.uint8)
            for y in range(nFs_2):
                fInd = y
                inds_f = np.where(follicles_tgt==follicles_u_tgt[fInd])[0]
                nTs = np.array([len(np.where(clsts_bt_tgt[inds_f]==z)[0]) for z in range(3)])
                follicles_bt_u_tgt[y] = np.argmax(nTs)
            inds_types = np.where(follicles_bt_u_tgt==x)[0]
            follicles_types = follicles_u_tgt[inds_types]
            inds_tgt_types[x] = np.where(pd.Series(follicles_tgt).isin(follicles_types))[0]
    else:
        for x in range(3):
            inds_tgt_types[x] = np.where(clsts_bt_tgt==x)[0]
    inds_src_types = [[],[],[]]
    for x in range(3):
        inds_src_types[x] = np.where(clsts_bt_src==x)[0]
    pi_types = np.zeros([pi_cx.shape[0],3])
    for x in range(3):
        pi_types[:,x] = np.sum(pi_cx[:,inds_tgt_types[x]],axis=1)
    inds_samples = [[[],[],[]],[[],[],[]],[[],[],[]]]
    for y in range(3):
        for x in range(len(inds_src_types[y])):
            rx = np.random.rand()
            if(rx<pi_types[inds_src_types[y][x],0]):
                inds_samples[y][0].append(inds_src_types[y][x])
            elif(rx<np.sum(pi_types[inds_src_types[y][x],0:2])):
                inds_samples[y][1].append(inds_src_types[y][x])
            else:
                inds_samples[y][2].append(inds_src_types[y][x])
    for x in range(3):
        transport_vals = np.zeros([adata_src.shape[0],])
        transport_vals[inds_samples[x][0]] = 1
        transport_vals[inds_samples[x][1]] = 2
        transport_vals[inds_samples[x][2]] = 3
        adata_src.obs['transport_{}'.format(x)] = transport_vals
        type_tp_map = {0:'other cells',1:'type 1',2:'type 2',3:'type 3'}
        adata_src.obs['transport_s_{}'.format(x)] = adata_src.obs['transport_{}'.format(x)].map(type_tp_map)
    return adata_src

def getInds_t(adata_src,adata_tgt,pi,fBool,t):
    ### This function is used to identify which cells, in each cell type are transported to each cell type.
    pi_cx = np.zeros(pi.shape)
    for x in range(pi.shape[0]):
        pi_cx[x,:] = pi[x,:]/np.sum(pi[x,:])
    clsts_bt_src = np.array(adata_src.obs['clsts_bt2'].tolist())
    clsts_bt_tgt = np.array(adata_tgt.obs['clsts_bt2'].tolist())
    inds_tgt_types = [[],[],[]]
    if(fBool):
        for x in range(3):
            follicles_tgt = np.array(adata_tgt.obs['follicle'].tolist())
            follicles_u_tgt = np.unique(follicles_tgt)
            follicles_u_tgt = np.array(follicles_u_tgt,dtype=np.int32)
            nFs_2 = len(follicles_u_tgt)
            follicles_bt_u_tgt = np.zeros([nFs_2,],dtype=np.uint8)
            for y in range(nFs_2):
                fInd = y
                inds_f = np.where(follicles_tgt==follicles_u_tgt[fInd])[0]
                nTs = np.array([len(np.where(clsts_bt_tgt[inds_f]==z)[0]) for z in range(3)])
                follicles_bt_u_tgt[y] = np.argmax(nTs)
            inds_types = np.where(follicles_bt_u_tgt==x)[0]
            follicles_types = follicles_u_tgt[inds_types]
            inds_tgt_types[x] = np.where(pd.Series(follicles_tgt).isin(follicles_types))[0]
    else:
        for x in range(3):
            inds_tgt_types[x] = np.where(clsts_bt_tgt==x)[0]
    inds_src_types = [[],[],[]]
    for x in range(3):
        inds_src_types[x] = np.where(clsts_bt_src==x)[0]
    pi_types = np.zeros([pi_cx.shape[0],3])
    for x in range(3):
        pi_types[:,x] = np.sum(pi_cx[:,inds_tgt_types[x]],axis=1)
    inds_samples = [[[],[],[]],[[],[],[]],[[],[],[]]]
    for y in range(3):
        for x in range(len(inds_src_types[y])):
            if(pi_types[inds_src_types[y][x],0]>t):
                inds_samples[y][0].append(inds_src_types[y][x])
            elif(pi_types[inds_src_types[y][x],1]>t):
                inds_samples[y][1].append(inds_src_types[y][x])
            elif(pi_types[inds_src_types[y][x],2]>t):
                inds_samples[y][2].append(inds_src_types[y][x])
    for x in range(3):
        transport_vals = np.zeros([adata_src.shape[0],])
        transport_vals[inds_samples[x][0]] = 1
        transport_vals[inds_samples[x][1]] = 2
        transport_vals[inds_samples[x][2]] = 3
        adata_src.obs['transport_{}'.format(x)] = transport_vals
        type_tp_map = {0:'other cells',1:'type 1',2:'type 2',3:'type 3'}
        adata_src.obs['transport_s_{}'.format(x)] = adata_src.obs['transport_{}'.format(x)].map(type_tp_map)
    return adata_src

def add_follicle_df(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        follicle_labels = np.array(adata_r.obs['follicle'].tolist())
        follicles_u = np.unique(follicle_labels)
        follicle_df = pd.DataFrame(index=follicles_u)
        adata_r.uns['follicles'] = follicle_df
    return adata_r

def follicle_average(adata,geneName):
    adata_r = adata.copy()
    if 'follicles' not in adata.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_ids = np.unique(adata_r.obs['follicle'])
    avg_mkr = np.zeros([len(follicle_ids),])
    for x in range(len(follicle_ids)):
        avg_mkr[x] = np.mean(adata_r[adata_R.obs['follicle']==follicle_ids[x],:].X[:,adata_r.var_names==geneName],axis=0)
    adata_r.uns['follicles']['avg_'+geneName] = avg_mkr
    follicle_avg_mkr = [avg_mkr[list(follicle_ids).index(x)] for x in adata_r.obs['follicle'].tolist()]
    adata_r.obs['follicle_'+geneName] = follicle_avg_mkr
    return adata_r

def follicle_average_obs(adata,obs_name):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata)
    #follicle_ids = np.unique(adata.obs['follicle'])
    follicle_ids = adata_r.uns['follicles'].index.tolist()
    avg_mkr = np.zeros([len(follicle_ids),])
    for x in range(len(follicle_ids)):
        inds_x = np.where(adata.obs['follicle']==follicle_ids[x])[0]
        avg_mkr[x] = np.mean(adata[inds_x,:].obs[obs_name],axis=0)
        #avg_mkr[x] = np.mean(adata[adata.obs['follicle']==follicle_ids[x],:].obs[obs_name],axis=0)
    adata_r.uns['follicles']['avg_'+obs_name] = avg_mkr
    return adata_r

def follicleSize(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    follicle_size = []
    for x in range(nFs):
        follicle_size.append(len(np.where(follicle_labels==follicles_u[x])[0]))
    adata_r.uns['follicles']['size'] = follicle_size
    return adata_r

def follicleSize_h(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle_h'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    follicle_size = []
    for x in range(nFs):
        follicle_size.append(len(np.where(follicle_labels==follicles_u[x])[0]))
    adata_r.uns['follicles']['size_h'] = follicle_size
    return adata_r

def follicle_areas(adata,regions):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    follicles_area = []
    all_areas = []
    for x in range(len(regions)):
        all_areas = all_areas+list(np.load('//broad/clearylab/Users/Peter/shalekOvary/follicle_areas_{}.npy'.format(regions[x])))
    for x in range(nFs):
        follicles_area.append(all_areas[np.array(follicles_u[x]-1,dtype=np.int32)])
    adata_r.uns['follicles']['area'] = follicles_area
    return adata_r

def follicle_areas_w_holes(adata,regions):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    follicles_area = []
    all_areas = []
    for x in range(len(regions)):
        all_areas = all_areas+list(np.load('//broad/clearylab/Users/Peter/shalekOvary/follicle_holes_areas_{}.npy'.format(regions[x])))
    for x in range(nFs):
        follicles_area.append(all_areas[np.array(follicles_u[x]-1,dtype=np.int32)])
    adata_r.uns['follicles']['area_w_holes'] = follicles_area
    return adata_r

def follicle_densities(adata,region):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = follicle_df(adata_r)
    if 'size' not in adata_r.uns['follicles']:
        adata_r = follicleSize(adata_r)
    if 'area'  not in adata_r.uns['follicles']:
        adata_r = follicle_areas(adata_r,region)
    adata_r.uns['follicles']['density'] = adata_r.uns['follicles']['size']/adata_r.uns['follicles']['area']
    return adata_r

def follicle_densities_h(adata,region):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = follicle_df(adata_r)
    if 'size' not in adata_r.uns['follicles']:
        adata_r = follicleSize_h(adata_r)
    if 'area'  not in adata_r.uns['follicles']:
        adata_r = follicle_areas_w_holes(adata_r,region)
    adata_r.uns['follicles']['density_h'] = adata_r.uns['follicles']['size_h']/adata_r.uns['follicles']['area_w_holes']
    return adata_r



def follicle_diameters(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    f_diams = np.zeros([nFs,3])
    xy = adata_r.obsm['spatial']
    for x in range(nFs):
        inds_x = np.where(follicle_labels==follicles_u[x])[0]
        if(len(inds_x)!=0):
            xy_x = xy[inds_x,:]
            f_diams[x,0] = np.max(xy_x[:,0])-np.min(xy_x[:,0])
            f_diams[x,1] = np.max(xy_x[:,1])-np.min(xy_x[:,1])
            f_diams[x,2] = (f_diams[x,0]+f_diams[x,1])/2
    adata_r.uns['follicles']['diameter'] = f_diams[:,2]
    return adata_r

def follicle_diameter_sweep(xy):
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

def follicle_diameters_angles(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    f_diams = np.zeros([nFs,])
    xy = adata_r.obsm['spatial']
    for x in range(nFs):
        inds_x = np.where(follicle_labels==follicles_u[x])[0]
        if(len(inds_x)>2):
            xy_x = xy[inds_x,:]
            f_diams[x] = follicle_diameter_sweep(xy_x)
    adata_r.uns['follicles']['diameter_angle'] = f_diams
    return adata_r

def get_ctr(xy):
    c_x = np.sum(xy[:,0])/xy.shape[0]
    c_y = np.sum(xy[:,1])/xy.shape[0]
    return [c_x,c_y]

def follicle_centroids(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    f_ctrs = np.zeros([nFs,2])
    xy = adata_r.obsm['spatial']
    for x in range(nFs):
        inds_x = np.where(follicle_labels==follicles_u[x])[0]
        if(len(inds_x)!=0):
            xy_x = xy[inds_x,:]
            ctr_x = get_ctr(xy_x)
            f_ctrs[x,:] = ctr_x
    adata_r.uns['follicles']['centroid_x'] = f_ctrs[:,0]
    adata_r.uns['follicles']['centroid_y'] = f_ctrs[:,1]
    return adata_r

def follicle_radial_dist(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    if 'centroid' not in adata_r.uns['follicles']:
        adata_r = follicle_centroids(adata_r)
    xy_all = get_ctr(adata_r.obsm['spatial'])
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    f_rads = np.zeros([nFs,])
    for x in range(nFs):
        inds_x = np.where(follicle_labels==follicles_u[x])[0]
        if(len(inds_x)!=0):
            ctr_xx = adata.uns['follicles']['centroid_x'][follicles_u[x]]
            ctr_xy = adata.uns['follicles']['centroid_y'][follicles_u[x]]
            ctr_x = [ctr_xx,ctr_xy]
            f_rads[x] = np.linalg.norm(np.array(ctr_x)-np.array(xy_all))
    adata_r.uns['follicles']['rad_dist'] = f_rads
    return adata_r

def cell_density_r(adata,rds):
    adata_r = adata.copy()
    xy = adata_r.obsm['spatial']
    _,inds_r = in_radius(xy,xy,rds)
    nNeighbors = [len(inds_r[x]) for x in range(len(inds_r))]
    adata_r.obs['n_neighbors'] = nNeighbors
    return adata_r
    
def cell_edge_dist(adata,regions):
    adata_r = adata.copy()
    edge_dists = np.zeros([adata_r.shape[0],len(regions)])
    pts = [shapely.geometry.Point(adata_r.obsm['spatial'][g,0],adata_r.obsm['spatial'][g,1]) for g in range(adata_r.shape[0])]
    for x in range(len(regions)):
        bounds = pd.read_csv('//broad/clearylab/Users/Peter/shalekOvary/follicleBoundaries/ovary_boundaries_{}.csv'.format(regions[x]))
        ovary_boundary = shapely.geometry.LineString(bounds)
        edge_dists[:,x] = shapely.distance(ovary_boundary,pts)
    adata_r.obs['edge_dist'] = np.min(edge_dists,axis=1)
    return adata_r

def follicle_edge_dist(adata,regions):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    if 'centroid' not in adata_r.uns['follicles']:
        adata_r = follicle_centroids(adata_r)
    edge_dists = np.zeros([adata_r.uns['follicles'].shape[0],len(regions)])
    pts = [shapely.geometry.Point(adata_r.uns['follicles']['centroid_x'].iloc[x],adata_r.uns['follicles']['centroid_y'].iloc[x]) for x in range(adata_r.uns['follicles'].shape[0])]
    for x in range(len(regions)):
        bounds = pd.read_csv('//broad/clearylab/Users/Peter/shalekOvary/follicleBoundaries/ovary_boundaries_{}.csv'.format(regions[x]))
        ovary_boundary = shapely.geometry.LineString(bounds)
        edge_dists[:,x] = shapely.distance(ovary_boundary,pts)
    adata_r.uns['follicles']['edge_dist'] = np.min(edge_dists,axis=1)
    return adata_r

def follicles_to_cells(adata):
    adata_r = adata.copy()
    follicle_columns = adata_r.uns['follicles'].columns.tolist()
    for y in follicle_columns:
        cell_follicle_data = np.zeros([adata.shape[0],])
        for x in range(adata.shape[0]):
            cell_follicle_data[x] = adata.uns['follicles'].loc[adata.obs['follicle'][x]][y]
        adata_r.obs['follicle_'+y] = cell_follicle_data
    return adata_r



def map_follicles(adata_src,adata_tgt,pi):
    adata_src_r = adata_src.copy()
    adata_tgt_r = adata_tgt.copy()
    if 'follicles' not in adata_src_r.uns:
        adata_src_r = add_follicle_df(adata_src_r)
    if 'follicles' not in adata_tgt_r.uns:
        adata_tgt_r = add_follicle_df(adata_tgt_r)
    nFs_src = adata_src_r.uns['follicles'].shape[0]
    nFs_tgt = adata_tgt_r.uns['follicles'].shape[0]
    f2f = np.zeros([nFs_src,nFs_tgt])
    for z in range(nFs_src):
        inds_x = np.where(adata_src_r.obs['follicle'].values==adata_src_r.uns['follicles'].index.values[z])[0]
        pi_x = pi[inds_x,:]
        for w in range(nFs_tgt):
            inds_y = np.where(adata_tgt_r.obs['follicle'].values==adata_tgt_r.uns['follicles'].index.values[w])[0]
            f2f[z,w] = np.sum(pi_x[:,inds_y])
        f2f[z,:] = f2f[z,:]/np.sum(f2f[z,:])
    adata_src_r.uns['follicle map'] = f2f
    for x in adata_tgt_r.uns['follicles'].keys():
        adata_src_r.uns['follicles'][x+'_t'] = np.matmul(adata_tgt_r.uns['follicles'][x].values,f2f.T)
        adata_src_r.uns['follicles'][x+'_d'] = adata_src_r.uns['follicles'][x+'_t']-adata_src_r.uns['follicles'][x]
    return adata_src_r
    
def follicle_types(adata):
    adata_r = adata.copy()
    if 'follicles' not in adata_r.uns:
        adata_r = add_follicle_df(adata_r)
    clsts_bt = np.array(adata_r.obs['clsts_bt2'].tolist())
    follicle_labels = np.array(adata_r.obs['follicle'].tolist())
    follicles_u = adata_r.uns['follicles'].index.tolist()
    nFs = len(follicles_u)
    follicle_types = []
    for x in range(nFs):
        fInd = x
        inds_f = np.where(follicle_labels==follicles_u[fInd])[0]
        if(len(inds_f)!=0):
            nTs = np.array([len(np.where(clsts_bt[inds_f]==z)[0]) for z in range(3)])
            follicle_types.append(np.argmax(nTs))
    adata_r.uns['follicles']['type'] = follicle_types
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

def knn_thresholded(xy_1,xy_2,k,t):
    nbrs = NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(xy_2)
    ds,inds = nbrs.kneighbors(xy_1)
    inds_t = []
    ds_t = []
    for x in range(len(ds)):
        ds_x = ds[x][ds[x]<t].tolist()
        ind_x = inds[x][ds[x]<t].tolist()
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        if(len(ind_i)!=0):
            #ind_i = ds_x.index(0)
            del ind_x[ind_i[0]]
            del ds_x[ind_i[0]]
        inds_t.append(ind_x)
        ds_t.append(ds_x)
    return ds_t,inds_t

def neighbors_delaunay(xy_1,xy_2):
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
        inds_xu.remove(ind_i)
        inds_d.append(inds_xu)
    return inds_d

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

def neighbors_knn_t_exclude(xy_1,xy_2,xy_e,k,t):
    nbrs = NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(xy_2)
    ds,inds = nbrs.kneighbors(xy_1)
    inds_t = []
    ds_t = []
    for x in range(len(ds)):
        ds_x = ds[x][ds[x]<t].tolist()
        ind_x = inds[x][ds[x]<t].tolist()
        ind_i = np.where(np.isclose(ds_x,0,atol=1e-3))[0]
        if(len(ind_i)!=0):
            #ind_i = ds_x.index(0)
            del ind_x[ind_i[0]]
            del ds_x[ind_i[0]]
        inds_t.append(ind_x)
        ds_t.append(ds_x)
    all_inds = []
    for x in inds_t:
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

def neighbors_delaunay_exclude(xy_1,xy_2,xy_e):
    inds_nbrs = neighbors_delaunay(xy_1,xy_2)
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

def neighbors_delaunay_ex(xy_1,xy_2):
    inds_d = []
    xy_2e = np.concatenate([xy_1,xy_2],axis=0)
    tri = Delaunay(xy_2e)
    for x in range(xy_1.shape[0]):
        tri_x = tri.simplices==x
        inds_x = np.where(np.sum(tri_x,axis=1))[0]
        tri_xi = tri.simplices[inds_x,:]
        inds_xu = np.unique(tri_xi.flatten()).tolist()
        inds_2 = np.where([inds_xu[y]>=xy_1.shape[0] for y in range(len(inds_xu))])[0]
        inds_xu2 = [inds_xu[y]-xy_1.shape[0] for y in inds_2]
        inds_d.append(inds_xu2)
    return inds_d

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

def neighborhood_n_layers(xy_1,xy_2,n,t):
    layers = np.ones([xy_2.shape[0],])*(n+2)
    inds_n0 = get_match_inds(xy_1,xy_2)
    layers[inds_n0] = 0
    xy_layer = copy.deepcopy(xy_1)
    xy_cat = copy.deepcopy(xy_1)
    for x in range(n):
        xy_layer,inds_n = neighbors_delaunay_t_exclude(xy_layer,xy_2,xy_cat,t)
        xy_cat = np.concatenate([xy_cat,xy_layer],axis=0)
        layers[inds_n] = x+1
    return layers

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



def average_neighbors(X_1,X_2,inds):
    X_n = copy.copy(X_1)
    for x in range(X_1.shape[0]):
        X_n[x,:] = np.nanmean(X_2[inds[x],:],axis=0)
    return X_n

def adata_counts(adata):
    X = adata.X
    if(isinstance(X,scipy.sparse._csr.csr_matrix)):
        X = X.toarray()
    return X

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
    X_src = adata_counts(adata_src)
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