import sklearn.metrics.pairwise
import numpy as np
import torch
from .fgw_solver import log_ugw_sinkhorn_f
from ot.gromov import fused_unbalanced_gromov_wasserstein,fused_unbalanced_gromov_wasserstein2
from fugw.mappings import FUGW

class SOCSModel:
    """
    Software implementation of Spatial Optimal transport with Contiguous Structures (SOCS), 
    a tool for trajectory inference in spatial transcriptomics

    Parameters
    ----------
    adata: anndata.AnnData
        AnnData object containing all relevant spatial transcriptomics data
    time_key: str
        Key in adata.obs labeling cells by timepoint
    spatial_key: str
        Key in adata.obsm giving cells' spatial coordinates
    expr_key: str
        Key in adata.obsm giving cells' gene expression profiles (e.g. 'X_pca'). If None, default to adata.X
    struct_key:
        Key in adata.obs labeling cells by contiguously-developing structure

    """
    def __init__(self,adata,time_key='time',spatial_key='spatial',expr_key=None,struct_key=None,block_key=None,gr_key=None):

        self.adata = adata
        self.time_key = time_key
        self.spatial_key = spatial_key
        self.expr_key = expr_key
        self.struct_key = struct_key
        #self.gr_key = gr_key
        #self.block_key = block_key
        #self.method=method
        #self.method2 = method2
        #self.div = div
        #self.ot_config = {'block_factor':1,'eps':0.01,'dFactor':1,'alpha':0.5,'rho':100,'rho2':100,'tol':1e-7,'tol_ot':1e-7,
        #                 'nIters':30,'print_per_iter':None,'struct_excl':[]}
        #for k in kwargs.keys():
        #    self.ot_config[k] = kwargs[k]
    def infer_map(self,t0,t1,verbose=False):
        """
        Solves the SOCS optimal transport problem, estimating ancestor-descendant relationships between two timepoints.

        Parameters
        ----------
        alpha: float
            Value between 0 and 1 that trades off geometric consistency and gene-expression consistency between ancestors and descendants.
            Smaller values of alpha will lead to 
        epsilon: float
            Value controlling entropy regularization of the optimal transport problem. Higher values of epsilon will lead to more entropic
            maps, with individual cells in the source (t1) dataset mapping to more cells in the target (t2) dataset.
        rho1: float
            Value controlling "unbalanced-ness" of the mapping's rows - that is, how tightly the row-sums of the mapping agree with the prior.
            Higher values of rho1 will result in more consistent row-sums.
        rho2: float
            Value controlling "unbalanced-ness" of the mapping's columns - that is, how tightly the column-sums of the mapping agree with the prior.
            Higher values of rho2 will result in more consistent column-sums.
        fb0: float
            Spatial distance added between cells in the source distribution which do not belong to the same contiguous structure.
        fb1: float
            Spatial distance added between cells in the target distribution which do not belong to the same contiguous structure.
        f0: float
            Normalization factor to account for the differenc in magnitude between the expression distance matrix and t0 geometric distance matrix
        f1: float
            Normalization factor to account for difference in magnitude between the expression distance matrix and t1 geometric distance matrix
        t0:
            Source time-point label, as stored in self.adata.obs[self.time_key].
        t1:
            Target time-point label, as stored in self.adata.obs[self.time_key].
        
            
        Returns
        -------
        tmap: np.ndarray
            Map from ancestor cells in the source (t0) dataset to descendant cells in the target (t1) dataset.
        """
        D0,D1,D01 = self.compute_distance_matrices(t0,t1,verbose)
        if('fb0' in self.ot_config.keys()):
            fb_0 = self.ot_config['fb0']
        else:
            fb_0 = np.max(D0)
        if('fb1' in self.ot_config.keys()):
            fb_1 = self.ot_config['fb1']
        else:
            fb_1 = np.max(D1)
        S0,S1 = self.compute_struct_matrices(t0,t1,fb_0,fb_1,verbose)
        if('f0' in self.ot_config.keys()):
            f0 = self.ot_config['f0']
        else:
            f0 = (np.max(D0)/np.max(D01))**2
        if('f1' in self.ot_config.keys()):
            f1 = self.ot_config['f1']
        else:
            f1 = (np.max(D1)/np.max(D01))**2
        tmap = self.compute_transport_map(D0+S0,D1+S1,D01,f0,f1,t0,t1,verbose)
        return tmap
    def compute_distance_matrices(self,t0,t1,verbose=False):
        if(verbose):
            print('Computing Distance Matrices')
        adata_0 = self.adata[self.adata.obs[self.t_col]==t0,:]
        adata_1 = self.adata[self.adata.obs[self.t_col]==t1,:]
        if(self.expr_key==None):
            expr_0 = adata_0.X
            expr_1 = adata_1.X
        else:
            expr_0 = adata_0.obsm[self.expr_key]
            expr_1 = adata_1.obsm[self.expr_key]
        xy_0 = adata_0.obsm[self.spatial_key]
        xy_1 = adata_1.obsm[self.spatial_key]
        D01 = np.ascontiguousarray(sklearn.metrics.pairwise.pairwise_distances(expr_0,Y=expr_1,metric='euclidean',n_jobs=1))
        D0 = np.ascontiguousarray(sklearn.metrics.pairwise.pairwise_distances(xy_0,Y=xy_0,metric='euclidean',n_jobs=1))
        D1 = np.ascontiguousarray(sklearn.metrics.pairwise.pairwise_distances(xy_1,Y=xy_1,metric='euclidean',n_jobs=1))
        return D0,D1,D01
    def compute_struct_matrices(self,t0,t1,fb_0,fb_1,verbose=False):
        if(verbose):
            print('Computing Structural Contiguity Distance Matrices')
        adata_0 = self.adata[self.adata.obs[self.t_col]==t0,:]
        adata_1 = self.adata[self.adata.obs[self.t_col]==t1,:]
        S_0 = np.zeros([adata_0.shape[0],adata_0.shape[0]])
        S_1 = np.zeros([adata_1.shape[0],adata_1.shape[0]])
        beta_0 = adata_0.obs[self.struct_key]
        beta_1 = adata_1.obs[self.struct_key]
        beta_vals_0 = np.unique(beta_0)
        beta_vals_1 = np.unique(beta_1)
        beta_vals_0 = [x for x in beta_vals_0 if x not in self.ot_config['struct_excl']]
        beta_vals_1 = [x for x in beta_vals_1 if x not in self.ot_config['struct_excl']]
        for x in range(len(beta_vals_0)):
            inds_s = np.where(beta_0==beta_vals_0[x])[0]
            inds_ns = np.where(beta_0!=beta_vals_0[x])[0]
            for y in range(len(inds_s)):
                S_0[inds_s[y],inds_ns] = fb_0
        for x in range(len(beta_vals_1)):
            inds_s = np.where(beta_1==beta_vals_1[x])[0]
            inds_ns = np.where(beta_1!=beta_vals_1[x])[0]
            for y in range(len(inds_s)):
                S_1[inds_s[y],inds_ns] = fb_1
        return S_0,S_1
    def compute_transport_map(self,D0,D1,D01,f0,f1,t0,t1,method,verbose=False):
        if(verbose):
            print('Computing Transport Map')
        p0 = np.ones([D0.shape[0],])/D0.shape[0]
        p1 = np.ones([D1.shape[0],])/D1.shape[0]
        if(self.gr_key!=None):
            gr_vals_0 = self.adata[self.adata.obs['time']==t0,:].obs[self.gr_key].to_numpy()
            #gr_vals_1 = self.adata[self.adata.obs['time']==t1,:].obs[self.gr_key].to_numpy()
            p0 = np.multiply(p0,gr_vals_0)
            #p1 = np.multiply(p1,gr_vals_1)
        p0 = torch.tensor(p0,dtype=torch.float64)
        p1 = torch.tensor(p1,dtype=torch.float64)
        D01 = torch.tensor(D01,dtype=torch.float64)
        D0 = torch.tensor(D0,dtype=torch.float64)
        D1 = torch.tensor(D1,dtype=torch.float64)
        if(self.method=='fugw_cpu'):
            pi_01, gamma = log_ugw_sinkhorn_f(p0, D0/f0, p1, D1/f1, D01, self.ot_config['alpha'], init=None, eps=self.ot_config['eps'],
                                    rho=self.ot_config['rho'], rho2=self.ot_config['rho2'],
                                    nits_plan=self.ot_config['nIters'], tol_plan=1e-30,
                                    nits_sinkhorn=10, tol_sinkhorn=1e-9,
                                    two_outputs=False,print_per_iter=None,alt=0)
        elif(self.method=='pot'):
            dFactor = self.ot_config['dFactor']
            pi_01,_= fused_unbalanced_gromov_wasserstein(D0/(f0*dFactor),D1/(f1*dFactor),wx=p0,wy=p1,reg_marginals=[self.ot_config['rho'],self.ot_config['rho2']],
                            epsilon=self.ot_config['eps'],alpha=self.ot_config['alpha'],M=D01/dFactor,max_iter=self.ot_config['nIters'],unbalanced_solver=self.method2,divergence=self.div,tol=self.ot_config['tol'],tol_ot=self.ot_config['tol_ot'],verbose=True)
        elif(self.method=='fugw_gpu'):
            adata_0 = self.adata[self.adata.obs[self.t_col]==t0,:]
            adata_1 = self.adata[self.adata.obs[self.t_col]==t1,:]
            if(self.expr_key==None):
                expr_0 = adata_0.X
                expr_1 = adata_1.X
            else:
                expr_0 = adata_0.obsm[self.expr_key]
                expr_1 = adata_1.obsm[self.expr_key]
            mapping = FUGW(alpha=self.ot_config['alpha'],eps=self.ot_config['eps'],rho=self.ot_config['rho'])
            _ = mapping.fit(expr_0.T,expr_1.T,source_geometry=D0,target_geometry=D1)
            pi_01 = mapping.pi
        return pi_01.numpy()
        

        
        