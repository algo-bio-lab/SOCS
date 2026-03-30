import sklearn.metrics.pairwise
import numpy as np
import torch
from .fgw_solver import log_ugw_sinkhorn_f
from ot.gromov import fused_unbalanced_gromov_wasserstein,fused_unbalanced_gromov_wasserstein2
from fugw.mappings import FUGW

class SOCSModel:
    def __init__(self,adata,t_col,spatial_key='spatial',expr_key=None,struct_key=None,block_key=None,gr_key=None,method='fugw_cpu',method2='mm',div='kl',**kwargs):
        self.adata = adata
        self.t_col = t_col
        self.spatial_key = spatial_key
        self.expr_key = expr_key
        self.struct_key = struct_key
        self.gr_key = gr_key
        self.block_key = block_key
        self.method=method
        self.method2 = method2
        self.div = div
        self.ot_config = {'block_factor':1,'eps':0.01,'dFactor':1,'alpha':0.5,'rho':100,'rho2':100,'tol':1e-7,'tol_ot':1e-7,
                         'nIters':30,'print_per_iter':None,'struct_excl':[]}
        for k in kwargs.keys():
            self.ot_config[k] = kwargs[k]
    def infer_map(self,t0,t1,verbose=False):
        D0,D1,D01 = self.compute_distance_matrices(t0,t1,verbose)
        if(self.block_key is not None):
            bVal = np.max(D01)
            B01 = self.compute_block_matrix(t0,t1,bVal,verbose)
            D01 = D01+B01
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
    def compute_block_matrix(self,t0,t1,bVal,verbose=False):
        if(verbose):
            print('Computing block matrices')
        adata_0 = self.adata[self.adata.obs[self.t_col]==t0,:]
        adata_1 = self.adata[self.adata.obs[self.t_col]==t1,:]
        B01 = np.zeros([adata_0.shape[0],adata_1.shape[0]])
        for y in range(len(self.block_key)):
            b_0 = adata_0.obs[self.block_key[y][0]]
            b_1 = adata_1.obs[self.block_key[y][1]]
            b_vals = np.unique(b_0)
            for x in range(len(b_vals)):
                if(x!=0):
                    inds_x0 = np.where(b_0==b_vals[x])[0]
                    inds_x1 = np.where(b_1==b_vals[x])[0]
                    for z in inds_x0:
                        B01[z,inds_x1] = bVal*self.ot_config['block_factor']
        return B01
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
        

        
        