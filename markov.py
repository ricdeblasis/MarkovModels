import numpy as np
import random
import itertools
import pandas as pd
from arch import arch_model
from statsmodels.tsa import stattools
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from MarkovModels.utility import recursive
import warnings
from itertools import chain
from bisect import bisect_left
from multiprocessing import Pool
from functools import partial

class SemiMarkov():
    """
    Semi-Markov model class.
    
    Allows the Weighted Indexed semi-Markov model.
    
    Parameters
    ----------
    process_discretize, {'Manual', 'KMeans', 'Sigma'}, str, default='Kmeans'
        Discretization method for the process. It requires the definition of the numbers 
        of states or the bin edges in the fit() method using the process_bins.
        
        'Manual': it requires manually defined bin edges. A list of edges has to
        be passed to the fit() method using the process_bins option.
        
        'KMeans': it uses the SciKit-Learn KMeans method to discretize the 
        process. It requires the definition of the numbers of bins in
        the fit() method using the process_bins option.
        
        'Sigma': starts from the center of the distribution and creates a number
        of bins defined in process_bins with a standard deviation width.
        It requires the definition of the numbers of bins in the fit() method
        using the process_bins option.
    
    index_model, {'EWMA', 'GARCH'}, str, default=None
        If None, the model reduces to a semi-Markov model without index.
        
        'EWMA': the index is computed using the Exponentially
        Weighted Moving Average.
        
        'GARCH': the index is computed using the Generalized
        Autoregressive Conditional Heteroskedasticity model.
    
    index_discretize, {'Manual', 'KMeans', 'Uniform'}, str, default='KMeans'
        Discretization method for the index.
        
        'Manual': it requires manually defined bin edges. A list of edges has to
        be passed to the fit() method using the index_bins option.
        
        'KMeans': it uses the SciKit-Learn KMeans method to discretize the 
        index. It requires the definition of the numbers of bins in
        the fit() method using the index_bins option.
        
        'Uniform': creates n equally spaced bins to discretize the index.
        It requires the definition of the numbers of bins in the fit() method
        using the index_bins option.
        
    kmeans_seed, int, default=None
        Set to int to make the randomness deterministic. Used for the KMeans 
        algorithm.
        
        
    Attributes
    ----------
    p : pandas DataFrame (states*states*index_nbins)
        Transition probabilities
   
    Notes
    -----
    (J,T): Markov renewal process with state space E
    J: random variable contatining jumps
    T: random variable containaing times
    Y: semi-Markov process
    V: Index
    p: transition probabilities
    f: conditional sojourn time distributions: P(T_{n+1}-T_n=t|J_n=i, J_{n+1}=j)
    b: P(J_{n+1}=j, T_{n+1}-T_n=t|J_n=i)
    G: waiting time distribution function in state i given that the process will be in state j
    Q: semi-Markov kernel
    H: Survival function in state i
    
    Examples
    --------
    >>> from from markov import SemiMarkov
    >>> 
    >>> sm = SemiMarkov(index_model='EWMA', index_discretize='Manual')
    >>> sm.fit(Y, smooth=.97, index_bins=[0,0.2,0.5,0.8,1.4,3.1])
    """

    def __init__(
        self,
        process_discretize='Kmeans',
        index_model=None,
        index_discretize='KMeans',
        kmeans_seed=None
        ):
        self.name = 'Weighted-Indexed Semi-Markov model'
        if index_model is None:
            self.name = 'Semi-Markov model'
        self.process_discretize = process_discretize
        self.index_model = index_model
        self.index_discretize = index_discretize
        self.kmeans_seed = kmeans_seed
        
        self.discretized = False
        self.simulation = None
        self.params = dict() # dictionary with extra parameters
        
    def _model_description(self):
        """Generates the model description for use by __str__ and related
        functions"""
        descr = {'Process discretization':str(self.process_discretize or 'No'),
                 'Index model':str(self.index_model or 'No'),
                 'Index discretization':self.index_discretize,
                 'Random seed':str(self.kmeans_seed or 'No')}
        return descr
        
    def __str__(self) -> str:
        descr = self._model_description()
        descr_str = self.name + "("
        for key, val in descr.items():
            if val and key:
                descr_str += key + ": " + val + ", "
        descr_str = descr_str[:-2]  # Strip final ', '
        descr_str += ")"

        return descr_str

    def __repr__(self) -> str:
        txt = self.__str__()
        txt.replace("\n", "")
        return txt + ", id: " + hex(id(self))
    
    def _repr_html_(self) -> str:
        """HTML representation for IPython Notebook"""
        descr = self._model_description()
        html = "<strong>" + self.name + "</strong> ("
        for key, val in descr.items():
            html += "<strong>" + key + ": </strong>" + val + ",\n"
        html += "<strong>ID: </strong> " + hex(id(self)) + ")"
        return html
    
    @staticmethod
    def _get_edges(y, s, n=1):
        edges = [y.std()*n*(-s/2+x) for x in range(s+1)][1:-1]
        edges.append(y.max())
        edges.insert(0,y.min())
        if y.max()==y.min():
            edges=None
        return edges    
        
    def _discretize_process(self, bins):
        """
        Method to discretize the process.
        Only Manual, KMeans, and Sigma methods are implemented
        """
        model = str(self.process_discretize or '')
        seed = self.kmeans_seed
        if isinstance(bins, int):
            self.nstates = bins
            if model.lower()=='sigma':
                edges = self._get_edges(self.data, bins)
            elif model.lower()=='kmeans':
                cl = MiniBatchKMeans(n_clusters=bins, random_state=seed)
                cl.fit(self.data.values.reshape((self.nobs,1)))
                self.params['KMeans results'] = cl
                edges = self.data.groupby(cl.labels_).max().values.tolist()
                edges.append(self.data.min())
                edges = sorted(edges)
            else:
                raise Exception("The process discretization must be 'Manual', Sigma' or 'KMeans'")
            self.params['Process bins'] = edges
            self.discretized = True
        elif isinstance(bins, list) & (model.lower()=='manual'):
            self.nstates = len(bins)-1
            self.params['Process bins'] = bins
            edges = bins
        else:
            self.nstates = self.data.unique()
            return self.data.copy()
        return pd.cut(self.data, edges, labels=False, include_lowest=True)-np.floor((self.nstates/2)).astype(dtype='int')
        
    def _compute_index(self, verbose):
        """
        Method to compute the index of the process.
        Only EWMA and GARCH methods are implemented
        """
        if verbose:
            display = 5
        else:
            display = 0
        
        if self.index_model == 'EWMA':
            if self.smooth is None:
                raise Exception('Lambda smoothing parameters is required')
            Jsq = self.J**2
            index = Jsq.ewm(alpha=(1-self.smooth)).mean()
            index_recursive_params = [0,1-self.smooth,self.smooth]
        elif self.index_model == 'GARCH':
            am = arch_model(self.J.astype(dtype='float'), mean='zero', vol=self.index_model)
            res = am.fit(update_freq=display, disp='off')
            self.params['GARCH results'] = res
            if res.optimization_result.success:
                index = res.conditional_volatility**2
                index_recursive_params = res.params.values
            else:
                raise Exception('GARCH optimization failed', res.optimization_result) 
        else:
            raise Exception('Wrong model type. Only EWMA and GARCH are implemented.')
        
        return index, index_recursive_params
    
    def _check_index_bins(self, edges):
        if isinstance(edges, list):
            vmax = self.index_cont.max()
            vmin = self.index_cont.min()
            if (vmin<min(edges)) | (vmax>max(edges)):
                raise Exception(f'The index_bins values {edges} must include all continuous index values. \
                                The max value of the continuous index is {vmax}')
            if sum(vmax<edges)>1:
                raise Exception(f'The index_bins values {edges} are too wide. \
                                The max value of the continuous index is {vmax}')
    
    def _discretize_index(self, bins):
        """
        Method to discretize the index of the process.
        Only Manual, KMeans, and Uniform methods are implemented
        """
        model = str(self.index_discretize or '')
        seed = self.kmeans_seed
        if (model.lower()=='manual') & isinstance(bins, list):      
            edges = bins
            self.index_nbins = len(bins)-1
        elif (model.lower()=='kmeans') & isinstance(bins, int):
            cl = MiniBatchKMeans(n_clusters=bins, random_state=seed)
            cl.fit(self.index_cont.values.reshape((self.njumps,1)))
            self.params['KMeans results'] = cl
            edges = self.index_cont.groupby(cl.labels_).max().values.tolist()
            edges.append(0)
            edges = sorted(edges)#[:-1]
            #edges.append(np.inf)
            self.index_nbins = bins
        elif (model.lower()=='uniform') & isinstance(bins, int):
            vmax = self.index_cont.max()
            bin_space = ((vmax-0)/bins)+0.000001
            edges = [i*bin_space for i in range(bins+1)]
            self.index_nbins = bins
        else:
            raise Exception('''The index discretization must be Manual, KMeans or Uniform. 
                               If Manual, a list of bin edges must be provided.
                               Otherwise, the number of bins must be provided.''')
            
        index = pd.cut(self.index_cont, edges, labels=False, include_lowest=True).astype(dtype='int')
        return index, edges
     
    def fit(self, Y, process_bins=None, index_bins=None, smooth=None, verbose=False):
        """Estimate WISM model.
        Parameters
        ----------
        Y: pd.Series,
            semi-Markov process.
            
        process_bins: list or int, default=None
            A list of bin edges for the 'Manual' discretization of the process
            or the number of bins for the 'KMeans' or 'Sigma' discretization of 
            the process.
            
        index_bins: list or int, default=None
            A list of bin edges for the 'Manual' discretization of the index
            or the number of bins for the 'KMeans' or 'Uniform' discretization of 
            the index.      
            
        smooth: int, default=None
            smoothing factor for the EWMA index model.
        
        verbose: bool, default=False
            Verbosity mode.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.data = Y # semi-Markov process
        self.data.name = 'Y'
        self.nobs = len(Y)
        self.Y = self._discretize_process(process_bins)     
        self.smooth = smooth
        
        # create the Jump process and related transition probabilities
        value_change = self.Y.ne(self.Y.shift())
        self.J = self.Y[value_change]
        self.J.name = 'J'
        self.njumps = len(self.J)
        
        if self.index_model is None:
            self.index_cont = pd.Series(self.J.var(), index=self.J.index)
            self.index = pd.Series(0, index=self.J.index)
            self.index_nbins = 1
            self.index_bins = [0,self.J.var()+1]
            self.index_recursive_params = [0,0,1]
        else:
            self.index_cont, self.index_recursive_params = self._compute_index(verbose)
            self._check_index_bins(index_bins)
            self.index, self.index_bins = self._discretize_index(index_bins)
            
        self.J = self.J.to_frame()
        self.J = self.J.assign(idx=self.index)

        # create the p_ij values and merge with the values
        p = self.J.copy()
        p.columns = ['i','idx']
        p['j'] = p['i'].shift(-1)
        p = p[:-1] # eliminate the last observation
        p['j'] = p['j'].astype(dtype='int')
        p['p'] = 1
        N_i = p.groupby(['idx','i'])['p'].count() # compute the denominator for q_ij
        p = p.groupby(['idx','i','j']).count()
        N_ij = p.copy() # save the denominator for f_ij
        # compute the p_ij
        p['p'] = p.reset_index().groupby(['idx','i'])['p'].transform(lambda x: x/x.sum()).values
        self.p = p
        
        if len(self.p.index)<(self.nstates*(self.nstates-1)*self.index_nbins):
            warnings.warn("Not all transition probabilities are available.")
        
        # create the T series with the sojourn times
        self.T = self.Y.groupby(value_change.cumsum()).value_counts().reset_index(level=0,drop=True).to_frame()
        self.T.columns=['T']
        self.T.reset_index(inplace=True)
        self.T.columns=['i','T']
        self.T = self.T.assign(idx=self.J['idx'].values)
        self.T['j'] = self.T['i'].shift(-1)
        self.T = self.T[:-1] # eliminate the last value as we don't know the arrivale state
        self.T['j'] = self.T['j'].astype(dtype='int')
        self.T['freq'] = 1

        # conditional sojourn time distributions: P(T_{n+1}-T_n=t|J_n=i, J_{n+1}=j)
        self.f = self.T.groupby(['idx','i','j','T'])['freq'].count().unstack().fillna(0).divide(N_ij.values, axis=0)
        # P(J_{n+1}=j, T_{n+1}-T_n=t|J_n=i)
        self.b = self.T.groupby(['idx','i','j','T'])['freq'].count().unstack().fillna(0).divide(N_i.reindex(p.reset_index(level=2).index).values, axis=0)

        # waiting time distribution function in state i given that the process will be in state j
        self.G = self.f.cumsum(axis=1) # P(T_{n+1}-T_n<=t|J_n=i, J_{n+1}=j)
        # semi-Markov kernel
        self.Q = self.b.cumsum(axis=1) # P(J_{n+1}=j, T_{n+1}-T_n<=t|J_n=i)
        # Survival function in state i
        self.H = self.Q.groupby('j').sum() # P(T_{n+1}-T_n<=t|J_n=i) = sum_j Q_ij(t)
        
        return self
        
    @property
    def prob(self):
        return self.p.unstack(level=2).fillna(0)
    
    def process_plot(self, figsize=None, nbins=200):
        """
        Plot the process series and its discretization
        """
        s = self.nstates
        bins = self.params['Process bins']
        data = self.data
        
        fig = figure(figsize=figsize)
        
        print('Discretization bins:', np.round(bins,3))
        print('Obs in each bin')
        print(self.Y.value_counts().sort_index())

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(data)
        ax.set_title('semi-Markov process')

        ax = fig.add_subplot(2, 2, 3)
        ax.hist(data, bins=nbins)
        ax.set_title(f'Histogram ({nbins} bins)')
        
        
        ax = fig.add_subplot(2, 2, 4)
        ax.hist(data, bins=bins)
        ax.set_title(f'Histogram with {s} states')
        fig.tight_layout()
        
    def index_plot(self, figsize=None, nbins=100):
        """
        Plot the continuous index series with its histogram and the
        fitted discretized index.
        """
        bin_model = self.index_discretize
        model = self.index_model
        n = self.index_nbins
        bins = self.index_bins

        fig = figure(figsize=figsize)

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(self.index_cont)
        ax.set_title(f'{model} index plot')

        ax = fig.add_subplot(2, 2, 3)
        ax.hist(self.index_cont, bins=nbins)
        ax.set_title(f'{model} index histogram ({nbins} bins)')
        
        
        ax = fig.add_subplot(2, 2, 4)
        arr = ax.hist(self.index_cont, bins=bins)
        for i in range(n):
            ax.text(arr[1][i],arr[0][i]+100,str(int(arr[0][i])))
        ax.set_title(f'{model} index histogram with {n} {bin_model} bins')
        fig.tight_layout()
     
    def _synt_acf(self, i, start, degree, lags):
        """
        Method to compute the autocorrelation function of the simulated 
        semi-Markov process
        """
        self.simulate_path(start)
        return stattools.acf(self.simulated_smp**degree, nlags=lags, fft=True)
    
    def compute_acf(self, degree=2, lags=100, start=100, simulations=1, cores=1):
        """
        Method to compute the autocorrelation function of the real and simulated
        semi-Markov process and the RMSE, MAE and MPE of both acf.
        
        Parameters
        ----------
        degree: int, default=2,
            Power of the process.
            
        lags: int, default=100
            Number of lags of the autocorrelation function.
            
        start: int, default=100
            Index to start the simulation.      
            
        simulations: int, default=1
            Number of simulations to create to compute the autocorrelation
            function. the resulting acf is the average of the simulated acf.
        
        cores: int, default=1
            Number of cores to use in the parallel processing. The system
            uses the Pool.imap_unordered function from multiprocessing.
            
        Returns
        -------
        ar_real : list
            autocorrelation values of the real semi-Markov process.
            
        ar_synt : list
            autocorrelation values of the synthetic semi-Markov process.
        """
        ar_real = stattools.acf(self.Y.iloc[start:]**degree, nlags=lags, fft=True)
        
        list_synt = []
        if cores>1:
            with Pool(cores) as p:
                for ar_synt in p.imap_unordered(partial(self._synt_acf, start=start, degree=degree, lags=lags), range(simulations)):
                    list_synt.append(ar_synt)
        else:
            for _ in range(simulations):
                self.simulate_path(start)
                list_synt.append(self._synt_acf(_, start, degree, lags))
                
        ar_synt = np.concatenate([list_synt], axis=0).mean(axis=0)
        ar_synt_error = np.concatenate([list_synt], axis=0).std(axis=0)/np.sqrt(simulations)
        
        diff = np.array([(r-s) for r, s in zip(ar_real, ar_synt)])
        perc_diff = np.array([(r-s)/r for r, s in zip(ar_real, ar_synt)])
        sq_diff = diff**2
        abs_diff = np.abs(diff)
        self.RMSE = np.sqrt(np.sum(sq_diff)/len(sq_diff))
        self.MAE = np.sum(abs_diff)/len(abs_diff)
        self.MPE = np.sum(perc_diff)/len(perc_diff)
        return ar_real, ar_synt
    
    @property
    def acf_score(self):
        return {'RMSE':self.RMSE, 'MAE':self.MAE, 'MPE':self.MPE}
    
    def acf_plot(self, degree=2, lags=100, start=100, simulations=1, cores=1, figsize=None):
        """
        Method to plot the autocorrelation function of the simulated and real 
        semi-Markov process. 
        """
        ar_real, ar_synt = self.compute_acf(degree, lags, start, simulations, cores)
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylim((0,0.3))
        ax.plot(ar_real, label='Real')
        ax.plot(ar_synt, label='Synth')
        ax.legend()
        
    def simulate_path(self, start=100):
        """
        Method to simulate a path of the semi-Markov process. 
        
        Parameters
        ----------
        start: int, default=100,
            Index to start the simulation.
            
        Returns
        -------
        simulated_mrp : Pandas DataFrame
            Simulated Markov Renewal process
            
        simulated_smp : Pandas Series
            Simulated semi-Markov process
        """
        N = self.njumps
        index_bins = np.ascontiguousarray(self.index_bins)
        omega, alpha, beta = self.index_recursive_params
        
        # compute the random selection of J and duration for each combination
        # of i, j states and duration
        J_dict = self.prob.copy()
        J_values = J_dict.columns.get_level_values(1).tolist()
        J_dict.columns = J_values
        J_dict = J_dict.T.to_dict('list')
        f_dict = self.f.T.to_dict('list')
        f_values = self.f.columns.tolist()
        
        random_J = {}
        for key in J_dict:
            random_J[key] = np.random.choice(J_values, size=N, p=J_dict[key])

        random_f = {}
        for key in f_dict:
            random_f[key] = np.random.choice(f_values, size=N, p=f_dict[key])
        
        # Set initial values
        vol_prev = self.index_cont.iloc[start]
        vbin_prev = self.index.iloc[start]
        J_prev = self.J.iloc[start,0]
        T = start
        
        smp = [] # semi-Markov process
        mrp = [] # Markov Renewal process
        i = 0
        while T<N:
            # sample J from transition probability matrix given previous state and vol level
            # J_next = np.random.choice(J_values, p=J_dict[(vbin_prev,J_prev)])
            J_next = random_J[(vbin_prev,J_prev)][i]
            # sample W from f          
            # duration = np.random.choice(f_values, p=f_dict[(vbin_prev,J_prev,J_next)])
            duration = random_f[(vbin_prev,J_prev,J_next)][i]
            # compute GARCH or EWMA recursively
            vol_next = omega + alpha*J_prev**2 + beta*vol_prev
            vbin_next = bisect_left(index_bins, vol_next)-1
            
            i += 1 # counter to select random values  
            if i>=N: # reset if exceeds the max index of presampled random values
                i = 0
            if (vbin_next,J_next) not in random_J.keys():
                continue
                
            
            mrp.append([J_prev, duration, vbin_prev])
            smp.append([J_prev]*duration)
            J_prev, vbin_prev, vol_prev = J_next, vbin_next, vol_next
            T += 1
            
        self.simulated_mrp = pd.DataFrame(mrp, columns=['J','T','idx'])
        self.simulated_smp = pd.Series(list(chain(*smp)), name='Y')
        return self.simulated_mrp, self.simulated_smp