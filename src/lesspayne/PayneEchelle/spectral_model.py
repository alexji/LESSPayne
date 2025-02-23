# code for a spectral model, i.e. predicting the spectrum of a single star in normalized space.
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy import interpolate
from scipy import signal
from scipy.stats import norm

#=======================================================================================================================

def leaky_relu(z):
    '''
    This is the activation function used by default in all our neural networks.
    '''
    return z*(z > 0) + 0.01*z*(z < 0)

def sigmoid(z):
    '''
    standard sigmoid
    '''
    return 1./(1 + np.exp(-z))

#--------------------------------------------------------------------------------------------------------------------------

def get_spectrum_from_neural_net(scaled_labels, NN_coeffs):

    '''
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit).
    Each label ranges from -0.5 to 0.5
    '''
    
    # assuming your NN has two hidden layers.
    w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
    spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
    return spectrum

class SpectralModel(object):
    """
    A class that encompasses a Payne spectral model.
    
    The coefficients of the model in order are:
    num_label: stellar labels (this is the trained NN)
    num_order*(polynomial_order+1): number of polynomial coefficients (this is the continuum model)
    num_chunk*2: number of nuisance parameters (this is the RV and vbroad)
    
    chunk_order_min and chunk_order_max specify which indices are used in each chunk.
    The indices are inclusive.
      Ex: if two chunks of 10 and 12 orders, it should be:
      chunk_order_min = [0, 10]
      chunk_order_max = [9, 21]
    """
    
    def __init__(self,
            NN_coeffs,
            num_stellar_labels,
            x_min, x_max,
            wavelength_payne,
            errors_payne,
            num_order, polynomial_order,
            num_chunk,
            chunk_order_min=None, chunk_order_max=None,
    ):
        self._NN_coeffs = NN_coeffs
        self._num_stellar_labels = num_stellar_labels
        self._x_min = x_min
        self._x_max = x_max
        self._wavelength_payne = wavelength_payne
        self._errors_payne = errors_payne
        self._num_order = num_order
        self._polynomial_order = polynomial_order
        self._num_chunk = num_chunk
        
        if chunk_order_min is None and chunk_order_max is None:
            self.chunk_order_min = [0]
            self.chunk_order_max = [self.num_order-1]
        else:
            self.chunk_order_min = chunk_order_min
            self.chunk_order_max = chunk_order_max
        
        self._verify_chunks()

    ### Functions to define in subclasses
    @staticmethod
    def load(fname, num_order, polynomial_order=6, errors_payne=None,
             num_chunk=1, chunk_order_min=None, chunk_order_max=None):
        """
        """
        raise NotImplementedError
    def get_spectrum_from_neural_net(self, scaled_labels):
        """
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        """
        raise NotImplementedError
    
    ### Functions with default behavior you may want to redefine
    def transform_coefficients(self, popt):
        """
        Transform coefficients into human-readable
        """
        popt_new = popt.copy()
        popt_new[:self.num_stellar_labels] = (popt_new[:self.num_stellar_labels] + 0.5)*(self.x_max-self.x_min) + self.x_min
        popt_new[0] = popt_new[0]*1000.
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            popt_new[irv] = popt_new[irv]*100.
        return popt_new
    def normalize_stellar_labels(self, labels):
        """
        Turn physical stellar parameter values into normalized values.
        """
        labels = np.ravel(labels)
        labels[0] = labels[0]/1000.
        new_labels = (labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        assert np.all(np.round(new_labels,3) >= -0.51), (new_labels, labels)
        assert np.all(np.round(new_labels,3) <=  0.51), (new_labels, labels)
        return new_labels
    def get_p0_initial_normspec(self, initial_labels=None, initial_rv=0.0, initial_vbroad=0.5):
        """
        Get p0 for optimization (assuming continuum params are all 0))
        """
        p0_initial = np.zeros(self.num_stellar_labels + self.coeff_poly*self.num_order + 2*self.num_chunk)
        ## Stellar labels
        if initial_labels is not None:
            p0_initial[0:self.num_stellar_labels] = self.normalize_stellar_labels(initial_labels)
        ## Continuum: set each order to a constant. The other coeffs are 0.
        p0_initial[self.num_stellar_labels::self.coeff_poly] = 1.0
        ## vbroad/rv
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)
            p0_initial[ivbroad] = initial_vbroad
            p0_initial[irv] = initial_rv
        return p0_initial
    def get_initial_bounds(self, bounds_set=None,
                           vbroadmin=0.1, vbroadmax=10,
                           rvmin=-500, rvmax=500,
                           poly_coeff_min=-1000, poly_coeff_max=1000):
        bounds = np.zeros((2, self.num_all_labels))
        # polynomial coefficients
        bounds[0,self.num_stellar_labels:] = poly_coeff_min
        bounds[1,self.num_stellar_labels:] = poly_coeff_max
        
        if bounds_set is None:
            bounds[0,:self.num_stellar_labels] = -0.5
            bounds[1,:self.num_stellar_labels] = 0.5
            
            for ichunk in range(self.num_chunk):
                irv = -1 - 2*(self.num_chunk - ichunk - 1)
                ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)
                bounds[0,ivbroad] = vbroadmin
                bounds[1,ivbroad] = vbroadmax
                bounds[0,irv] = rvmin/100.
                bounds[1,irv] = rvmax/100.
        else:
            bounds[:,:self.num_stellar_labels] = bounds_set[:,:self.num_stellar_labels]
            bounds[:,-2*self.num_chunk:] = bounds_set[:,-2*self.num_chunk:]
        return bounds
    def get_print_string(self, params):
        pprint =  self.transform_coefficients(params)
        spstr = f"Teff={pprint[0]:.0f} logg={pprint[1]:.2f} FeH={pprint[2]:.2f} aFe={pprint[3]:.2f}"
        chunkstrs = []
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)
            chunkstrs.append(f"  chunk {self.chunk_order_min[ichunk]}-{self.chunk_order_max[ichunk]} rv={pprint[irv]:.1f} vbroad={pprint[ivbroad]:.1f}")
        chunkstr = "\n".join(chunkstrs)
        return spstr+"\n"+chunkstr
        
    ### The main model evaluation
    def evaluate(self, labels, wavelength, wavelength_normalized=None):
        """
        Evaluate this model at these labels and wavelength
        """
        # Get normalized wavelength for continuum evaluation
        if wavelength_normalized is None:
            wavelength_normalized = self.whitten_wavelength(wavelength)*100.
        
        num_order, num_pixel = wavelength.shape
        spec_predict = np.zeros(num_order*num_pixel)
        errs_predict = np.zeros(num_order*num_pixel)
        
        # make payne models
        _full_spec = self.get_spectrum_from_neural_net(
            scaled_labels = labels[:self.num_stellar_labels]
        )
        
        # allow different RV and broadening for each chunk
        spec_predict = np.zeros(num_order*num_pixel)
        errs_predict = np.zeros(num_order*num_pixel) 
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)

            # Broadening kernel
            win = norm.pdf((np.arange(21)-10.)*(self.wavelength_payne[1]-self.wavelength_payne[0]),\
                           scale=labels[ivbroad]/3e5*5000)
            win = win/np.sum(win)
            
            # vbroad and RV
            full_spec = signal.convolve(_full_spec, win, mode='same')
            full_spec = self.doppler_shift(self.wavelength_payne, full_spec, labels[irv]*100.)
            errors_spec = self.doppler_shift(self.wavelength_payne, self.errors_payne, labels[irv]*100.)
        
            # interpolate into the observed wavelength
            f_flux_spec = interpolate.interp1d(self.wavelength_payne, full_spec)
            f_errs_spec = interpolate.interp1d(self.wavelength_payne, errors_spec)
            
            # loop over all orders
            for k in range(self.chunk_order_min[ichunk], self.chunk_order_max[ichunk]+1):
                scale_poly = 0
                for m in range(self.coeff_poly):
                    scale_poly += (wavelength_normalized[k,:]**m)*labels[self.num_stellar_labels+self.coeff_poly*k+m]
                spec_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_flux_spec(wavelength[k,:])
                errs_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_errs_spec(wavelength[k,:])
        
        return spec_predict, errs_predict
    
    ### Generally useful static methods
    @staticmethod
    def whitten_wavelength(wavelength):
        """
        normalize the wavelength of each order to facilitate the polynomial continuum fit
        """
        wavelength_normalized = np.zeros(wavelength.shape)
        for k in range(wavelength.shape[0]):
            mean_wave = np.mean(wavelength[k,:])
            wavelength_normalized[k,:] = (wavelength[k,:]-mean_wave)/mean_wave
        return wavelength_normalized
    @staticmethod
    def doppler_shift(wavelength, flux, dv):
        """
        dv is in km/s
        positive dv means the object is moving away.
        """
        c = 2.99792458e5 # km/s
        doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
        new_wavelength = wavelength * doppler_factor
        new_flux = np.interp(new_wavelength, wavelength, flux)
        return new_flux
    
    ### Class Properties
    # Properties of the Payne model
    @property
    def NN_coeffs(self):
        return self._NN_coeffs
    @property
    def num_stellar_labels(self):
        return self._num_stellar_labels
    @property
    def x_min(self):
        return self._x_min
    @property
    def x_max(self):
        return self._x_max
    @property
    def wavelength_payne(self):
        return self._wavelength_payne
    @property
    def errors_payne(self):
        return self._errors_payne
    # Nuisance parameters
    @property
    def num_order(self):
        return self._num_order
    @property
    def coeff_poly(self):
        return self._polynomial_order + 1
    @property
    def polynomial_order(self):
        return self._polynomial_order
    @property
    def num_chunk(self):
        return self._num_chunk
    @property
    def num_all_labels(self):
        return self.num_stellar_labels + self.coeff_poly*self.num_order + 2*self.num_chunk
    
    ### Setters
    def set_polynomial_order(self, polynomial_order):
        self._polynomial_order = polynomial_order
    def set_num_order(self, num_order):
        self._num_order = num_order
    
    ### Internal functions
    def _verify_chunks(self):
        assert self.num_chunk == len(self.chunk_order_min)
        assert self.num_chunk == len(self.chunk_order_max)
        
        all_orders = [np.arange(self.chunk_order_min[i], self.chunk_order_max[i]+1) for i in range(self.num_chunk)]
        all_orders = np.concatenate(all_orders)
        assert len(all_orders) == self.num_order, (len(all_orders), self.num_order)
        assert len(all_orders) == len(np.unique(all_orders))

class DefaultPayneModel(SpectralModel):
    @staticmethod
    def load(fname, num_order, polynomial_order=6, errors_payne=None,
             num_chunk=1, chunk_order_min=None, chunk_order_max=None):
        
        tmp = np.load(fname)
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        wavelength_payne = tmp["wavelength_payne"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
        
        num_stellar_labels = w_array_0.shape[1]
        
        if errors_payne is None:
            errors_payne = np.zeros_like(wavelength_payne)
        
        return DefaultPayneModel(
            NN_coeffs, num_stellar_labels, x_min, x_max,
            wavelength_payne, errors_payne,
            num_order, polynomial_order, num_chunk,
            chunk_order_min=chunk_order_min,
            chunk_order_max=chunk_order_max
        )

    def get_spectrum_from_neural_net(self, scaled_labels):
        """
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        """
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self.NN_coeffs
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2
        return spectrum
        
class DefaultMIKEModel(DefaultPayneModel):
    @staticmethod
    def load(fname, num_order_blue, num_order_red,
             polynomial_order=6, errors_payne=None):
        
        tmp = np.load(fname)
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        wavelength_payne = tmp["wavelength_payne"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
        
        num_stellar_labels = w_array_0.shape[1]
        
        if errors_payne is None:
            errors_payne = np.zeros_like(wavelength_payne)
        
        num_order = num_order_blue + num_order_red
        num_chunk = 2
        chunk_order_min = [0, num_order_blue]
        chunk_order_max = [num_order_blue-1, num_order-1]
        
        return DefaultPayneModel(
            NN_coeffs, num_stellar_labels, x_min, x_max,
            wavelength_payne, errors_payne,
            num_order, polynomial_order, num_chunk,
            chunk_order_min=chunk_order_min,
            chunk_order_max=chunk_order_max
        )

class YYLiPayneModel(SpectralModel):
    @staticmethod
    def load(fname, num_order, polynomial_order=6, errors_payne=None,
             num_chunk=1, chunk_order_min=None, chunk_order_max=None):
        
        tmp = np.load(fname)
        w_array_0 = tmp["w_array_0"]
        w_array_1 = tmp["w_array_1"]
        w_array_2 = tmp["w_array_2"]
        b_array_0 = tmp["b_array_0"]
        b_array_1 = tmp["b_array_1"]
        b_array_2 = tmp["b_array_2"]
        x_min = tmp["x_min"]
        x_max = tmp["x_max"]
        wavelength_payne = tmp["wavelength_payne"]
        NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
        
        num_stellar_labels = w_array_0.shape[1]
        # Teff, logg, vt, feh, c/fe, mg/fe, ca/fe and ti/fe
        
        if errors_payne is None:
            errors_payne = np.zeros_like(wavelength_payne)
        
        return YYLiPayneModel(
            NN_coeffs, num_stellar_labels, x_min, x_max,
            wavelength_payne, errors_payne,
            num_order, polynomial_order, num_chunk,
            chunk_order_min=chunk_order_min,
            chunk_order_max=chunk_order_max
        )


    def get_spectrum_from_neural_net(self, scaled_labels):
        """
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        """

        # assuming your NN has two hidden layers.
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self.NN_coeffs
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, sigmoid(outside)) + b_array_2
        return sigmoid(spectrum)
    
    ### Functions with default behavior you may want to redefine
    def get_p0_initial_normspec(self, initial_labels=None, initial_rv=0.0, initial_vbroad=0.5):
        """
        Get p0 for optimization (assuming continuum params are all 0))
        """
        p0_initial = np.zeros(self.num_stellar_labels + self.coeff_poly*self.num_order + 2*self.num_chunk)
        ## Stellar labels
        if initial_labels is not None:
            p0_initial[0:self.num_stellar_labels] = self.normalize_stellar_labels(initial_labels)
        ## Continuum: set each order to a constant. The other coeffs are 0.
        p0_initial[self.num_stellar_labels::self.coeff_poly] = 1.0
        ## vbroad/rv
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)
            p0_initial[ivbroad] = initial_vbroad
            p0_initial[irv] = initial_rv
        return p0_initial
    def get_print_string(self, params):
        pprint =  self.transform_coefficients(params)
        # Teff, logg, vt, feh, c/fe, mg/fe, ca/fe and ti/fe
        spstr1 = f"Teff={pprint[0]:.0f} logg={pprint[1]:.2f} vt={pprint[2]:.2f} FeH={pprint[3]:+.2f}"
        spstr2 = f"CFe={pprint[4]:+.2f} MgFe={pprint[5]:+.2f} CaFe={pprint[6]:+.2f} TiFe={pprint[7]:+.2f}"
        spstr = f"{spstr1}\n{spstr2}"
        chunkstrs = []
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            ivbroad = -2 - 2*(self.num_chunk - ichunk - 1)
            chunkstrs.append(f"  chunk {self.chunk_order_min[ichunk]}-{self.chunk_order_max[ichunk]} rv={pprint[irv]:.1f} vbroad={pprint[ivbroad]:.1f}")
        chunkstr = "\n".join(chunkstrs)
        return spstr+"\n"+chunkstr

    def transform_coefficients(self, popt):
        """
        Transform coefficients into human-readable
        """
        popt_new = popt.copy()
        popt_new[:self.num_stellar_labels] = (popt_new[:self.num_stellar_labels] + 0.5)*(self.x_max-self.x_min) + self.x_min
        for ichunk in range(self.num_chunk):
            irv = -1 - 2*(self.num_chunk - ichunk - 1)
            popt_new[irv] = popt_new[irv]*100.
        return popt_new
    def normalize_stellar_labels(self, labels):
        """
        Turn physical stellar parameter values into normalized values.
        """
        labels = np.ravel(labels)
        new_labels = (labels - self.x_min) / (self.x_max - self.x_min) - 0.5
        assert np.all(np.round(new_labels,3) >= -0.51), (new_labels, labels)
        assert np.all(np.round(new_labels,3) <=  0.51), (new_labels, labels)
        return new_labels
    
