import numpy as np

from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy import optimize
import warnings


# Flatten
def one_d_flatten(array_tiff, scope='whole_y', region=[]):
    assert scope in ['line_x', 'whole_x', 'line_y', 'whole_y']
    if region is not None:
        assert (np.array(array_tiff.shape) == np.array(region.shape)).all() # shape should match
    else:
        region = np.ones_like(array_tiff, dtype=np.bool_)

    image = array_tiff.copy()
    H, W = image.shape
    match scope:
        case 'line_x' | 'whole_x':
            A = np.ones((W, 2))
            A[:, 0] = (np.arange(1, W + 1) * 1.0 / W)
            results = []
            for i in range(H):
                img_row, reg_row = image[i, :], region[i, :]
                valid_n = np.sum(reg_row)
                if valid_n <= 2: # no region selected
                    bias = np.mean(img_row)
                    result_row = img_row - bias
                else:
                    coef, resids, rank, s = np.linalg.lstsq(A[reg_row, :], img_row[reg_row], rcond=None)
                    result_row = img_row - np.dot(A, coef)
                results.append(result_row)
            result_image = np.stack(results, axis=0)
            line_image = image - result_image
            avg_line = np.mean(line_image, axis=0, keepdims=True)
    
            if scope == 'whole_x':
                result_image = image - avg_line

        case 'line_y' | 'whole_y':
            A = np.ones((H, 2))
            A[:, 0] = (np.arange(1, H + 1) * 1.0 / H)
            results = []
            for i in range(W):
                img_column, reg_column = image[:, i], region[:, i]
                valid_n = np.sum(reg_column)
                if valid_n <= 2: # no region selected
                    bias = np.mean(img_column)
                    result_column = img_column - bias
                else:
                    coef, resids, rank, s = np.linalg.lstsq(A[reg_column, :], img_column[reg_column], rcond=None)
                    result_column = img_column - np.dot(A, coef)
                results.append(result_column)
            result_image = np.stack(results, axis=1)
            line_image = image - result_image
            avg_line = np.mean(line_image, axis=1, keepdims=True)
    
            if scope == 'whole_y':
                result_image = image - avg_line

    return result_image


def n_d_legendre_flatten(array_tiff, scope='line_x',degree = 2 ,region=None):
    assert scope in ['line_x', 'whole_x', 'line_y', 'whole_y']
    if region is not None:
        assert (np.array(array_tiff.shape) == np.array(region.shape)).all() # shape should match
    else:
        region = np.ones_like(array_tiff, dtype=np.bool_)

    image = array_tiff.copy()
    H, W = image.shape
    match scope:
        case 'line_x' | 'whole_x':
            A = np.zeros((W, degree + 1))
            for i in range(degree + 1):
                A[:, i] = __legendre_polynomial(i, np.arange(1, W + 1) )
            results = []
            for i in range(H):
                img_row, reg_row = image[i, :], region[i, :]
                valid_n = np.sum(reg_row)
                coef, _, _ , _ = np.linalg.lstsq(A[reg_row,:], img_row[reg_row], rcond=None)
                result_row = img_row - np.dot(A, coef) 
                results.append(result_row)
            result_image = np.stack(results, axis=0)
            line_image = image - result_image
            
            if scope == 'whole_x':
                avg_line = np.mean(line_image, axis=0, keepdims=True)
                result_image = image - avg_line

        case 'line_y' | 'whole_y':
            A = np.zeros((H, degree + 1))
            for i in range(degree + 1):
                A[:, i] = __legendre_polynomial(i, np.arange(1, H + 1))
            results = []
            for i in range(W):
                img_column, reg_column = image[:, i], region[:, i]
                valid_n = np.sum(reg_column)
                if valid_n <= 2: # no region selected
                    bias = np.mean(img_column)
                    result_column = img_column - bias
                else:
                    coef, resids, rank, s = np.linalg.lstsq(A[reg_column, :], img_column[reg_column], rcond=None)
                    result_column = img_column - np.dot(A, coef)
                results.append(result_column)
            result_image = np.stack(results, axis=1)
            line_image = image - result_image
            
            if scope == 'whole_y':
                avg_line = np.mean(line_image, axis=1, keepdims=True)
                result_image = image - avg_line
    return result_image

def __legendre_polynomial(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_n_minus_2 = np.ones_like(x)
        P_n_minus_1 = x
        for k in range(2, n+1):
            P_n = ((2*k - 1) * x * P_n_minus_1 - (k - 1) * P_n_minus_2) / k
            P_n_minus_2 = P_n_minus_1
            P_n_minus_1 = P_n
        return P_n

def grating_flatten(array_tiff):
    whole_x = one_d_flatten(array_tiff,scope='whole_x')
    whole_x_y = one_d_flatten(whole_x,scope='whole_y')
    top_region, bottom_region = __grating_region(whole_x_y)
    whole_x_y_top = one_d_flatten(whole_x_y,scope='line_x',region=top_region)
    whole_x_y_top_bottom = one_d_flatten(whole_x_y_top,scope='line_x',region=bottom_region)
    return whole_x_y_top_bottom


def n_d_grating_flatten(array_tiff, x_degree, y_degree):
    whole_x = n_d_legendre_flatten(array_tiff,scope='whole_x',degree=x_degree)
    whole_x_y = n_d_legendre_flatten(whole_x,scope='whole_y',degree=y_degree)
    top_region, bottom_region = __grating_region(whole_x_y)
    whole_x_y_top = one_d_flatten(whole_x_y,scope='line_x',region=top_region)
    whole_x_y_top_bottom = one_d_flatten(whole_x_y_top,scope='line_x',region=bottom_region)
    return whole_x_y_top_bottom

def cluster_kmeans(gray,n_cluster:int=2):
    data  = np.reshape(gray.copy(), (-1,1))
    kmeans = KMeans(n_clusters=n_cluster,n_init=10 ,tol=1e-6,random_state=0)
    #kmeans = KMeans(n_clusters=n_cluster,n_init='auto',tol=1e-4)
    kmeans.fit_predict(data)
    cluster_id = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    split_data_hist = []
    split_data_edges = []
    for id in np.unique(cluster_id):
        subset = data[cluster_id==id]
        hist, edges = region_histogram(subset)
        split_data_hist.append(hist)
        split_data_edges.append(edges)
    return split_data_hist, split_data_edges

def one_peak_gaussian_fit(x,y):
    y, denominator = __norm_gaussian(x,y)
    popt_gauss = __one_peak_gaussian_fit(x, y)
    fit_data = __one_peak_gaussian(x, *popt_gauss)
    return fit_data*denominator, popt_gauss

# Metric
def cal_statistic(data):
    mean = np.mean(data)
    minimum = np.min(data)
    maximum = np.max(data)
    return mean, minimum, maximum
    
def cal_r_q(data):
    return np.std(data)

def cal_average_line(data, axis:int):
    avg_line = np.sum(data, axis = axis ) / np.shape(data)[axis]
    return avg_line

def cal_r_pv(data, axis:int = 0):
    avg_line = cal_average_line(data,axis)
    return np.max(avg_line) - np.min(avg_line)

def rel_err(ideal:float,measure:float):
    eps = 7/3. - 4/3. -1 
    return abs(ideal-measure)/(ideal+eps)

def region_histogram(region):
    bins = __optimal_bins(region)
    hist,edges = np.histogram(region,bins)
    return hist, edges

def cal_histogram_range(flatten_image):
    split_hists, split_edges = cluster_kmeans(flatten_image)
    # change unit micro -> pico
    scale = 1000000
    # popt_gauss = (amplitude, center, sigma)
    _, popt_gauss_1 = one_peak_gaussian_fit(split_edges[0][:-1]*scale, split_hists[0])
    _, popt_gauss_2 = one_peak_gaussian_fit(split_edges[1][:-1]*scale, split_hists[1])
    #change unit pico -> micro -> nano
    popt_gauss_1 = list(popt_gauss_1)
    popt_gauss_2 = list(popt_gauss_2)
    popt_gauss_1[1] = popt_gauss_1[1]/scale * 1000
    popt_gauss_2[1] = popt_gauss_2[1]/scale * 1000
    histogram_range = np.abs((popt_gauss_1[1]-popt_gauss_2[1]))
    return histogram_range, popt_gauss_1, popt_gauss_2

def cal_histogram_range_with_infimum(flatten_image):
    split_hists, split_edges = cluster_kmeans(flatten_image)
    # change unit Pico -> Micro
    scale = 1000000
    ## add dummy infimum value: len(hist)+1 = len(edges)
    split_hists[0] = np.insert(split_hists[0],0,0)
    split_hists[1] = np.insert(split_hists[1],0,0)
    # popt_gauss = (amplitude, center, sigma)
    _, popt_gauss_1 = one_peak_gaussian_fit(split_edges[0]*scale, split_hists[0])
    _, popt_gauss_2 = one_peak_gaussian_fit(split_edges[1]*scale, split_hists[1])
    #change unit Micro -> Pico -> nano
    popt_gauss_1 = list(popt_gauss_1)
    popt_gauss_2 = list(popt_gauss_2)
    popt_gauss_1[1] = popt_gauss_1[1]/scale * 1000
    popt_gauss_2[1] = popt_gauss_2[1]/scale * 1000
    histogram_range = np.abs((popt_gauss_1[1]-popt_gauss_2[1]))
    return histogram_range, popt_gauss_1, popt_gauss_2

def calculate_psd_rq(freq,P_den):
   P = np.trapz(P_den,freq) ## um^2
   Rq = np.sqrt(P) ## um
   return P, Rq

def calculate_psd(tiff_dict):
    size_x = tiff_dict['HEADER']['scanSizeWidth']
    size_y = tiff_dict['HEADER']['scanSizeHeight']
    Freq_x, tmp_x = _calculate_psd(tiff_dict['IMAGE'],size_x,axis=-1)
    Freq_y, tmp_y = _calculate_psd(tiff_dict['IMAGE'],size_y,axis=0)
    X_value = np.mean(tmp_x,axis=0)
    Y_value = np.mean(tmp_y,axis=1)
    return X_value, Y_value, Freq_x, Freq_y,tmp_x, tmp_y

## Private
def __minMax(npy):
    # Min = 0, Max = 1
    min_npy = np.min(npy)
    denominator = (np.max(npy) - min_npy) + 1e-16
    return (npy - min_npy) / denominator, denominator

def __norm_gaussian(x,y):
    # integral = 1
    y = np.abs(y)
    eps = 7/3. - 4/3. -1 
    denominator = np.abs(trapz(x,y)) + eps
    return y / denominator, denominator

def __optimal_bins(gray):
    vmin = np.min(gray)
    vmax = np.max(gray)
    # cvt Scale (Pico -> Nano)
    bound = int((vmax - vmin)* 1000)
    if bound < 10:
        bound = 128
    count = gray.size
    sqrt_count = int(np.sqrt(count))

    if (sqrt_count < bound) & (count < bound * 50):
        bins = sqrt_count                                                  
    else: bins = bound
    if bins > 128:
        bins = 128
    return bins

def __grating_region(array_tiff):
    """K-Means 기반의 영역 분할"""
    split_hists, split_edges = cluster_kmeans(array_tiff)
    min_1 = np.min(split_edges[0])
    min_2 = np.min(split_edges[1])
    max_1 = np.max(split_edges[0])
    max_2 = np.max(split_edges[1])
    
    if max_1 < min_2:
        split_pos = min_2
    elif max_2 < min_1:
        split_pos = min_1
    else:
        print('ERROR')
    region = np.zeros_like(array_tiff, dtype=np.bool_)
    top_region = np.where(array_tiff > split_pos ,True, region)
    bottom_region = np.where(array_tiff <= split_pos,True,region)
    return top_region, bottom_region

def __clip_negative(x,y):
    clip_index = np.where(y > 0)
    return x[clip_index], y[clip_index]

def __interpolate(x,y, num:int=10000):
    fun = interp1d(x,y,kind='cubic')
    x_new = np.linspace(x.min(), x.max(),num=num,endpoint=True)
    y_new = fun(x_new)
    return x_new, y_new

def __one_peak_gaussian_fit(x_ori,y_ori):
    """Do not use pico & Nano scale
        fit_data = __one_peak_gaussian(x_axis,popt_gauss[0],popt_gauss[1],popt_gauss[2])
        fit_data = __one_peak_gaussian(x_axis,*popt_gauss)
    """
    x = x_ori.copy()
    y = y_ori.copy() 
    x, y = __interpolate(x, y)
    x,y = __clip_negative(x, y)

    amp = y.max()
    cen = x[np.where(y == amp)][0]
    sigma = np.std(y)
    warnings.filterwarnings("error", category=UserWarning)
    try:
        popt_gauss, _ = optimize.curve_fit(__one_peak_gaussian, x, y, p0=[amp, cen, sigma])
    except UserWarning as e:
        popt_gauss = (amp,cen,sigma)
    except RuntimeError:
        popt_gauss = (amp,cen,sigma)
    return popt_gauss

def __one_peak_gaussian(x, amp,cen,sigma):
    eps = 7/.3 - 4/.3 - 1
    return amp *( 1 / (eps + sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x-cen) / (sigma + eps))**2)))

def __calculate_psd(signal,scan_size,axis=-1):
    from scipy.signal import periodogram
    fs = len(signal)/scan_size
    freq, Pxx_den = periodogram(signal,fs,scaling='density',axis=axis,nfft=None)
    return freq, Pxx_den
