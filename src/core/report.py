import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import NullLocator
import matplotlib.gridspec as gridspec
from matplotlib import cm
from glob import glob
from sklearn.cluster import KMeans
from os import makedirs
import matplotlib
matplotlib.use('agg')

from src.core.analysis import region_histogram, cal_histogram_range_with_infimum, cal_histogram_range

def __auto_palette_range(gray,cmap):
    # 95.44 % 
    vmin = np.mean(gray) - 1.96 * np.std(gray)
    vmax = np.mean(gray) + 1.96 * np.std(gray)
    hist, edges = np.histogram(gray, bins=cmap.N, range=(vmin, vmax))
    boundary_norm = BoundaryNorm(edges, cmap.N)
    return boundary_norm

def write_csv(data_dict,csv_path:str,metric_keys:list):
    format_string = '.3f'
    means = {metric: np.mean(data) for metric, data in data_dict.items() if metric in metric_keys}
    
    with open(csv_path, 'w', newline='\n') as csvfile:
        wr = csv.writer(csvfile)
        for metric, mean in means.items():
            wr.writerow([f'Average {metric}', f'{mean:{format_string}}'])
        wr.writerow([])   
        wr.writerow(data_dict.keys())

        for i in range(len(data_dict[metric_keys[0]])):
            #row = [data[i] for data in data_dict.values()]
            #wr.writerow(row)
            row = []
            for data in data_dict.values():
                if isinstance(data[i], float):
                    row.append(f'{data[i]:{format_string}}')
                else:
                    row.append(data[i])
            wr.writerow(row)
    return None

def plot_tiff(gray, cmap, size:tuple=(8,8)):
    fig = plt.figure(figsize=size, facecolor='white')
    boundary_norm = __auto_palette_range(gray, cmap)
    plt.imshow(gray,cmap=cmap,norm=boundary_norm)
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    return fig

def plot_tiff_3d(gray, cmap, size:tuple=(8,8)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    boundary_norm = __auto_palette_range(gray, cmap)
    x = range(gray.shape[0])
    y = range(gray.shape[1])
    mesh_x, mesh_y = np.meshgrid(x,y)
    ax.set_zlim(gray.min(), gray.max())
    surf = ax.plot_surface(mesh_x, mesh_y, gray, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return fig

def ls_tiff(dir:str):
    tiff_list = glob(dir+'/*.tiff')
    return tiff_list

def plot_region_histogram(region, size:tuple=(8,8)):
    # Be careful ifimum shift
    fig = plt.figure(figsize=size, facecolor='white')
    hist , edges = region_histogram(region)
    ## add dummy infimum value: len(hist)+1 = len(edges)
    pl_hist = np.insert(hist,0,0)
    plt.plot(edges*1000,pl_hist)
    plt.ylim([0, hist.max()])
    plt.xlim(edges.min()*1000,edges.max()*1000)
    return fig

def plot_region_histogram_range(region,size:tuple=(8,8)):
    # Be careful ifimum shift
    fig = plt.figure(figsize=size, facecolor='white')
    hist , edges = region_histogram(region)
    pl_hist = np.insert(hist,0,0)
    plt.plot(edges*1000,pl_hist,'k',linewidth=3,label='region histogram')

    histogram_range, popt_1, popt_2 = cal_histogram_range_with_infimum(region)
    histogram_range_cal, _, _ = cal_histogram_range(region)
    plt.vlines(popt_1[1],0, hist.max(), colors='r', linestyles='-', label='peak_1')
    plt.vlines(popt_2[1],0, hist.max(), colors='b', linestyles='-', label='peak_2')
    str_title_1 = f'Calculation range: {histogram_range_cal:4.2f} nm\n'
    str_title_2 = f'Plot range: {histogram_range:4.2f} nm'
    plt.title('Region hitogram\n'+str_title_1+str_title_2)
    plt.ylabel('pxl')
    plt.xlabel('nm')
    plt.ylim([0, hist.max()])
    plt.xlim(edges.min()*1000,edges.max()*1000)
    plt.legend()

    return fig

def plot_histogram(gray, cmap, size:tuple=(8, 8)):
    fig = plt.figure(figsize=size, facecolor='white')
    hist , edges = region_histogram(gray)
    pl_hist = np.insert(hist,0,hist[0])
    norm = __auto_palette_range(gray, cmap)
    num_segments = len(pl_hist) - 1
    segment_colors = [cmap(norm(value)) for value in edges]
    edges *= 1000 # cvt scale (pico -> nano)
    for i in range(num_segments):
        plt.fill_between(edges[i:i+2], pl_hist[i:i+2], color=segment_colors[i],step='post')
    plt.ylim([0, hist.max()])
    plt.xlim(edges.min(),edges.max())
    return fig

def plot_kmeans(gray, n_cluster:int=2,size:tuple=(8,8)):
    fig, ax = plt.subplots()
    fig.set_size_inches(size[0],size[1])
    data  = np.reshape(gray.copy(), (-1,1))
    kmeans = KMeans(n_clusters=n_cluster,n_init='auto')
    cluster_id = kmeans.fit_predict(data)
    bins = np.linspace(data.min(), data.max(), 128)
    for id in np.unique(cluster_id):
        subset = data[cluster_id==id]
        ax.hist(subset, bins=bins, alpha=0.9, label=f"Cluster {id}")
    ax.legend() 
    return fig

def default_report(file_name, dir, dict_tiff, cmap, data):
    #data = [['Metric', 'unit', float], ['Metric', 'unit, float]]
    #Init
    gray = dict_tiff['IMAGE'] 
    unit = dict_tiff['HEADER']['unit']
    xlabel = dict_tiff['HEADER']['scanSizeWidth']
    ylabel = dict_tiff['HEADER']['scanSizeHeight']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 13

    fig = plt.figure(figsize=(14,10)) #(width:column, height:row)
    plt.suptitle(file_name,fontsize=19,ha="left",va='top',weight='normal',x=0.05)
    fig.subplots_adjust(wspace = 1,hspace=1)
    gs = gridspec.GridSpec(6,11) #(row, column)

    ax1 = plt.subplot(gs[0:7,0:8])
    plt.tick_params(top=False,left=False,bottom=False,direction='in')
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.xlabel(f'{xlabel} µm')
    plt.ylabel(f'{ylabel} µm',rotation=-90,labelpad=15)
    plt.ylabel(f'{ylabel} µm')
    ax1.yaxis.set_label_position('right')

    boundary_norm = __auto_palette_range(gray, cmap)
    if (xlabel == 0) | (ylabel == 0):
        aspect = 'auto'
    else: aspect = ylabel/xlabel
    plt.imshow(gray,cmap=cmap,norm=boundary_norm,aspect=aspect)
    cbar = plt.colorbar(location='left',pad=0.03,shrink=0.5)
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.set_title('nm',fontsize=15)
    
    ax2 = plt.subplot(gs[0:2,8:14])
    plt.axis('off')
    plt.rcParams['font.size'] = 18
    column_labels = ['Item', 'Unit', 'Value']
    column_colour = (0,173.0/255,1)
    column_colours = [column_colour,column_colour,column_colour]
    ax2.table(cellText=data, colLabels=column_labels,\
        colLoc='left',cellLoc='left',colColours=column_colours,\
        bbox = [0, -0.2, 1.0, 1.0]) # bbox = [left, bottom, with, height]
    
    plt.tight_layout()
    makedirs(dir,exist_ok=True)
    plt.savefig(dir +'Report_' + file_name + '.png')
    plt.close()
    return None
