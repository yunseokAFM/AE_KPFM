import os
import numpy as np
from pspylib.tiff.reader import TiffReader

def _cvt_uint16(value):
    word = ""
    for alphabet in value[0]:
        if alphabet != 0:
            word += chr(alphabet)
        else:
            pass
    return word

def header_dict(tiff_path):
    result_dict = {}
    tiff = TiffReader(tiff_path)
    Header = tiff.data.scanHeader.scanHeader
    for i in Header.items():
        if i[-1][-1] == 'uint16':
            cvt_data = _cvt_uint16(i[-1])
            result_dict[i[0]] = cvt_data
        else:
            result_dict[i[0]] = i[-1][0]
    return result_dict

def spectHeader_dict(tiff_path):
    result_dict = {}
    tiff = TiffReader(tiff_path)
    SpectHeader = tiff.data.spectHeader.spectHeader
    for i in SpectHeader.items():
        if i[-1][-1] == 'uint16':
            cvt_data = _cvt_uint16(i[-1])
            result_dict[i[0]] = cvt_data
        else:
            result_dict[i[0]] = i[-1][0]
    return result_dict

def tiff2array(tiff_path, save_file=False, save_path='None'):
    tiff = TiffReader(tiff_path)
    Zdata = tiff.data.scanData.ZData
    header = tiff.data.scanHeader.scanHeader
    dshape = (int(header['height'][0]), int(header['width'][0]))
    npy = np.reshape(Zdata, dshape)
    npy = np.flipud(npy)
    if save_file and (save_path == 'None'):
        file_path_src = os.path.splitext(tiff_path)
        file_name = file_path_src[-1].split('.')[0] + '.npy'
        file_path = file_path_src[0] + file_name
        print('SAVE: ', file_path)
        np.save(file_path, npy)
    elif save_file:
        np.save(save_path, npy)
    else:
        return npy

def tiff_info(tiff_path):
    dict = {}
    tiff = TiffReader(tiff_path)
    header_d = header_dict(tiff_path)
    spectHeader_d = spectHeader_dict(tiff_path)
    # spectChInfo_d = spectChInfo_dict(tiff_path)
    image_a = tiff2array(tiff_path)
    dict['HEADER'] = header_d
    dict['SPECT_HEADER'] = spectHeader_d
    # dict['SPECT_CH_INFO'] = spectChInfo_d
    dict['IMAGE'] = image_a
    return dict, tiff

