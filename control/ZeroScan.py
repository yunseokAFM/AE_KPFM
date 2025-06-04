from SmartRemote import SmartRemote as SR
sr = SR()

def zeroScanNX(baseDir='D:\SPMdata\성재욱\SmartRemote 1\Project\Data', pixelHeight:int=256, pixelWidth:int=256, scanRate:float=2.):
    _channles_init()
    _setScanGeometry(PH=pixelHeight, PW=pixelWidth, w=0, h=0)
    _setScanOption()
    _setScanRate(scanRate)
    _setZservo()
    _setXYservo()
    _setDataLocation(baseDir)
    _startImageScan()


def zeroScanFX(baseDir,pixelHeight:int=256, offset_x:float=2., offset_y:float=2.,target_X:float=2., target_Y:float=2.,pixelWidth:int=256, width:float=2.,height:float=2., scanRate:float=2.):
    #_autoAlign()
    #_channles_init()

    _setScanGeometry(PH=pixelHeight, PW=pixelWidth, w=width, h=height, oX=offset_x, oY=offset_y)
    _setScanOption()
    _moveTo(Tx = target_X, Ty = target_Y)
    _setScanRate(scanRate)
    _setZservo()
    #_setXYservo()
    _approach()
    #_pspdAlign()
    _setDataLocation(baseDir)
    _startImageScan()

def _autoAlign():
    reply = sr.run('spm.fx.analyzer.autoAlign()')
    return reply['result']

def _pspdAlign():
    reply = sr.run('spm.fx.analyzer.alignPspd()')
    return reply['result']

def _channles_init():
    mesg=(
        "chs = spm.scan.channels.names() \n\
        for (var i=0; i < chs.length; i++){ \n\
        print('Remove: '+chs[i]) \n\
        spm.scan.channels.remove(chs[i]) \n\
        } \n\
        spm.addChannel('ChannelZHeight')"
    )
    reply = sr.run(mesg)
    return reply['result']

def _setScanGeometry(PH:int=256, PW:int=128, w:int=20, h:int=20, oX:int=1, oY:int=0, ro:int=0):
    from json import dumps
    geometry = {"pixelHeight": PH, "pixelWidth": PW, "width": w, "height": h, "offsetX": oX, "offsetY": oY, "rotation": ro}
    json_geometry = dumps(geometry, indent=2)
    reply = sr.run(f'spm.scan.setScanGeometry({json_geometry})')
    return reply['result']


def _moveTo(Tx:float=1., Ty:float=1., Sx:float=1., Sy:float=1.):
    from json import dumps
    reply = sr.run(f'spm.xystage.moveTo({Tx},{Ty},{Sx},{Sy})')
    return reply['result']


def _setScanOption(sineScan:bool=False, overScan:bool=False, toWay:bool=False, detDriven:bool=False):
    from json import dumps
    option = {'biasReduction': {'lower': -1000, 'reduction': 0.8, 'upper':1000,'use':False},
              'detDriven':detDriven,
              'forceSlopeCorrection': True,
              'interlace': True,
              'overScan': {'enable':overScan,'percent':5},
              'sineScan': sineScan,
              'skipScan': {'applied':'scan2Only','height':0.2, 'rate':2,'skipped':'always'},
              'slowScan': 'end',
              'twoWay': toWay}
    json_option = dumps(option, indent=2)
    result = sr.run(f'spm.scan.options= {json_option}')
    try:
        assert result == "Done"
    except AssertionError as e:
        print(e)

def _setScanRate(rate:float=1.):
    reply = sr.run(f'spm.scan.rate = {rate}')
    return reply['result']

def _setXYservo(mode:str='off'):
    reply = sr.run(f'spm.xyservo.mode=\'{mode}\'')
    return reply['result']

def _setZservo(gain:list=[1,1,0.5,0.5], setpoint:float=1):
    from json import dumps
    gain_dict = {'p': gain[0], 'i': gain[1], 'z+':gain[2], 'z-':gain[3]}
    json_gain = dumps(gain_dict)
    reply = sr.run('spm.zservo.enable = true\nspm.zservo.setpoint.normalized = true')
    reply = sr.run(f'spm.zservo.gain = {json_gain}\nspm.zservo.setpoint.normValue = {setpoint}')
    return reply['result']

def _setDataLocation(baseDir:str='C:\\SpmData', file_name:str='ZeroScan'):
    from json import dumps
    loc = {'baseDir': baseDir, 'subDir': '%1_ZeroScan', 'cameraSave': False, 'direction': 'auto', 'fileName' : file_name, 'fileSuffix':' %1_%N_%G', 'jpegSave': False, 'precision':1}
    json_loc = dumps(loc)
    reply = sr.run(f'spm.dataLocation={json_loc}')
    return reply['result']

# Approach
def _approach(mode:str="q+s"):
    reply = sr.run(f'spm.approach.start(\'{mode}\')')
    return reply['result']

# Image Scan
def _startImageScan():
    reply = sr.run('spm.scan.startImageScan()')

def get_xy_stage():

    msg = 'var xy = spm.xystage.pos\n'
    msg += 'var info = xy.x + \',\' + xy.y \n'
    msg += 'info;'
    reply = sr.run(msg)
    value = reply['value'].split(',')
    return float(value[0]), float(value[1])
