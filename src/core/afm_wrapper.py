from json import dumps

from .SmartRemote import SmartRemote

class AFM_Wrapper:

    def __init__(self):
        self.__sr = SmartRemote()
        
    def clear_channels(self):
        """ Clearing all channels"""
        mesg=( 
            "chs = spm.scan.channels.names() \n\
            for (var i=0; i < chs.length; i++){ \n\
            print('Remove: '+chs[i]) \n\
            spm.scan.channels.remove(chs[i]) \n\
            }"
            )
        reply = self.__sr.run(mesg)
        return reply['result']

    def add_channel_unit(self, ch:str='ChannelZHeight',unit:str='um'):
        """Add channles with unit
        
        Args:
            ch (str, optional): channel name. Defaults to 'ChannelZHeight.
            unit (str, optional): channel unit. Defaults to 'um'
        """
        channel = {'name': ch, 'unit':unit}
        json_channel = dumps(channel,indent=2)      
        reply = self.__sr.run(f"spm.scan.channels.add({json_channel})")
        return reply['result']

    def add_channel(self, ch:str='ChannelZHeight'):
        """Add channles
        
        Args:
            ch (str, optional): channel name. Defaults to 'ChannelZHeight.
        """
        reply = self.__sr.run(f"spm.scan.channels.add(\'{ch}\')")
        return reply['result']

    def remove_channel(self,ch:str='ChannelZHeight'):
        """Delete channles
        
        Args:
            ch (str, optional): channel name. Defaults to 'ChannelZHeight.
        
        """
        reply = self.__sr.run(f'spm.scan.channels.remove(\'{ch}\')')
        return reply['result']

    def set_head_mode(self, mode:str='contact'):
        """Set head mode

        Args:
            mode (str, mode): contact, ncm, tapping. Defaults to 'contact'.
        """
        reply = self.__sr.run(f'spm.head.setMode(\'{mode}\')')
        return reply['result']

    def set_scan_geometry(self, pixel_width:int=256, pixel_height:int=256, \
                          width:int=20, height:int=20,\
                          offset_x:int=0, offset_y:int=0, rotation:int=0): 
        """Set Scan geometry"""
        geometry = {"pixelHeight": pixel_height,
                    "pixelWidth": pixel_width,
                    "width": width,
                    "height": height,
                    "offsetX": offset_x,
                    "offsetY": offset_y,
                    "rotation": rotation}
        json_geometry = dumps(geometry,indent=2)
        reply = self.__sr.run(f'spm.scan.setScanGeometry({json_geometry})')
        return reply['result']

    def set_scan_option(self, sine_scan:bool=False,over_scan:bool=True, \
                        over_scan_percent:int=5, tow_way:bool=True, \
                        det_driven:bool=False, force_slope_correction:bool=False, \
                        interlace:bool=False,slow_scan:str='end',
                        bias_reduction_lower:float=-1000,
                        bias_reduction_reduction:float=0.8,
                        bias_reduction_upper:float=1000,
                        bias_reduction_use:bool=False):
        """Set Scan option"""
        option = {'biasReduction': {'lower': bias_reduction_lower,
                                    'reduction': bias_reduction_reduction,
                                    'upper':bias_reduction_upper,
                                    'use':bias_reduction_use},
                'detDriven':det_driven,
                'forceSlopeCorrection': force_slope_correction,
                'interlace': interlace,
                'overScan': {'enable':over_scan,'percent':over_scan_percent},
                'sineScan': sine_scan,
                'skipScan': {'applied':'scan2Only','height':0.2, 'rate':2,'skipped':'always'},
                'slowScan': slow_scan,
                'twoWay': tow_way}
        print(option)
        json_option = dumps(option,indent=2)
        reply = self.__sr.run(f'spm.scan.options= {json_option}')
        return reply['result']

    def set_scan_rate(self, rate:float=1.):
        """Set scan Rate

        Args:
            rate (float, optional): Hz. Defaults to 1..

        Returns:
            str: Done
        """
        reply = self.__sr.run(f'spm.scan.rate = {rate}')
        return reply['result']

    def enable_xy_servo(self, mode:str='on'):
        """ XY servo on/off

        Args:
            mode (str, optional): on/off. Defaults to 'on'.
        """
        reply = self.__sr.run(f'spm.xyservo.mode=\'{mode}\'')
        return reply['result']

    def enable_z_servo(self, mode:str='true'):
        assert mode in ['true', 'false']
        reply = self.__sr.run(f'spm.zservo.enable = {mode}')
        return reply['result']

    def set_z_servo(self, gain:list=[1,1,0.5,0.5], setpoint:float=1):
        """Set Z servo parameter

        Args:
            gain (list, optional): _description_. Defaults to [1,1,0.5,0.5].
            setpoint (float, optional): _description_. Defaults to 1.
        """
        gain_dict = {'z+': gain[0],
                    'z-': gain[1],
                    'p':gain[2],
                    'i':gain[3]}
        json_gain = dumps(gain_dict)
        reply = self.__sr.run('spm.zservo.enable = true\n\
                     spm.zservo.setpoint.normalized = true')
        reply = self.__sr.run(f'spm.zservo.gain = {json_gain}\n\
                     spm.zservo.setpoint.normValue = {setpoint}')
        return reply['result']
    
    def get_normalized_zservo_setpoint(self):
        """Set Z servo set point (nm)"""
        reply = self.__sr.run('spm.zservo.nomalized = false')
        reply = self.__sr.run(f'spm.zservo.setpoint.value')
        return reply['value']
        
    def set_normalized_zservo_setpoint(self, setpoint:float):
        """Set Z servo position (nm)"""
        reply = self.__sr.run('spm.zservo.enable = true\n\
                     spm.zservo.setpoint.normalized = false')
        reply = self.__sr.run(f'spm.zservo.setpoint.value = {setpoint}')
        return reply['result']

    def set_data_location(self, base_dir:str='C:\\SpmData', sub_dir:str='zeroscan' , file_name:str=''):
        """Set data Location

        Args:
            base_dir (_type_, optional): basedir path. Defaults to 'C:\SpmData'.
            file_name (str, optional): file name. Defaults to 'ZeroScan'.
        """
        loc = {'baseDir': base_dir,
            'subDir': f'{sub_dir}',
            'cameraSave': False,
            'direction': 'auto',
            'fileName' : file_name,
            'fileSuffix':' %1_%N_%G',
            'jpegSave': False,
            'precision':1,
            }
        json_loc = dumps(loc)
        reply = self.__sr.run(f'spm.dataLocation={json_loc}')
        return reply['result']

    def set_approach_option(self, fast_speed:float=600.0, quick_speed:float=100.0, \
                            slow_speed:float = 10.0, incremental_speed:float=10, \
                            fast_error_threshold:int=97, error_threshold:int=95, \
                            target_pos:float=0.0, focus_on_cantilever:bool=True):
        """This is an aggregation of approach option parameters. 
        It includes quick speed, slow speed, incremental speed, error threshold, target position and approach type.

        Args:
            fast_speed (float, optional): fast approach speed (Micormeter/second). Defaluts to 600.0
            quick_speed (float, optional): quick approach speed (Micrometer/second). Defaults to 100.0.
            slow_speed (float, optional): slow approach speed (Micrometer/second). Defaults to 100.0.
            incremental_speed (float, optional): incremental approach speed (Micrometer/second). Defaults to 10.0
            target_pos (float, optional): Z position target (Micrometer). Defaults to 0.0.
        """
        option = {'fastSpeed' : fast_speed,
                  'quickSpeed': quick_speed,
                  'slowSpeed' : slow_speed,
                  'incrementalSpeed': incremental_speed,
                  'fastErrorThreshold' : fast_error_threshold,
                  'errorThreshold' : error_threshold,
                  'targetPos': target_pos,
                  'focusOnCantilever': focus_on_cantilever
                  }
        json_option = dumps(option)
        reply = self.__sr.run(f'spm.approach.setOption({json_option})')

    def start_approach(self, mode:str='q+s'):
        """Start approach
        
        Args:
            mode (str, optional): "quick", "q" for quick approach
                                  "quickAndsafe", "q+s" for quick-and-safe approach
                                  "incremental", "inc" for incremental approach
                                  "incrementalzscanner", "incz" for incremental-zscanner approach .
                                  Defaults to "q+s".
        """
        reply = self.__sr.run(f'spm.approach.start(\'{mode}\')')
        return reply['result']
    

    def query_scan_status(self):
        return self.__sr.query_scan_status()

    def status_approach(self):
        reply = self.__sr.run('spm.approach.state')
        return reply['result']
    
    def stop_approach(self):
        reply = self.__sr.run('spm.approach.stop()')
        return reply['result']
   
    def start_image_scan(self):
        reply = self.__sr.run('spm.scan.startImageScan()')
        return reply['result']
    
    def stop_scan(self):
        reply = self.__sr.run('spm.scan.stop()')
        return reply['result']

    def trigger_image_scan(self):
        """Not waiting until the scan is finished."""
        reply = self.__sr.run('spm.scan.triggerImageScan()')
        return reply['result']

    def trigger_line_scan(self):
        """ Start Line Scan"""
        reply = self.__sr.run('spm.scan.triggerLineScan()')
        return reply['result']

    ## Positioning
    def moveto_z_scanner(self,height:float):
        """Move To Z Scanner"""
        reply =self.__sr.run(f'spm.zscanner.moveTo({height})')

    def moveto_xy_stage(self, target_x:float, target_y:float,\
                        norm_speed_x:float, norm_speed_y:float):
        reply = self.__sr.run(f'spm.xystage.moveTo({target_x}, {target_y}, {norm_speed_x}, {norm_speed_y})')
        return reply['result']

    def moveto_z_stage(self,target:float,norm_speed:float):
        reply = self.__sr.run(f'spm.zstage.moveTo({target}, {norm_speed})')
        return reply['result']
    
    def moveto_focus_stage(self,target:float,norm_speed:float):
        reply = self.__sr.run(f'spm.focusstage.moveTo({target}, {norm_speed})')
        return reply['result']

    def lift_z_stage(self,dist:float):
        """ Lift Z stage

        Args:
            dist (float): Micrometer

        Raises:
            ValueError: Negative Number
        """
        try:
            if dist < 0: raise ValueError('ERROR: Negative number')
        except ValueError as e:
            print(e)
            return False
        reply = self.__sr.run(f'spm.zstage.move({dist},1)')
        return reply['result']

    def reset_xy_stage(self):
        """reset XY stage"""
        reply = self.__sr.run('spm.xystage.reset()')
        return reply['result']

    def reset_z_stage(self):
        """reset Z stage"""
        reply = self.__sr.run('spm.zstage.reset()')
        return reply['result']
    
    def reset_focus_stage(self):
        """Reset Focus stage"""
        reply = self.__sr.run('spm.focusstage.reset()')
        return reply['result']

    def ncm_sweep_auto(self, target_amp:float=25.0, start_freq:float=10.0, \
                        end_freq:float=1000, init_drive:float=9.0):
        """Starts ncm sweep with auto options

        Args:
            target_amp (float): Target amplitude (nm)
            start_freq (float): The start value of the initial frequency range (Hz)
            end_freq (float): The last value of the initial frequency range (Hz)
            init_drive (float): The drive strength (%)
        """
        reply = self.__sr.run(f'spm.ncm.sweepAuto({target_amp}, {start_freq}, {end_freq}, {init_drive})')
        return reply['result']
    
    def ncm_sweep_auto_full_range(self):
        """Starts ncm sweep on full_range """
        reply = self.__sr.run('spm.ncm.sweepAutoFullRange()')
        return reply['result']

    def retract_z_scanner(self):
        reply = self.__sr.run('spm.zscanner.retractAll()')
        return reply['result']
    
    def get_z_scanner_pos(self):
        reply = self.__sr.run('spm.zscanner.pos')
        pos = float(reply['value'])
        return pos

    def read_ncm_amp(self):
        self.add_channle('ncmamplitude')
        amp = self.__sr.run('spm.readChannel(\'ncmamplitude\').value')['value']
        unit = self.__sr.run('spm.readChannel(\'ncmamplitude\').unit')['value']
        raw_amp = self.__sr.run('spm.readChannel(\'ncmamplitude\').rawValue')['value']
        return float(amp), unit, float(raw_amp)

    def read_ncm_avg_amp(self,count:int, interval:int):
        # Need fix: Check return value
        self.add_channles('ncmamplitude')
        reply = self.__sr.run(f'spm.readChannelAveraged(\'ncmamplitude\',{count}, {interval})')
        return reply['result']

    def set_z_scanner_range(self,percent:int=70):
        reply = self.__sr.run(f'spm.zscanner.changeRangeTo({percent})')
        return reply['result']

    def set_xy_scanner_range(self,percent:int=100):
        reply = self.__sr.run(f'spm.xyscanner.changeRangeTo({percent})')
        return reply['result']
    
    def get_focus_stage_position(self):
        reply = self.__sr.run('spm.focusstage.pos')
        return reply['value']
    