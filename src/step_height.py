from src.core.afm_wrapper import AFM_Wrapper
from src.core.Checker import CHECKER
import src.core.analysis
import src.core.report
import src.core.tiff
from time import sleep

class Step_height:
    
    def __init__(self):
        self.base_dir = ''
        self.sub_dir = 'Step Height'
        self.file_name = 'RAW'
        self.__parameters = {
            'iteration':5,
            'gain': [1, 1, 1, 1],
            'setpoint':1,
            'xy_scanner_range': 100,
            'z_scanner_range': 70,
            'pixels': [256, 256],
            'scan_rate': 1,
            'scan_size': [10, 10],
            'xy_stage_move_to_enable': False,
            'xy_stage_move_to_x': 0.0,
            'xy_stage_move_to_y': 0.0,
            'xy_stage_move_to_lift_z': 3000.0           
        }
        self.afm = AFM_Wrapper()

    def set_params(self, params):
        self.__parameters = params

    def get_params(self):
        return self.__parameters

    def move_to(self):
        if self.__parameters['xy_stage_move_to_enable']:
            z = self.__parameters['xy_stage_move_to_lift_z']
            x = self.__parameters['xy_stage_move_to_x']
            y = self.__parameters['xy_stage_move_to_y']
            try:
                self.afm.lift_z_stage(z)
            except:
                print('Lift Fail')
                return None
            self.afm.moveto_xy_stage(x,y,1,1)
        else:
            pass

    def set(self):
        self.afm.set_data_location(self.base_dir, self.sub_dir, self.file_name)
        self.afm.set_head_mode('ncm')
        self.afm.clear_channels()
        self.afm.add_channel('ChannelNcmAmplitude')
        self.afm.add_channel('ChannelNcmPhase')
        self.afm.add_channel('ChannelErrorSignal')
        self.afm.add_channel('ChannelZDriveOrTopography')
        self.afm.stop_scan()
     
        self.afm.set_scan_geometry(self.__parameters['pixels'][0],self.__parameters['pixels'][1],
                                   self.__parameters['scan_size'][0], self.__parameters['scan_size'][0],
                                   )
        self.afm.set_scan_option(over_scan_percent=5)
        self.afm.set_scan_rate(self.__parameters['scan_rate'])
        self.afm.enable_xy_servo()
        self.afm.enable_z_servo()
        self.afm.set_z_servo(self.__parameters['gain'], self.__parameters['setpoint'])
        self.afm.ncm_sweep_auto_full_range()
        self.afm.set_approach_option()
    
    def unset(self):    
        self.afm.set_scan_option_dict(self.bck_scan_option)
        self.afm.set_scan_geometry_dict(self.bck_scan_geometry)
        self.afm.set_scan_option_dict(self.bck_scan_option)
    
    def approach(self):
        self.afm.start_approach('q+s')
        self.afm.lift_z_stage(10)
        self.afm.ncm_sweep_auto_full_range()
        self.afm.start_approach('q+s')
 
    def run(self):
        self.afm.trigger_image_scan()
        isscan = True
        while isscan:
            test = self.afm.query_scan_status()
            if test == 'true':isscan=True;
            else:isscan=False
            sleep(2)
    
    def done(self):
        self.afm.stop_scan()
        self.afm.stop_scan()
        self.afm.lift_z_stage(100)

    def stop(self):
        checker = CHECKER()
        checker.abort_approach()
        checker.abort_scan()

    def analyze(self):
        find_dir = self.base_dir + '\\' + self.sub_dir 
        tiff_list = src.core.report.ls_tiff(find_dir)
        read_tiff = src.core.tiff.Read_tiff()
        write_tiff = src.core.tiff.Write_tiff()
            
        result_mean = []
        result_min = []
        result_max = []
        result_name = []
        result_range = []
        for tiff_path in tiff_list:
            tiff_name = tiff_path.split('\\')[-1]
            tmp_name = tiff_name.split('.')[0]
            condition_1 = tmp_name.split('_')[-3] == 'Z Height'
            condition_2 = tmp_name.split('_')[-2] == 'Forward'
            tmp_name_list = tiff_name.split('_')
            tmp_name = ''
            for s in tmp_name_list[1:]: tmp_name += '_' + s
            tmp_name = tmp_name.split('.')[0]
            tmp_name = tmp_name[2:]
            if condition_1 & condition_2:
                dict_tiff , ori_tiff = read_tiff(tiff_path)
                image = dict_tiff['IMAGE']
                cmap = dict_tiff['COLORMAP']
                flatten_image = src.core.analysis.grating_flatten(image)
                tmp_range,_,_ = src.core.analysis.cal_histogram_range(flatten_image)
                # convert scale : um ->nm
                scale_factor = 1E+3
                tmp_mean, tmp_min, tmp_max = src.core.analysis.cal_statistic(flatten_image)
                tmp_mean *= scale_factor; tmp_min *= scale_factor; tmp_max *= scale_factor
                dict_tiff['IMAGE'] = flatten_image
                save_path = self.base_dir + '\\' +self.sub_dir + '\\Flatten\\' + 'Flatten_' + tmp_name + '.tiff'
                report_dir = self.base_dir + '\\' +self.sub_dir + '\\Report\\'
                write_tiff(ori_tiff,dict_tiff,save_path)
                tmp_data = [
                            ['Mean', 'nm',f'{tmp_mean:.3f}'],
                            ['Min', 'nm', f'{tmp_min:.3f}'],
                            ['Max', 'nm', f'{tmp_max:.3f}'],
                            ['Range', 'nm', f'{tmp_range:.3f}']
                            ]
                flatten_image *= scale_factor
                src.core.report.default_report(tmp_name, report_dir, dict_tiff, cmap, tmp_data)

                result_name.append(tmp_name)
                result_mean.append(tmp_mean)
                result_min.append(tmp_min)
                result_max.append(tmp_max)
                result_range.append(tmp_range)
        
        csv_path = self.base_dir + '\\' + self.sub_dir + "\\result.csv"
        result_dict = {'File name': result_name,
                       'Mean (nm)': result_mean,
                       'Min (nm)' : result_min,
                       'Max (nm)' : result_max,
                       'Range (nm)': result_range
                      } 
        src.core.report.write_csv(result_dict,csv_path,['Mean (nm)','Min (nm)','Max (nm)', 'Range (nm)'])
