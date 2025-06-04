import os
import shutil
import time
import math
import pandas as pd
from glob import glob
from datetime import datetime
from typing import List, Tuple, Optional

from control.ZeroScan_v2 import zeroScanFX, get_xy_stage, _moveTo
from Segmentation import segmentation
from SmartRemote import SmartRemote as SR

class SmartScanningSystem:
    def __init__(self, base_data_dir: str = "D:\\SpmData\\JaeukSung\\Project\\Project\\Data"):

        self.base_data_dir = base_data_dir
        self.approximate_dir = os.path.join(base_data_dir, "Approximate")
        self.precision_dir = os.path.join(base_data_dir, "Precision")
        

        os.makedirs(self.approximate_dir, exist_ok=True)
        os.makedirs(self.precision_dir, exist_ok=True)
        

        self.sr = SR()
        self.check_connections()
    
    def check_connections(self) -> bool:
       
        try:
            connection = self.sr.check_connection()
            print(f'SmartRemote connection: {connection}')
            
            
            status = self.sr.check_status()  
            print(f'PowerScript status: {status}')
            
            return connection and status
        except Exception as e:
            print(f"Connection check failed: {e}")
            return False
    
    def generate_scan_session_id(self) -> str:
        
        return datetime.now().strftime("%y%m%d_%H%M%S")
    
    def perform_approximate_scan(self, x_range: float, y_range: float, 
                               scan_resolution: int = 256, scan_speed: float = 0.5) -> str:

        session_id = self.generate_scan_session_id()
        scan_dir = os.path.join(self.approximate_dir, f"{session_id}_ZeroScan")
        
        try:
            print(f"Starting approximate scan: {x_range}x{y_range} μm")
            zeroScanFX(scan_dir, scan_resolution, 0, 0, scan_resolution, 
                      x_range, y_range, scan_speed)
            print(f"Approximate scan completed: {scan_dir}")
            return scan_dir
            
        except Exception as e:
            print(f"Error during approximate scan: {e}")
            return ""
    
    def analyze_scan_data(self, scan_dir: str) -> List[List[float]]:

        try:
            print("Analyzing scan data for segmentation...")
            center_list, mask_sizes = segmentation(scan_dir)
            
            if not center_list:
                print("No objects detected in the scan")
                return []
            
            print(f"Found {len(center_list)} objects for precision scanning")
            return center_list
            
        except Exception as e:
            print(f"Error during scan analysis: {e}")
            return []
    
    def calculate_precision_scan_params(self, detection_result: List[float], 
                                      original_range: Tuple[float, float],
                                      scan_resolution: int = 256,
                                      size_multiplier: float = 4.0) -> Tuple[float, float, float]:

        x_pos, y_pos, detected_size = detection_result[:3]
        x_range, y_range = original_range
        
        
        center_pixel = scan_resolution / 2
        offset_x = (x_pos - center_pixel) / scan_resolution * x_range
        offset_y = -(y_pos - center_pixel) / scan_resolution * y_range  
        
        
        normalized_size = detected_size / (scan_resolution ** 2)
        calculated_size = math.sqrt(normalized_size * x_range * y_range)
        scan_size = size_multiplier + calculated_size
        
        return offset_x, offset_y, scan_size
    
    def perform_precision_scan(self, offset_x: float, offset_y: float, 
                             scan_size: float, scan_index: int,
                             scan_resolution: int = 256, scan_speed: float = 1.0) -> bool:

        try:
            session_id = self.generate_scan_session_id()
            precision_scan_dir = os.path.join(self.precision_dir, 
                                            f"{session_id}_Precision_{scan_index}")
            
            print(f"Precision scan {scan_index}: offset=({offset_x:.2f}, {offset_y:.2f}), size={scan_size:.2f}")
            
            zeroScanFX(precision_scan_dir, scan_resolution, offset_x, offset_y, 
                      scan_resolution, scan_size, scan_size, scan_speed)
            
            print(f"Precision scan {scan_index} completed")
            return True
            
        except Exception as e:
            print(f"Error during precision scan {scan_index}: {e}")
            return False
    
    def move_stage_safely(self, target_x: float, target_y: float, 
                         max_attempts: int = 3) -> bool:

        for attempt in range(max_attempts):
            try:
                print(f"Moving stage to ({target_x:.2f}, {target_y:.2f}) - Attempt {attempt + 1}")
                _moveTo(target_x, target_y)
                
                
                time.sleep(1)
                current_x, current_y = get_xy_stage()
                
                
                if abs(current_x - target_x) < 1.0 and abs(current_y - target_y) < 1.0:
                    print(f"Stage moved successfully to ({current_x:.2f}, {current_y:.2f})")
                    return True
                else:
                    print(f"Stage position mismatch: target=({target_x:.2f}, {target_y:.2f}), "
                          f"actual=({current_x:.2f}, {current_y:.2f})")
                    
            except Exception as e:
                print(f"Stage movement attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)  
        
        return False
    
    def run_grid_scan(self, x_range: float = 60, y_range: float = 60,
                     x_iterations: int = 5, y_iterations: int = 5,
                     approximate_resolution: int = 256,
                     precision_resolution: int = 256) -> bool:

        if not self.check_connections():
            print("Cannot proceed: SmartRemote connection failed")
            return False
        
        print(f"Starting grid scan: {x_iterations}x{y_iterations} grid, "
              f"each point {x_range}x{y_range} μm")
        
        
        try:
            start_x, start_y = get_xy_stage()
            print(f"Starting position: ({start_x:.2f}, {start_y:.2f})")
        except Exception as e:
            print(f"Error getting initial position: {e}")
            return False
        
        total_scans = 0
        successful_scans = 0
        
        for i in range(x_iterations):
            for j in range(y_iterations):
                try:
                    print(f"\n--- Grid position ({i+1}/{x_iterations}, {j+1}/{y_iterations}) ---")
                    
                    
                    current_x, current_y = get_xy_stage()
                    print(f"Current stage position: ({current_x:.2f}, {current_y:.2f})")
                    
                    
                    scan_dir = self.perform_approximate_scan(x_range, y_range, approximate_resolution)
                    if not scan_dir:
                        print("Approximate scan failed, skipping this position")
                        continue
                    
                    
                    detections = self.analyze_scan_data(scan_dir)
                    if not detections:
                        print("No objects detected, moving to next position")
                        successful_scans += 1  
                        total_scans += 1
                        continue
                    
                    
                    precision_success_count = 0
                    for k, detection in enumerate(detections):
                        offset_x, offset_y, scan_size = self.calculate_precision_scan_params(
                            detection, (x_range, y_range), approximate_resolution
                        )
                        
                        if self.perform_precision_scan(offset_x, offset_y, scan_size, k, 
                                                     precision_resolution):
                            precision_success_count += 1
                    
                    print(f"Precision scans completed: {precision_success_count}/{len(detections)}")
                    successful_scans += 1
                    total_scans += 1
                    
                    
                    if j < y_iterations - 1:  
                        target_y = current_y + y_range
                        if not self.move_stage_safely(current_x, target_y):
                            print("Failed to move to next Y position")
                            break
                    
                except Exception as e:
                    print(f"Error at grid position ({i}, {j}): {e}")
                    total_scans += 1
                    continue
            
            
            if i < x_iterations - 1:
                try:
                    current_x, current_y = get_xy_stage()
                    target_x = start_x + (i + 1) * x_range
                    target_y = start_y  
                    
                    if not self.move_stage_safely(target_x, target_y):
                        print("Failed to move to next X position")
                        break
                        
                except Exception as e:
                    print(f"Error moving to next column: {e}")
                    break
        
        
        print(f"\n--- Grid Scan Completed ---")
        print(f"Total positions: {total_scans}")
        print(f"Successful scans: {successful_scans}")
        print(f"Success rate: {successful_scans/total_scans*100:.1f}%" if total_scans > 0 else "N/A")
        
        return successful_scans > 0


if __name__ == "__main__":
    
    scanner = SmartScanningSystem()
    
    
    scan_params = {
        'x_range': 60,          # μm
        'y_range': 60,          # μm
        'x_iterations': 5,      # 5x5 
        'y_iterations': 5,
        'approximate_resolution': 256,
        'precision_resolution': 256
    }
    

    success = scanner.run_grid_scan(**scan_params)
    
    if success:
        print("Grid scanning completed successfully!")
    else:
        print("Grid scanning failed or completed with errors.")
