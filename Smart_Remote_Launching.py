import os
import time
import math
import traceback
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from contextlib import contextmanager

from ZeroScan import zeroScanFX, get_xy_stage, _moveTo
from Optimized_Segmentation import OptimizedSegmentation
from SmartRemote import SmartRemote

@dataclass
class ScanParams:
    """Scan parameter"""
    x_range: float = 60.0
    y_range: float = 60.0
    x_iterations: int = 5
    y_iterations: int = 5
    approximate_resolution: int = 256
    precision_resolution: int = 256
    scan_speed: float = 0.5
    precision_speed: float = 1.0
    size_multiplier: float = 4.0
    score_threshold: float = 0.7

class StagePosition:
    x: float
    y: float
    z: float = 0.0

class ConnectionManager:
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sr = None
    
    def connect(self) -> bool:
        """Connect SmartRemote"""
        for attempt in range(self.max_retries):
            try:
                self.sr = SmartRemote()
                
                # Test connection
                connection = self.sr.check_connection()
                status = self.sr.check_status()
                
                if connection and status:
                    print(f"SmartRemote connected successfully")
                    return True
                else:
                    print(f"Connection test failed - Connection: {connection}, Status: {status}")
                    
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return False
    
    def is_connected(self) -> bool:
        try:
            return self.sr and self.sr.check_connection()
        except:
            return False

class StageController:
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn = connection_manager
        self.tolerance = 1.0 
    
    def get_position(self) -> Optional[StagePosition]:
        try:
            x, y = get_xy_stage()
            return StagePosition(x, y)
        except Exception as e:
            print(f"Error getting stage position: {e}")
            return None
    
    def move_to(self, target: StagePosition, max_attempts: int = 3) -> bool:
        for attempt in range(max_attempts):
            try:
                print(f"Moving to ({target.x:.2f}, {target.y:.2f}) - Attempt {attempt + 1}")
                _moveTo(target.x, target.y)
                
                time.sleep(1)
                current = self.get_position()
                
                if current and self._position_reached(current, target):
                    print(f"Stage moved successfully to ({current.x:.2f}, {current.y:.2f})")
                    return True
                else:
                    if current:
                        print(f"Position mismatch: target=({target.x:.2f}, {target.y:.2f}), "
                              f"actual=({current.x:.2f}, {current.y:.2f})")
                    
            except Exception as e:
                print(f"Move attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2)
        
        return False
    
    def _position_reached(self, current: StagePosition, target: StagePosition) -> bool:
        return (abs(current.x - target.x) < self.tolerance and 
                abs(current.y - target.y) < self.tolerance)

class ScanController:
    
    def __init__(self, base_data_dir: str = "D:\\SpmData\\JaeukSung\\Project\\Project\\Data"):
        self.base_data_dir = base_data_dir
        self.approximate_dir = os.path.join(base_data_dir, "Approximate")
        self.precision_dir = os.path.join(base_data_dir, "Precision")
        
        os.makedirs(self.approximate_dir, exist_ok=True)
        os.makedirs(self.precision_dir, exist_ok=True)
    
    def generate_session_id(self) -> str:
        return datetime.now().strftime("%y%m%d_%H%M%S")
    
    def perform_approximate_scan(self, params: ScanParams) -> Optional[str]:
        session_id = self.generate_session_id()
        scan_dir = os.path.join(self.approximate_dir, f"{session_id}_ZeroScan")
        
        try:
            print(f"Approximate scan: {params.x_range}x{params.y_range} μm")
            zeroScanFX(
                scan_dir, params.approximate_resolution, 0, 0, 
                params.approximate_resolution, params.x_range, 
                params.y_range, params.scan_speed
            )
            print(f"Scan completed: {scan_dir}")
            return scan_dir
            
        except Exception as e:
            print(f"Approximate scan failed: {e}")
            traceback.print_exc()
            return None
    
    def perform_precision_scan(self, offset_x: float, offset_y: float, 
                             scan_size: float, scan_index: int, 
                             params: ScanParams) -> bool:
        try:
            session_id = self.generate_session_id()
            precision_dir = os.path.join(self.precision_dir, 
                                       f"{session_id}_Precision_{scan_index}")
            
            print(f"Precision scan {scan_index}: "
                  f"offset=({offset_x:.2f}, {offset_y:.2f}), size={scan_size:.2f}")
            
            zeroScanFX(
                precision_dir, params.precision_resolution, 
                offset_x, offset_y, params.precision_resolution, 
                scan_size, scan_size, params.precision_speed
            )
            
            print(f"Precision scan {scan_index} completed")
            return True
            
        except Exception as e:
            print(f"Precision scan {scan_index} failed: {e}")
            return False

class SmartScanningSystem:
    
    def __init__(self, base_data_dir: str = "D:\\SpmData\\JaeukSung\\Project\\Project\\Data"):
        self.conn_manager = ConnectionManager()
        self.stage_controller = StageController(self.conn_manager)
        self.scan_controller = ScanController(base_data_dir)
        self.segmentation = OptimizedSegmentation()
        self.stats = {'total_scans': 0, 'successful_scans': 0, 'objects_found': 0}
    
    def initialize(self) -> bool:
        print("Initializing Smart Scanning System...")
        
        if not self.conn_manager.connect():
            print("Failed to connect to SmartRemote")
            return False
        
        try:
            self.segmentation.load_model()
            print("Segmentation model loaded")
            return True
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False
    
    def analyze_scan_data(self, scan_dir: str, params: ScanParams) -> List[List[float]]:
        try:
            print("Analyzing scan data...")
            center_list, _ = self.segmentation.segment_directory(
                scan_dir, params.score_threshold
            )
            
            if center_list:
                print(f"Found {len(center_list)} objects")
                self.stats['objects_found'] += len(center_list)
            else:
                print("No objects detected")
            
            return center_list
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            traceback.print_exc()
            return []
    
    def calculate_precision_params(self, detection: List[float], 
                                 original_range: Tuple[float, float],
                                 params: ScanParams) -> Tuple[float, float, float]:
        x_pos, y_pos, detected_size = detection[:3]
        x_range, y_range = original_range
        
        center_pixel = params.approximate_resolution / 2
        offset_x = (x_pos - center_pixel) / params.approximate_resolution * x_range
        offset_y = -(y_pos - center_pixel) / params.approximate_resolution * y_range
        
        normalized_size = detected_size / (params.approximate_resolution ** 2)
        calculated_size = math.sqrt(normalized_size * x_range * y_range)
        scan_size = params.size_multiplier + calculated_size
        
        return offset_x, offset_y, scan_size
    
    def scan_session(self, grid_pos: Tuple[int, int]):
        print(f"\n--- Grid position {grid_pos} ---")
        self.stats['total_scans'] += 1
        
        try:
            yield
            self.stats['successful_scans'] += 1
        except Exception as e:
            print(f"Scan session failed: {e}")
            traceback.print_exc()
    
    def run_grid_scan(self, params: ScanParams) -> bool:
        if not self.initialize():
            return False
        
        print(f"Starting grid scan: {params.x_iterations}x{params.y_iterations}")
        print(f"Each point: {params.x_range}x{params.y_range} μm")
        
        start_pos = self.stage_controller.get_position()
        if not start_pos:
            print("Failed to get starting position")
            return False
        
        print(f"Starting position: ({start_pos.x:.2f}, {start_pos.y:.2f})")
        
        for i in range(params.x_iterations):
            for j in range(params.y_iterations):
                with self.scan_session((i+1, j+1)):
                    if not self._process_grid_position(i, j, params, start_pos):
                        continue
                
                if j < params.y_iterations - 1:
                    current_pos = self.stage_controller.get_position()
                    if current_pos:
                        target = StagePosition(current_pos.x, current_pos.y + params.y_range)
                        self.stage_controller.move_to(target)
            
            if i < params.x_iterations - 1:
                target = StagePosition(
                    start_pos.x + (i + 1) * params.x_range, 
                    start_pos.y
                )
                if not self.stage_controller.move_to(target):
                    print("Failed to move to next column")
                    break
        
        self._print_summary()
        return self.stats['successful_scans'] > 0
    
    def _process_grid_position(self, i: int, j: int, params: ScanParams, 
                             start_pos: StagePosition) -> bool:
        try:
            current_pos = self.stage_controller.get_position()
            if current_pos:
                print(f"Current position: ({current_pos.x:.2f}, {current_pos.y:.2f})")
            
            scan_dir = self.scan_controller.perform_approximate_scan(params)
            if not scan_dir:
                return False
            
            detections = self.analyze_scan_data(scan_dir, params)
            if not detections:
                return True  
            
            precision_count = 0
            for k, detection in enumerate(detections):
                offset_x, offset_y, scan_size = self.calculate_precision_params(
                    detection, (params.x_range, params.y_range), params
                )
                
                if self.scan_controller.perform_precision_scan(
                    offset_x, offset_y, scan_size, k, params
                ):
                    precision_count += 1
            
            print(f"Precision scans: {precision_count}/{len(detections)}")
            return True
            
        except Exception as e:
            print(f"Error processing position ({i}, {j}): {e}")
            return False
    
    def _print_summary(self):
        print(f"\n--- Grid Scan Summary ---")
        print(f"Total positions: {self.stats['total_scans']}")
        print(f"Successful scans: {self.stats['successful_scans']}")
        print(f"Objects found: {self.stats['objects_found']}")
        
        if self.stats['total_scans'] > 0:
            success_rate = self.stats['successful_scans'] / self.stats['total_scans'] * 100
            print(f"Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    scan_params = ScanParams(
        x_range=60,
        y_range=60,
        x_iterations=5,
        y_iterations=5,
        approximate_resolution=256,
        precision_resolution=256,
        scan_speed=0.5,
        precision_speed=1.0,
        size_multiplier=4.0,
        score_threshold=0.7
    )
    
    scanner = SmartScanningSystem()
    success = scanner.run_grid_scan(scan_params)
    
    if success:
        print("Grid scanning completed successfully!")
    else:
        print("Grid scanning failed or completed with errors.")
