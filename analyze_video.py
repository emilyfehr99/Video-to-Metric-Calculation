#!/usr/bin/env python3
"""
Hockey Metrics Tool - Main Analysis Script
Standalone tool for analyzing hockey videos and generating advanced metrics.
"""

import cv2
import numpy as np
import json
import time
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from metrics_tracker import HockeyMetricsTracker


class RoboflowDetector:
    """Simple Roboflow detection interface for the standalone tool."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Roboflow detector."""
        self.config = config
        self.roboflow_config = config.get('roboflow', {})
        
        # Try to import inference_sdk
        try:
            from inference_sdk import InferenceHTTPClient
            
            # Initialize inference client
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.roboflow_config.get('api_key')
            )
            self.workspace = self.roboflow_config.get('workspace_name')
            self.workflow_id = self.roboflow_config.get('workflow_id')
            print(f"✅ Roboflow initialized - Workspace: {self.workspace}, Workflow: {self.workflow_id}")
        except ImportError:
            print("❌ inference_sdk not available - using mock detection for demo")
            self.client = None
    
    def detect_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Detect players and puck in a frame."""
        if self.client is None:
            # Mock detection for demo purposes
            return self._mock_detection(frame, frame_id)
        
        try:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, frame) # Changed from tmp_path to frame
                temp_path = tmp_file.name
            
            try:
                # Run Roboflow detection using the correct method
                result = self.client.run_workflow(
                    workspace_name=self.workspace,
                    workflow_id=self.workflow_id,
                    images={"image": temp_path},
                    use_cache=True
                )
                
                # Process results
                detections = self._process_roboflow_result(result, frame)
                return detections
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"❌ Roboflow detection failed: {e}")
            return self._mock_detection(frame, frame_id)
    
    def _mock_detection(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """Generate mock detections for demo purposes."""
        height, width = frame.shape[:2]
        
        # Simulate some detections
        players = []
        puck_detections = []
        
        # Add some mock players
        if frame_id % 30 < 15:  # Every 0.5 seconds
            players.append({
                'player_id': f'player_{frame_id // 30}',
                'rink_position': {'x': width * 0.3, 'y': height * 0.4},
                'confidence': 0.9,
                'class': 'player'
            })
            players.append({
                'player_id': f'player_{frame_id // 30 + 1}',
                'rink_position': {'x': width * 0.7, 'y': height * 0.6},
                'confidence': 0.8,
                'class': 'player'
            })
        
        # Add mock puck
        puck_x = width * 0.5 + (frame_id % 60 - 30) * 0.01  # Moving puck
        puck_y = height * 0.5 + (frame_id % 60 - 30) * 0.01
        puck_detections.append({
            'rink_position': {'x': puck_x, 'y': puck_y},
            'confidence': 0.85,
            'class': 'puck'
        })
        
        return {
            'players': players,
            'puck_detections': puck_detections
        }
    
    def _process_roboflow_result(self, result: Any, frame: np.ndarray) -> Dict[str, Any]:
        """Process Roboflow detection results using all available classes."""
        # Process Roboflow result silently
        
        if not result or not isinstance(result, list) or len(result) == 0:
            # No Roboflow result, using mock detection
            return self._mock_detection(frame, 0)
        
        # Extract predictions from Roboflow response
        
        # Handle different Roboflow response formats
        if isinstance(result, list) and len(result) > 0:
            if 'predictions' in result[0]:
                # Standard format
                predictions = result[0]['predictions']
            elif 'label_visualization' in result[0] and 'predictions' in result[0]['label_visualization']:
                # New format with label_visualization
                predictions = result[0]['label_visualization']['predictions']
            else:
                # Try to find predictions in the result structure
                predictions = []
                for item in result:
                    if isinstance(item, dict):
                        if 'predictions' in item:
                            predictions = item['predictions']
                            break
                        elif 'label_visualization' in item and 'predictions' in item['label_visualization']:
                            predictions = item['label_visualization']['predictions']
                            break
        else:
            predictions = []
        
        # Predictions processed silently
        
        # Get frame dimensions for coordinate conversion
        frame_height, frame_width = frame.shape[:2]
        
        # Categorize detections by class
        players = []
        puck_detections = []
        field_detection = None
        goal_zones = []
        center_circles = []
        red_circles = []
        blue_lines = []
        center_line = None
        
        for pred in predictions:
            class_name = pred.get('class', '').lower()
            confidence = pred.get('confidence', 0.0)
            
                    # Process detections silently
            
            # Skip low confidence detections
            if confidence < 0.5:
                continue
            
            detection_data = {
                'x': pred.get('x', 0),
                'y': pred.get('y', 0),
                'width': pred.get('width', 0),
                'height': pred.get('height', 0),
                'confidence': confidence,
                'class': class_name
            }
            
            # Categorize by class
            if class_name in ['player', 'home', 'away']:
                # Map home/away to player for processing
                player_id = f"player_{len(players)}"
                players.append({
                    'player_id': player_id,
                    'rink_position': {'x': detection_data['x'], 'y': detection_data['y']},
                    'confidence': confidence,
                    'class': 'player',
                    'team': class_name if class_name in ['home', 'away'] else 'unknown'
                })
            
            elif class_name == 'puck':
                puck_detections.append({
                    'rink_position': {'x': detection_data['x'], 'y': detection_data['y']},
                    'confidence': confidence,
                    'class': 'puck'
                })
                # Puck detected silently
            
            elif class_name == 'field':
                field_detection = detection_data
                # Field detected silently
            
            elif class_name == 'goalzone':
                goal_zones.append(detection_data)
            
            elif class_name == 'center__circle':  # Note: double underscore
                center_circles.append(detection_data)
            
            elif class_name == 'red_circle':
                red_circles.append(detection_data)
            
            elif class_name == 'blue_line':
                blue_lines.append(detection_data)
            
            elif class_name == 'center_line':
                center_line = detection_data
        
        # Convert coordinates to rink space (normalized 0-1, then scaled to rink dimensions)
        rink_width, rink_height = 1400, 600  # Default rink dimensions
        
        print(f"🔄 Converting coordinates: frame={frame_width}x{frame_height} -> rink={rink_width}x{rink_height}")
        
        # Convert player and puck coordinates to rink space
        for player in players:
            if 'rink_position' in player:
                # Convert from video coordinates to rink coordinates
                video_x = player['rink_position']['x']
                video_y = player['rink_position']['y']
                
                # Normalize to 0-1 range
                norm_x = video_x / frame_width
                norm_y = video_y / frame_height
                
                # Scale to rink dimensions
                rink_x = norm_x * rink_width
                rink_y = norm_y * rink_height
                
                player['rink_position'] = {'x': rink_x, 'y': rink_y}
                print(f"👤 Player: {video_x:.1f},{video_y:.1f} -> {rink_x:.1f},{rink_y:.1f}")
        
        for puck in puck_detections:
            if 'rink_position' in puck:
                # Convert from video coordinates to rink coordinates
                video_x = puck['rink_position']['x']
                video_y = puck['rink_position']['y']
                
                # Normalize to 0-1 range
                norm_x = video_x / frame_width
                norm_y = video_y / frame_height
                
                # Scale to rink dimensions
                rink_x = norm_x * rink_width
                rink_y = norm_y * rink_height
                
                puck['rink_position'] = {'x': rink_x, 'y': rink_y}
                # Puck coordinates converted silently
        
        return {
            'players': players,
            'puck_detections': puck_detections,
            'field_detection': field_detection,
            'goal_zones': goal_zones,
            'center_circles': center_circles,
            'red_circles': red_circles,
            'blue_lines': blue_lines,
            'center_line': center_line
        }


class VideoProcessor:
    """Process hockey videos and generate metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize video processor."""
        self.config = config
        self.metrics_tracker = HockeyMetricsTracker(config)
        self.detector = RoboflowDetector(config)
        
        # Processing settings
        self.processing_config = config.get('processing', {})
        self.max_frames = self.processing_config.get('max_frames')
        self.start_time = self.processing_config.get('start_time')
        self.end_time = self.processing_config.get('end_time')
        self.fps_override = self.processing_config.get('fps_override')
        
        # Output settings
        self.output_dir = Path(config.get('analysis', {}).get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Video Processor initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Max frames: {self.max_frames or 'unlimited'}")
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process a hockey video and generate metrics."""
        print(f"\n🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = 0
        end_frame = total_frames
        
        if self.start_time is not None:
            start_frame = int(self.start_time * fps)
        if self.end_time is not None:
            end_frame = int(self.end_time * fps)
        if self.max_frames:
            end_frame = min(end_frame, start_frame + self.max_frames)
        
        frames_to_process = end_frame - start_frame
        
        print(f"📊 Video info: {total_frames} total frames, {fps:.2f} FPS")
        print(f"🎯 Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
        
        # Process frames
        frame_idx = start_frame
        processed_frames = 0
        start_time = time.time()
        
        try:
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self._process_single_frame(frame, frame_idx, fps)
                if frame_result:
                    processed_frames += 1
                
                frame_idx += 1
                
                # Progress update
                if processed_frames % 30 == 0:  # Every second at 30 FPS
                    elapsed = time.time() - start_time
                    frames_per_second = processed_frames / elapsed
                    remaining_frames = frames_to_process - processed_frames
                    eta = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                    
                    print(f"⏳ Progress: {processed_frames}/{frames_to_process} frames "
                          f"({processed_frames/frames_to_process*100:.1f}%) - ETA: {eta:.1f}s")
                
        finally:
            cap.release()
        
        # Generate final metrics
        metrics_summary = self.metrics_tracker.get_metrics_summary()
        
        # Export results
        self._export_results(video_path, metrics_summary)
        
        processing_time = time.time() - start_time
        print(f"\n✅ Processing complete in {processing_time:.2f}s")
        print(f"📈 Generated {metrics_summary.get('total_events', 0)} hockey events")
        
        return {
            'processing_time': processing_time,
            'processed_frames': processed_frames,
            'metrics_summary': metrics_summary,
            'output_directory': str(self.output_dir)
        }
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> Optional[Dict[str, Any]]:
        """Process a single video frame."""
        try:
            # Detect players and puck
            detections = self.detector.detect_frame(frame, frame_idx)
            
            if not detections:
                return None
            
            # Process with metrics tracker
            result = self.metrics_tracker.process_frame(detections, frame_idx, fps)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing frame {frame_idx}: {str(e)}")
            return None
    
    def _export_results(self, video_path: str, metrics_summary: Dict[str, Any]):
        """Export all processing results."""
        video_name = Path(video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Export metrics summary
        metrics_path = self.output_dir / f"{video_name}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        # 2. Export events to CSV
        events_csv_path = self.output_dir / f"{video_name}_events_{timestamp}.csv"
        self.metrics_tracker.export_events_to_csv(str(events_csv_path))
        
        # 3. Export processing report
        report_path = self.output_dir / f"{video_name}_report_{timestamp}.txt"
        self._generate_report(report_path, video_path, metrics_summary)
        
        print(f"\n📁 Results exported to: {self.output_dir}")
        print(f"   📊 Metrics summary: {metrics_path}")
        print(f"   📋 Events CSV: {events_csv_path}")
        print(f"   📝 Processing report: {report_path}")
    
    def _generate_report(self, report_path: Path, video_path: str, metrics: Dict[str, Any]):
        """Generate a human-readable processing report."""
        with open(report_path, 'w') as f:
            f.write("HOCKEY METRICS ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Video: {video_path}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Period information
            f.write("PERIOD INFORMATION:\n")
            f.write("-" * 20 + "\n")
            for period in metrics.get('periods', []):
                f.write(f"Period {period['period_number']}:\n")
                f.write(f"  Home attacking: {period.get('home_attacking', 'unknown')}\n")
                f.write(f"  Away attacking: {period.get('away_attacking', 'unknown')}\n")
                end_time = period.get('end_timestamp', 0)
                if end_time is None:
                    f.write(f"  Duration: ongoing\n\n")
                else:
                    f.write(f"  Duration: {end_time - period['start_timestamp']:.1f}s\n\n")
            
            # Team metrics
            f.write("TEAM METRICS:\n")
            f.write("-" * 15 + "\n")
            team_metrics = metrics.get('team_metrics', {})
            for team, team_data in team_metrics.items():
                f.write(f"{team.upper()} team:\n")
                f.write(f"  Controlled entries: {team_data['controlled_entries']}\n")
                f.write(f"  Dump-ins: {team_data['dump_ins']}\n")
                f.write(f"  Shots: {team_data['shots']}\n")
                f.write(f"  Low-to-high passes: {team_data['passes']}\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-" * 18 + "\n")
            f.write(f"Total events: {metrics.get('total_events', 0)}\n")
            f.write(f"Total duration: {metrics.get('total_duration_seconds', 0):.1f}s\n")
            f.write(f"Zone entries: {metrics.get('zone_entries', 0)}\n")
            f.write(f"Breakouts: {metrics.get('breakouts', 0)}\n")
            f.write(f"Shots on goal: {metrics.get('shots_on_goal', 0)}\n")
            
            # Zone entry efficiency
            efficiency = metrics.get('zone_entry_efficiency', {})
            if efficiency:
                f.write(f"Controlled entry rate: {efficiency.get('controlled_entry_rate', 0):.1%}\n")
                f.write(f"Dump-in rate: {efficiency.get('dump_in_rate', 0):.1%}\n")
            
            # Events per minute
            events_per_min = metrics.get('events_per_minute', {})
            if events_per_min:
                f.write(f"\nEVENTS PER MINUTE:\n")
                f.write("-" * 20 + "\n")
                for event_type, rate in events_per_min.items():
                    f.write(f"{event_type}: {rate:.2f}/min\n")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hockey Metrics Tool - Video Analysis")
    parser.add_argument("video_path", help="Path to hockey video file")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--start-time", type=float, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, help="End time in seconds")
    parser.add_argument("--fps-override", type=float, help="Override video FPS")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.output_dir:
            config['analysis']['output_dir'] = args.output_dir
        if args.max_frames:
            config['processing']['max_frames'] = args.max_frames
        if args.start_time:
            config['processing']['start_time'] = args.start_time
        if args.end_time:
            config['processing']['end_time'] = args.end_time
        if args.fps_override:
            config['processing']['fps_override'] = args.fps_override
        
        # Initialize processor
        processor = VideoProcessor(config)
        
        # Process video
        results = processor.process_video(args.video_path)
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏒 HOCKEY METRICS ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"⏱️  Processing time: {results['processing_time']:.2f}s")
        print(f"🎬 Processed frames: {results['processed_frames']}")
        print(f"📊 Total events: {results['metrics_summary'].get('total_events', 0)}")
        print(f"📁 Output directory: {results['output_directory']}")
        
        # Print key metrics
        metrics = results['metrics_summary']
        if metrics:
            print("\n📈 KEY METRICS:")
            print(f"   🎯 Controlled entries: {metrics.get('controlled_entries', 0)}")
            print(f"   🏒 Dump-ins: {metrics.get('dump_ins', 0)}")
            print(f"   🚀 Breakouts: {metrics.get('breakouts', 0)}")
            print(f"   🥅 Shots on goal: {metrics.get('shots_on_goal', 0)}")
            print(f"   🔄 Low-to-high passes: {metrics.get('low_to_high_passes', 0)}")
            
            # Zone entry efficiency
            efficiency = metrics.get('zone_entry_efficiency', {})
            if efficiency:
                controlled_rate = efficiency.get('controlled_entry_rate', 0)
                print(f"   📊 Controlled entry rate: {controlled_rate:.1%}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
