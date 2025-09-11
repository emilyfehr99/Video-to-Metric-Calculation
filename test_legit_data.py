#!/usr/bin/env python3
"""
Test Video-to-Metric-Calculation with legitimate 100-frame data
"""

import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from metrics_tracker import HockeyMetricsTracker

def test_with_legit_data():
    """Test the metrics tracker with the legitimate 100-frame data."""
    print("🏒 Testing Video-to-Metric-Calculation with Legitimate 100-Frame Data")
    print("=" * 70)
    
    # Load the legitimate tracking data
    json_file = "../Computer-Vision-for-Hockey/output/tracking_results_20250910_092038/player_detection_data_20250910_092735.json"
    
    try:
        with open(json_file, 'r') as f:
            tracking_data = json.load(f)
        print(f"✅ Loaded legitimate tracking data from {json_file}")
    except Exception as e:
        print(f"❌ Failed to load tracking data: {e}")
        return
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        print("✅ Loaded configuration")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Initialize tracker
    try:
        tracker = HockeyMetricsTracker(config)
        print("✅ Initialized HockeyMetricsTracker")
    except Exception as e:
        print(f"❌ Failed to initialize tracker: {e}")
        return
    
    # Process the legitimate data
    print("\n🔄 Processing legitimate 100-frame data...")
    
    total_events = 0
    processed_frames = 0
    
    for frame_data in tracking_data.get('frames', []):
        try:
            # Convert the tracking data format to what the metrics tracker expects
            frame_id = frame_data.get('frame_id', 0)
            
            # Extract players and puck data
            players = []
            puck_data = None
            
            for detection in frame_data.get('players', []):
                if detection.get('type') == 'player':
                    players.append({
                        'player_id': detection.get('player_id'),
                        'rink_position': detection.get('rink_position', {}),
                        'team': detection.get('team', 'unknown'),
                        'confidence': detection.get('team_confidence', 0.5)
                    })
                elif detection.get('type') == 'puck':
                    puck_data = {
                        'rink_position': detection.get('rink_position', {}),
                        'velocity': detection.get('speed', 0),
                        'confidence': 0.9
                    }
            
            # Create frame data in expected format
            processed_frame_data = {
                'frame_id': frame_id,
                'timestamp': frame_data.get('timestamp', 0.0),
                'players': players,
                'puck': puck_data
            }
            
            # Process frame
            events = tracker.process_frame(processed_frame_data, frame_id, fps=30.0)
            
            if events:
                total_events += len(events)
                print(f"   Frame {frame_id}: {len(events)} events detected")
            
            processed_frames += 1
            
        except Exception as e:
            print(f"   ❌ Error processing frame {frame_data.get('frame_id', 'unknown')}: {e}")
            continue
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"   Processed frames: {processed_frames}")
    print(f"   Total events detected: {total_events}")
    print(f"   Events per frame: {total_events/processed_frames if processed_frames > 0 else 0:.2f}")
    
    # Get final metrics
    try:
        final_metrics = tracker.get_final_metrics()
        print(f"\n🏒 FINAL METRICS:")
        for team, metrics in final_metrics.items():
            if team != 'processing_info':
                print(f"   {team}:")
                for metric, value in metrics.items():
                    print(f"     {metric}: {value}")
    except Exception as e:
        print(f"❌ Failed to get final metrics: {e}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_with_legit_data()
