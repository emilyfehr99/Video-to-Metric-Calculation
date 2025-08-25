#!/usr/bin/env python3
"""
Hockey Metrics Tool - Demo Script
Demonstrates the standalone hockey metrics tool with simulated data.
"""

import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from metrics_tracker import HockeyMetricsTracker


def run_demo():
    """Run a demonstration of the hockey metrics tool."""
    
    print("🏒 HOCKEY METRICS TOOL - STANDALONE DEMO")
    print("=" * 50)
    print("This demo shows how the tool works with simulated hockey data.")
    print("No real video files needed!\n")
    
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print("❌ Configuration file not found. Please run setup first.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize tracker
    print("🚀 Initializing Hockey Metrics Tracker...")
    tracker = HockeyMetricsTracker(config)
    
    # Simulate hockey game data
    print("\n🎬 Simulating hockey game data...")
    
    fps = 30.0
    frame_id = 0
    
    # Period 1: Home attacks north
    print("\n📅 PERIOD 1: Home team attacks NORTH")
    print("-" * 40)
    
    for i in range(600):  # 20 seconds
        frame_id += 1
        timestamp = frame_id / fps
        
        # Simulate puck movement from defensive to offensive zone
        if i < 200:  # First 6.7 seconds
            puck_pos = (700, 200)  # Defensive zone (south)
        elif i < 400:  # Next 6.7 seconds
            puck_pos = (700, 400)  # Crossing blue line
        else:  # Last 6.7 seconds
            puck_pos = (700, 500)  # Offensive zone (north)
        
        # Simulate player near puck
        player_pos = (700, puck_pos[1] - 30)
        
        frame_data = {
            'players': [
                {
                    'player_id': 'home_player_1',
                    'rink_position': {'x': player_pos[0], 'y': player_pos[1]},
                    'confidence': 0.9,
                    'class': 'player'
                }
            ],
            'puck_detections': [
                {
                    'rink_position': {'x': puck_pos[0], 'y': puck_pos[1]},
                    'confidence': 0.8,
                    'class': 'puck'
                }
            ]
        }
        
        # Process frame
        result = tracker.process_frame(frame_data, frame_id, fps)
        
        # Show key events
        if result['events']:
            for event in result['events']:
                if event['event_type'] == 'controlled_entry':
                    print(f"🎯 Frame {frame_id}: Controlled entry detected for HOME team")
                elif event['event_type'] == 'shot_on_goal':
                    print(f"🥅 Frame {frame_id}: Shot on goal by HOME team")
    
    # Period 2: Home attacks south
    print(f"\n⏸️  Period 1 ended at frame {frame_id}")
    print("Teams switching sides...")
    
    print("\n📅 PERIOD 2: Home team attacks SOUTH")
    print("-" * 40)
    
    # Force period detection
    tracker.detect_period_start(frame_id + 1, (frame_id + 1) / fps, "south")
    
    for i in range(600):  # Another 20 seconds
        frame_id += 1
        timestamp = frame_id / fps
        
        # Now home team attacks south, so their offensive zone is at the bottom
        if i < 200:
            puck_pos = (700, 500)  # Defensive zone (north)
        elif i < 400:
            puck_pos = (700, 300)  # Crossing blue line
        else:
            puck_pos = (700, 200)  # Offensive zone (south)
        
        # Simulate player near puck
        player_pos = (700, puck_pos[1] + 30)
        
        frame_data = {
            'players': [
                {
                    'player_id': 'home_player_1',
                    'rink_position': {'x': player_pos[0], 'y': player_pos[1]},
                    'confidence': 0.9,
                    'class': 'player'
                }
            ],
            'puck_detections': [
                {
                    'rink_position': {'x': puck_pos[0], 'y': puck_pos[1]},
                    'confidence': 0.8,
                    'class': 'puck'
                }
            ]
        }
        
        # Process frame
        result = tracker.process_frame(frame_data, frame_id, fps)
        
        # Show key events
        if result['events']:
            for event in result['events']:
                if event['event_type'] == 'controlled_entry':
                    print(f"🎯 Frame {frame_id}: Controlled entry detected for HOME team (now attacking SOUTH)")
                elif event['event_type'] == 'shot_on_goal':
                    print(f"🥅 Frame {frame_id}: Shot on goal by HOME team (now attacking SOUTH)")
    
    # Period 3: Home attacks north again
    print(f"\n⏸️  Period 2 ended at frame {frame_id}")
    print("Teams switching sides again...")
    
    print("\n📅 PERIOD 3: Home team attacks NORTH again")
    print("-" * 40)
    
    # Force period detection
    tracker.detect_period_start(frame_id + 1, (frame_id + 1) / fps, "north")
    
    for i in range(300):  # 10 seconds
        frame_id += 1
        timestamp = frame_id / fps
        
        # Home team back to attacking north
        if i < 150:
            puck_pos = (700, 200)  # Defensive zone (south)
        else:
            puck_pos = (700, 500)  # Offensive zone (north)
        
        player_pos = (700, puck_pos[1] - 30)
        
        frame_data = {
            'players': [
                {
                    'player_id': 'home_player_1',
                    'rink_position': {'x': player_pos[0], 'y': player_pos[1]},
                    'confidence': 0.9,
                    'class': 'player'
                }
            ],
            'puck_detections': [
                {
                    'rink_position': {'x': puck_pos[0], 'y': puck_pos[1]},
                    'confidence': 0.8,
                    'class': 'puck'
                }
            ]
        }
        
        result = tracker.process_frame(frame_data, frame_id, fps)
        
        if result['events']:
            for event in result['events']:
                if event['event_type'] == 'controlled_entry':
                    print(f"🎯 Frame {frame_id}: Controlled entry detected for HOME team (back to attacking NORTH)")
    
    return tracker


def analyze_results(tracker):
    """Analyze and display the results of the simulation."""
    
    print("\n" + "=" * 50)
    print("📊 FINAL METRICS ANALYSIS")
    print("=" * 50)
    
    # Get comprehensive metrics
    summary = tracker.get_metrics_summary()
    
    # Display period information
    print(f"\n📅 PERIOD INFORMATION:")
    for period in summary.get('periods', []):
        print(f"   Period {period['period_number']}:")
        print(f"     Home attacking: {period.get('home_attacking', 'unknown')}")
        print(f"     Away attacking: {period.get('away_attacking', 'unknown')}")
        end_time = period.get('end_timestamp', 0)
        if end_time is None:
            print(f"     Duration: ongoing")
        else:
            print(f"     Duration: {end_time - period['start_timestamp']:.1f}s")
    
    # Display team-specific metrics
    print(f"\n🏆 TEAM-SPECIFIC METRICS:")
    team_metrics = summary.get('team_metrics', {})
    for team, metrics in team_metrics.items():
        print(f"   {team.upper()} team:")
        print(f"     Controlled entries: {metrics['controlled_entries']}")
        print(f"     Dump-ins: {metrics['dump_ins']}")
        print(f"     Shots: {metrics['shots']}")
        print(f"     Low-to-high passes: {metrics['passes']}")
    
    # Display overall metrics
    print(f"\n📈 OVERALL METRICS:")
    print(f"   Total events: {summary.get('total_events', 0)}")
    print(f"   Total duration: {summary.get('total_duration_seconds', 0):.1f}s")
    print(f"   Zone entries: {summary.get('zone_entries', 0)}")
    print(f"   Breakouts: {summary.get('breakouts', 0)}")
    print(f"   Shots on goal: {summary.get('shots_on_goal', 0)}")
    
    # Zone entry efficiency
    efficiency = summary.get('zone_entry_efficiency', {})
    if efficiency:
        print(f"   Controlled entry rate: {efficiency.get('controlled_entry_rate', 0):.1%}")
        print(f"   Dump-in rate: {efficiency.get('dump_in_rate', 0):.1%}")
    
    # Events per minute
    events_per_min = summary.get('events_per_minute', {})
    if events_per_min:
        print(f"\n⏱️  EVENTS PER MINUTE:")
        for event_type, rate in events_per_min.items():
            print(f"   {event_type}: {rate:.2f}/min")
    
    return summary


def export_demo_results(tracker, summary):
    """Export the demo results to files."""
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Export events to CSV
    events_csv = output_dir / "demo_events.csv"
    tracker.export_events_to_csv(str(events_csv))
    
    # Export metrics summary to JSON
    metrics_json = output_dir / "demo_metrics.json"
    tracker.export_metrics_to_json(str(metrics_json))
    
    # Export detailed analysis
    analysis_file = output_dir / "demo_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("HOCKEY METRICS TOOL - STANDALONE DEMO RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PERIOD INFORMATION:\n")
        for period in summary.get('periods', []):
            f.write(f"Period {period['period_number']}:\n")
            f.write(f"  Home attacking: {period.get('home_attacking', 'unknown')}\n")
            f.write(f"  Away attacking: {period.get('away_attacking', 'unknown')}\n")
            end_time = period.get('end_timestamp', 0)
            if end_time is None:
                f.write(f"  Duration: ongoing\n\n")
            else:
                f.write(f"  Duration: {end_time - period['start_timestamp']:.1f}s\n\n")
        
        f.write("TEAM-SPECIFIC METRICS:\n")
        team_metrics = summary.get('team_metrics', {})
        for team, metrics in team_metrics.items():
            f.write(f"{team.upper()} team:\n")
            f.write(f"  Controlled entries: {metrics['controlled_entries']}\n")
            f.write(f"  Dump-ins: {metrics['dump_ins']}\n")
            f.write(f"  Shots: {metrics['shots']}\n")
            f.write(f"  Low-to-high passes: {metrics['passes']}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"Total events: {summary.get('total_events', 0)}\n")
        f.write(f"Total duration: {summary.get('total_duration_seconds', 0):.1f}s\n")
        f.write(f"Zone entries: {summary.get('zone_entries', 0)}\n")
        f.write(f"Breakouts: {summary.get('breakouts', 0)}\n")
        f.write(f"Shots on goal: {summary.get('shots_on_goal', 0)}\n")
    
    print(f"\n📁 Demo results exported to: {output_dir}")
    print(f"   📋 Events CSV: {events_csv}")
    print(f"   📊 Metrics JSON: {metrics_json}")
    print(f"   📝 Analysis: {analysis_file}")


def main():
    """Main demo function."""
    
    print("🚀 Starting Hockey Metrics Tool Demo...")
    print("This demo shows how the standalone tool works.\n")
    
    try:
        # Run the simulation
        tracker = run_demo()
        
        # Analyze results
        summary = analyze_results(tracker)
        
        # Export results
        export_demo_results(tracker, summary)
        
        print("\n✅ Demo completed successfully!")
        print("\n🎯 KEY FEATURES DEMONSTRATED:")
        print("   • Dynamic team side switching after periods")
        print("   • Team-aware zone entry detection")
        print("   • Period-based attacking direction tracking")
        print("   • Comprehensive metrics calculation")
        print("   • Export to multiple formats (CSV, JSON, TXT)")
        print("   • Completely standalone - no dependencies on other projects!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
