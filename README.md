# 🏒 Hockey Metrics Tool

A **standalone, independent** hockey analytics tool that tracks advanced metrics like entry denials, controlled entries, breakouts, shots on goal, and low-to-high passes.

**This is NOT part of the Computer-Vision-for-Hockey project** - it's a completely separate tool you can run on its own.

## 🎯 What This Tool Does

- **Tracks hockey events** using Roboflow detection
- **Handles team side switching** after periods automatically
- **Calculates advanced metrics** like zone entry efficiency
- **Exports data** to CSV, JSON, and other formats
- **Works independently** - no dependencies on other projects

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Roboflow
Edit `config.json` with your Roboflow API credentials:
```json
{
  "api_key": "your_roboflow_key",
  "workspace_name": "your_workspace",
  "workflow_id": "your_workflow"
}
```

### 3. Run Analysis
```bash
python analyze_video.py video.mp4
```

## 📁 Project Structure

```
hockey_metrics_tool/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.json              # Configuration file
├── analyze_video.py         # Main analysis script
├── src/
│   ├── metrics_tracker.py   # Core analytics engine
│   ├── video_processor.py   # Video processing
│   └── data_exporter.py     # Data export utilities
├── examples/                 # Example videos and outputs
└── outputs/                  # Analysis results
```

## 🎬 How to Use

### Basic Usage
```bash
# Analyze a video file
python analyze_video.py hockey_game.mp4

# Analyze with custom config
python analyze_video.py hockey_game.mp4 --config my_config.json

# Analyze specific time range
python analyze_video.py hockey_game.mp4 --start-time 300 --end-time 1800
```

### Advanced Usage
```bash
# Process multiple videos
python analyze_video.py video1.mp4 video2.mp4 video3.mp4

# Export to specific format
python analyze_video.py hockey_game.mp4 --format csv

# Custom output directory
python analyze_video.py hockey_game.mp4 --output-dir my_results
```

## 📊 Metrics Tracked

### Zone Events
- **Controlled Entries** - Player with puck crosses blue line
- **Dump-ins** - Puck crosses blue line without player control
- **Breakouts** - Puck exits defensive zone
- **Zone Exits/Entries** - Movement across goal lines

### Offensive Actions
- **Shots on Goal** - Puck movement toward goal areas
- **Low-to-High Passes** - Passes from defensive to offensive zones

### Team Analysis
- **Zone Entry Efficiency** - Controlled vs. dump-in rates
- **Possession Metrics** - Time in each zone
- **Period Performance** - Metrics broken down by period

## 🔄 Dynamic Team Side Switching

This tool automatically handles the fact that teams switch sides after each period:

- **Period 1**: Home attacks NORTH, Away attacks SOUTH
- **Period 2**: Home attacks SOUTH, Away attacks NORTH
- **Period 3**: Home attacks NORTH, Away attacks SOUTH

All metrics are calculated correctly regardless of which direction each team is attacking.

## 📁 Output Files

### 1. Events CSV
Detailed frame-by-frame events with:
- Event type and timestamp
- Player and puck positions
- Team attribution
- Period information
- Confidence scores

### 2. Metrics Summary JSON
Comprehensive game statistics:
- Team-specific metrics
- Period breakdowns
- Zone entry efficiency
- Event rates per minute

### 3. Processing Report
System performance metrics:
- Frames processed
- Detection success rates
- Processing time
- Error counts

## 🛠️ Configuration

### Basic Settings
```json
{
  "api_key": "your_roboflow_key",
  "workspace_name": "your_workspace",
  "workflow_id": "your_workflow",
  "output_dir": "outputs",
  "rink_dimensions": [1400, 600]
}
```

### Advanced Settings
```json
{
  "period_detection": {
    "enabled": true,
    "period_switch_threshold": 300,
    "default_home_attacking": "north"
  },
  "detection": {
    "min_confidence": 0.5,
    "velocity_threshold": 50.0
  }
}
```

## 🔧 Customization

### Adding New Metrics
1. Define event type in `metrics_tracker.py`
2. Implement detection logic
3. Add to metrics calculation
4. Update export functions

### Modifying Zone Boundaries
Adjust zone definitions in the tracker to match your specific rink setup.

## 📈 Example Output

### Team Metrics
```
HOME team:
  Controlled entries: 12
  Dump-ins: 3
  Shots: 8
  Low-to-high passes: 15

AWAY team:
  Controlled entries: 8
  Dump-ins: 7
  Shots: 5
  Low-to-high passes: 12
```

### Zone Entry Efficiency
```
Controlled entry rate: 80.0%
Dump-in rate: 20.0%
```

## 🚨 Troubleshooting

### Common Issues
1. **Roboflow API errors** - Check API key and workspace
2. **Video not processing** - Verify video format and file path
3. **Metrics seem wrong** - Check rink dimensions in config

### Getting Help
- Check the logs in the output directory
- Verify your Roboflow workflow is working
- Ensure video has clear hockey action

## 🔮 Future Features

- **Real-time processing** for live streams
- **Web dashboard** for viewing results
- **Team color detection** for automatic team identification
- **Advanced analytics** like player fatigue and formation analysis

## 📄 License

This is a standalone tool. Use it however you want!

---

**🎯 This tool is completely independent and gives you professional hockey analytics without any dependencies on other projects!**
