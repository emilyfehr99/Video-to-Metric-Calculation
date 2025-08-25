#!/usr/bin/env python3
"""
Hockey Metrics Tracker - Standalone Version
Advanced analytics system for tracking hockey-specific metrics using Roboflow detection.
Completely independent from other projects.
"""

import cv2
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from datetime import datetime
import math
import os


class ZoneType(Enum):
    """Hockey zone types for analysis."""
    DEFENSIVE = "defensive"
    NEUTRAL = "neutral"
    OFFENSIVE = "offensive"


class EventType(Enum):
    """Hockey event types for tracking."""
    ENTRY_DENIAL = "entry_denial"
    CONTROLLED_ENTRY = "controlled_entry"
    DUMP_IN = "dump_in"
    BREAKOUT = "breakout"
    SHOT_ON_GOAL = "shot_on_goal"
    LOW_TO_HIGH_PASS = "low_to_high_pass"
    ZONE_EXIT = "zone_exit"
    ZONE_ENTRY = "zone_entry"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"


@dataclass
class ZoneBoundary:
    """Represents a zone boundary line."""
    name: str
    y_coordinate: float  # Y-coordinate in rink space
    zone_above: ZoneType
    zone_below: ZoneType


@dataclass
class HockeyEvent:
    """Represents a hockey event with metadata."""
    event_type: EventType
    frame_id: int
    timestamp: float
    player_id: Optional[str]
    puck_position: Tuple[float, float]
    player_positions: Dict[str, Tuple[float, float]]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class PeriodInfo:
    """Information about a game period."""
    period_number: int
    start_frame: int
    end_frame: Optional[int]
    start_timestamp: float
    end_timestamp: Optional[float]
    home_attacking_direction: str  # "north" or "south" in rink coordinates
    away_attacking_direction: str


class HockeyMetricsTracker:
    """
    Advanced hockey metrics tracking system using Roboflow detection.
    Handles dynamic team side switching after periods.
    Completely standalone - no dependencies on other projects.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hockey metrics tracker.
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        self.config = config
        self.analysis_config = config.get('analysis', {})
        self.period_config = config.get('period_detection', {})
        
        # Rink dimensions
        rink_dims = self.analysis_config.get('rink_dimensions', [1400, 600])
        self.rink_width, self.rink_height = rink_dims
        
        # Define zone boundaries (in rink coordinate space)
        self.zone_boundaries = self._setup_zone_boundaries()
        
        # Event tracking
        self.events: List[HockeyEvent] = []
        self.current_frame_data: Optional[Dict] = None
        self.previous_frame_data: Optional[Dict] = None
        
        # Player tracking state
        self.player_positions: Dict[str, Tuple[float, float]] = {}
        self.puck_positions: List[Tuple[float, float]] = []
        self.puck_velocity: List[Tuple[float, float]] = []
        
        # Team possession tracking with dynamic side switching
        self.team_possession: Optional[str] = None  # 'home' or 'away'
        self.possession_changes: List[Dict] = []
        
        # Period tracking for side switching
        self.periods: List[PeriodInfo] = []
        self.current_period: Optional[PeriodInfo] = None
        self.period_start_detected = False
        
        # Metrics storage
        self.metrics_summary: Dict[str, Any] = {}
        
        # Configuration from config file
        self.min_confidence = self.analysis_config.get('min_confidence', 0.5)
        self.velocity_threshold = self.analysis_config.get('velocity_threshold', 50.0)
        self.entry_zone_threshold = self.analysis_config.get('entry_zone_threshold', 20.0)
        
        # Period detection settings
        self.period_detection_enabled = self.period_config.get('enabled', True)
        self.period_switch_threshold = self.period_config.get('period_switch_threshold', 300)
        self.default_home_attacking = self.period_config.get('default_home_attacking', 'north')
        
        print(f"🏒 Hockey Metrics Tracker initialized")
        print(f"   Rink dimensions: {self.rink_width} x {self.rink_height}")
        print(f"   Period detection: {'enabled' if self.period_detection_enabled else 'disabled'}")
        print(f"   Default home attacking: {self.default_home_attacking}")
        
    def _setup_zone_boundaries(self) -> List[ZoneBoundary]:
        """Setup zone boundary definitions based on NHL rink dimensions."""
        # These will be updated dynamically when Field is detected
        boundaries = [
            # Goal line (south end) - will be updated with actual field detection
            ZoneBoundary("goal_line_south", 0, ZoneType.DEFENSIVE, ZoneType.NEUTRAL),
            
            # Blue line (south end) - will be updated with actual field detection
            ZoneBoundary("blue_line_south", 0, ZoneType.DEFENSIVE, ZoneType.NEUTRAL),
            
            # Center line - will be updated with actual field detection
            ZoneBoundary("center_line", 0, ZoneType.NEUTRAL, ZoneType.NEUTRAL),
            
            # Blue line (north end) - will be updated with actual field detection
            ZoneBoundary("blue_line_north", 0, ZoneType.NEUTRAL, ZoneType.OFFENSIVE),
            
            # Goal line (north end) - will be updated with actual field detection
            ZoneBoundary("goal_line_north", 0, ZoneType.NEUTRAL, ZoneType.NEUTRAL)
        ]
        return boundaries
    
    def update_rink_geometry(self, field_detection: Dict[str, Any]):
        """
        Update rink geometry based on detected Field class.
        This makes the system work with any video dimensions.
        """
        if not field_detection:
            return
        
        # Extract field dimensions from Roboflow detection
        field_width = field_detection.get('width', 0)
        field_height = field_detection.get('height', 0)
        field_x = field_detection.get('x', 0)
        field_y = field_detection.get('y', 0)
        
        if field_width == 0 or field_height == 0:
            print("⚠️  Invalid field dimensions detected")
            return
        
        # Update rink dimensions based on actual field detection
        self.rink_width = field_width
        self.rink_height = field_height
        
        # Calculate actual zone boundaries based on NHL rink proportions
        # NHL rink: Neutral zone ~25%, Offensive/Defensive zones ~37.5% each
        
        # Goal lines (at the ends of the rink)
        goal_line_south_y = field_y - (field_height / 2)  # Bottom edge
        goal_line_north_y = field_y + (field_height / 2)  # Top edge
        
        # Blue lines (at 25% and 75% of rink length from each end)
        blue_line_south_y = goal_line_south_y + (field_height * 0.25)  # 25% from bottom
        blue_line_north_y = goal_line_south_y + (field_height * 0.75)  # 75% from bottom
        
        # Center line (at 50% of rink length)
        center_line_y = field_y  # Center of field
        
        # Update zone boundaries with actual detected positions
        for boundary in self.zone_boundaries:
            if boundary.name == "goal_line_south":
                boundary.y_coordinate = goal_line_south_y
            elif boundary.name == "blue_line_south":
                boundary.y_coordinate = blue_line_south_y
            elif boundary.name == "center_line":
                boundary.y_coordinate = center_line_y
            elif boundary.name == "blue_line_north":
                boundary.y_coordinate = blue_line_north_y
            elif boundary.name == "goal_line_north":
                boundary.y_coordinate = goal_line_north_y
        
        print(f"🏒 Rink geometry updated from Field detection:")
        print(f"   Actual dimensions: {field_width:.0f} x {field_height:.0f}")
        print(f"   Blue lines at: {blue_line_south_y:.0f}, {blue_line_north_y:.0f}")
        print(f"   Center line at: {center_line_y:.0f}")
    
    def detect_period_start(self, frame_id: int, timestamp: float, 
                           home_attacking_direction: str = None) -> bool:
        """
        Detect the start of a new period.
        
        Args:
            frame_id: Current frame number
            timestamp: Current timestamp
            home_attacking_direction: Which direction home team attacks ("north" or "south")
            
        Returns:
            True if period start detected, False otherwise
        """
        if home_attacking_direction is None:
            home_attacking_direction = self.default_home_attacking
            
        # Simple period detection: if we haven't detected a period start yet
        # or if there's been a long gap in activity
        if not self.period_start_detected:
            # First period start
            period_num = 1
            self.period_start_detected = True
        else:
            # Check if we should start a new period
            if self.current_period and self.current_period.end_frame is None:
                # Check for period end (long gap in activity)
                frames_since_last_event = frame_id - self.current_period.start_frame
                if frames_since_last_event > self.period_switch_threshold:
                    # End current period
                    self.current_period.end_frame = frame_id - self.period_switch_threshold
                    self.current_period.end_timestamp = timestamp - (self.period_switch_threshold / 30.0)  # Assuming 30 FPS
                    
                    # Start new period
                    period_num = len(self.periods) + 1
                else:
                    return False
            else:
                period_num = len(self.periods) + 1
        
        # Create new period
        away_attacking_direction = "south" if home_attacking_direction == "north" else "north"
        
        new_period = PeriodInfo(
            period_number=period_num,
            start_frame=frame_id,
            end_frame=None,
            start_timestamp=timestamp,
            end_timestamp=None,
            home_attacking_direction=home_attacking_direction,
            away_attacking_direction=away_attacking_direction
        )
        
        self.periods.append(new_period)
        self.current_period = new_period
        
        # Add period start event
        period_event = HockeyEvent(
            event_type=EventType.PERIOD_START,
            frame_id=frame_id,
            timestamp=timestamp,
            player_id=None,
            puck_position=(0, 0),  # Not relevant for period events
            player_positions={},
            confidence=1.0,
            metadata={
                'period_number': period_num,
                'home_attacking': home_attacking_direction,
                'away_attacking': away_attacking_direction
            }
        )
        self.events.append(period_event)
        
        print(f"🏒 Period {period_num} started - Home attacking {home_attacking_direction}, Away attacking {away_attacking_direction}")
        return True
    
    def get_current_attacking_directions(self) -> Tuple[str, str]:
        """
        Get the current attacking directions for home and away teams.
        
        Returns:
            Tuple of (home_attacking_direction, away_attacking_direction)
        """
        if not self.current_period:
            # Default to first period
            home_attacking = self.default_home_attacking
            away_attacking = "south" if home_attacking == "north" else "north"
            return home_attacking, away_attacking
        
        return self.current_period.home_attacking_direction, self.current_period.away_attacking_direction
    
    def get_zone_for_team(self, position: Tuple[float, float], team: str) -> ZoneType:
        """
        Determine which zone a position is in for a specific team.
        
        Args:
            position: (x, y) coordinates in rink space
            team: 'home' or 'away'
            
        Returns:
            ZoneType for the team at that position
        """
        if not self.current_period:
            # Default to first period
            home_attacking = self.default_home_attacking
            away_attacking = "south" if home_attacking == "north" else "north"
        else:
            home_attacking = self.current_period.home_attacking_direction
            away_attacking = self.current_period.away_attacking_direction
        
        y = position[1]
        
        if team == "home":
            if home_attacking == "north":
                # Home attacking north (top of rink)
                if y < self.rink_height * 0.25:
                    return ZoneType.DEFENSIVE
                elif y > self.rink_height * 0.75:
                    return ZoneType.OFFENSIVE
                else:
                    return ZoneType.NEUTRAL
            else:
                # Home attacking south (bottom of rink)
                if y > self.rink_height * 0.75:
                    return ZoneType.DEFENSIVE
                elif y < self.rink_height * 0.25:
                    return ZoneType.OFFENSIVE
                else:
                    return ZoneType.NEUTRAL
        else:  # away team
            if away_attacking == "north":
                # Away attacking north (top of rink)
                if y < self.rink_height * 0.25:
                    return ZoneType.DEFENSIVE
                elif y > self.rink_height * 0.75:
                    return ZoneType.OFFENSIVE
                else:
                    return ZoneType.NEUTRAL
            else:
                # Away attacking south (bottom of rink)
                if y > self.rink_height * 0.75:
                    return ZoneType.DEFENSIVE
                elif y < self.rink_height * 0.25:
                    return ZoneType.OFFENSIVE
                else:
                    return ZoneType.NEUTRAL
    
    def process_frame(self, frame_data: Dict, frame_id: int, fps: float = 30.0) -> Dict[str, Any]:
        """
        Process a frame and detect hockey events using all available Roboflow classes.
        
        Args:
            frame_data: Frame data from Roboflow detection
            frame_id: Current frame number
            fps: Video frames per second
            
        Returns:
            Dictionary containing detected events and metrics
        """
        timestamp = frame_id / fps
        
        # Update frame data
        self.previous_frame_data = self.current_frame_data
        self.current_frame_data = frame_data
        
        # Extract all available detections
        players = frame_data.get('players', [])
        puck_detections = frame_data.get('puck_detections', [])
        field_detection = frame_data.get('field_detection', None)
        goal_zones = frame_data.get('goal_zones', [])
        center_circles = frame_data.get('center_circles', [])
        red_circles = frame_data.get('red_circles', [])
        blue_lines = frame_data.get('blue_lines', [])
        center_line = frame_data.get('center_line', None)
        
        # Update rink geometry if Field is detected
        if field_detection and not self._is_rink_geometry_initialized():
            self.update_rink_geometry(field_detection)
        
        # Update tracking state
        self._update_player_positions(players)
        self._update_puck_tracking(puck_detections, frame_id, fps)
        
        # Period detection (if enabled)
        if self.period_detection_enabled:
            self.detect_period_start(frame_id, timestamp)
        
        # Detect events using enhanced detection methods
        frame_events = []
        
        # 1. Zone Entry/Exit Detection (team-aware with actual rink elements)
        zone_events = self._detect_zone_events_team_aware(frame_id, timestamp, fps)
        frame_events.extend(zone_events)
        
        # 2. Shot Detection (using GoalZone detection)
        shot_events = self._detect_shots_team_aware(frame_id, timestamp, goal_zones)
        frame_events.extend(shot_events)
        
        # 3. Pass Detection (team-aware)
        pass_events = self._detect_passes_team_aware(frame_id, timestamp)
        frame_events.extend(pass_events)
        
        # 4. Enhanced Team Possession Changes (using faceoff circles)
        possession_events = self._detect_possession_changes(frame_id, timestamp, center_circles, red_circles)
        frame_events.extend(possession_events)
        
        # Store events
        self.events.extend(frame_events)
        
        # Calculate real-time metrics
        current_metrics = self._calculate_current_metrics(frame_id)
        
        return {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'events': [asdict(event) for event in frame_events],
            'metrics': current_metrics,
            'player_count': len(players),
            'puck_detected': len(puck_detections) > 0,
            'current_period': self.current_period.period_number if self.current_period else 1,
            'home_attacking': self.current_period.home_attacking_direction if self.current_period else self.default_home_attacking,
            'rink_elements_detected': {
                'field': field_detection is not None,
                'goal_zones': len(goal_zones),
                'center_circles': len(center_circles),
                'red_circles': len(red_circles),
                'blue_lines': len(blue_lines),
                'center_line': center_line is not None
            }
        }
    
    def _is_rink_geometry_initialized(self) -> bool:
        """Check if rink geometry has been initialized from Field detection."""
        return any(boundary.y_coordinate != 0 for boundary in self.zone_boundaries)
    
    # ... [Rest of the methods remain the same as in the original file] ...
    
    def _update_player_positions(self, players: List[Dict]):
        """Update player position tracking."""
        for player in players:
            if player.get('rink_position') and player.get('player_id'):
                pos = (
                    player['rink_position']['x'],
                    player['rink_position']['y']
                )
                self.player_positions[player['player_id']] = pos
    
    def _update_puck_tracking(self, puck_detections: List[Dict], frame_id: int, fps: float):
        """Update puck position and velocity tracking."""
        if puck_detections:
            puck = puck_detections[0]  # Assume single puck
            if puck.get('rink_position'):
                current_pos = (
                    puck['rink_position']['x'],
                    puck['rink_position']['y']
                )
                
                self.puck_positions.append(current_pos)
                
                # Calculate velocity if we have previous positions
                if len(self.puck_positions) >= 2:
                    prev_pos = self.puck_positions[-2]
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    velocity = (dx, dy)
                    self.puck_velocity.append(velocity)
                
                # Keep only recent history (last 30 frames)
                if len(self.puck_positions) > 30:
                    self.puck_positions.pop(0)
                if len(self.puck_velocity) > 29:
                    self.puck_velocity.pop(0)
    
    def _detect_zone_events_team_aware(self, frame_id: int, timestamp: float, fps: float) -> List[HockeyEvent]:
        """Detect zone entry/exit events with team awareness using actual rink elements."""
        events = []
        
        if not self.puck_positions or len(self.puck_positions) < 2:
            return events
        
        current_puck_pos = self.puck_positions[-1]
        previous_puck_pos = self.puck_positions[-2]
        
        # Enhanced boundary detection using actual rink elements
        for boundary in self.zone_boundaries:
            if self._crossed_boundary_enhanced(previous_puck_pos, current_puck_pos, boundary):
                # Determine event type based on direction and current period
                event_type = self._classify_zone_crossing_team_aware(
                    previous_puck_pos, current_puck_pos, boundary
                )
                
                if event_type:
                    # Find nearest player for controlled entries
                    nearest_player = self._find_nearest_player(current_puck_pos)
                    
                    # Determine which team this event belongs to
                    attacking_team = self._determine_attacking_team(current_puck_pos, boundary)
                    
                    # Enhanced metadata with rink element information
                    event = HockeyEvent(
                        event_type=event_type,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        player_id=nearest_player,
                        puck_position=current_puck_pos,
                        player_positions=self.player_positions.copy(),
                        confidence=0.8,
                        metadata={
                            'boundary': boundary.name,
                            'direction': 'forward' if current_puck_pos[1] > previous_puck_pos[1] else 'backward',
                            'attacking_team': attacking_team,
                            'period': self.current_period.period_number if self.current_period else 1,
                            'puck_trajectory': {
                                'start': previous_puck_pos,
                                'end': current_puck_pos,
                                'distance': self._calculate_distance(previous_puck_pos, current_puck_pos)
                            }
                        }
                    )
                    events.append(event)
        
        return events
    
    def _crossed_boundary_enhanced(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                                   boundary: ZoneBoundary) -> bool:
        """Enhanced boundary crossing detection considering both X and Y coordinates."""
        if boundary.y_coordinate == 0:
            return False  # Boundary not yet initialized
        
        y1, y2 = pos1[1], pos2[1]
        x1, x2 = pos1[0], pos2[0]
        boundary_y = boundary.y_coordinate
        
        # Check if positions are on opposite sides of boundary
        y_crossed = (y1 < boundary_y and y2 >= boundary_y) or (y1 > boundary_y and y2 <= boundary_y)
        
        # For blue lines, also consider significant lateral movement
        if boundary.name in ["blue_line_south", "blue_line_north"]:
            # Check if puck moved significantly in both directions
            y_movement = abs(y2 - y1)
            x_movement = abs(x2 - x1)
            
            # Must cross Y boundary AND have meaningful movement
            return y_crossed and (y_movement > 10 or x_movement > 10)
        
        return y_crossed
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _crossed_boundary(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                          boundary: ZoneBoundary) -> bool:
        """Check if puck crossed a specific boundary."""
        y1, y2 = pos1[1], pos2[1]
        boundary_y = boundary.y_coordinate
        
        # Check if positions are on opposite sides of boundary
        return (y1 < boundary_y and y2 >= boundary_y) or (y1 > boundary_y and y2 <= boundary_y)
    
    def _classify_zone_crossing_team_aware(self, pos1: Tuple[float, float], pos2: Tuple[float, float], 
                                          boundary: ZoneBoundary) -> Optional[EventType]:
        """Classify zone crossing events with team awareness."""
        # Determine direction
        moving_forward = pos2[1] > pos1[1]  # Moving toward north end
        
        if boundary.name == "blue_line_north" and moving_forward:
            # Check if this is a controlled entry or dump-in
            nearest_player = self._find_nearest_player(pos2)
            if nearest_player and self._is_controlled_entry(pos2, nearest_player):
                return EventType.CONTROLLED_ENTRY
            else:
                return EventType.DUMP_IN
        
        elif boundary.name == "blue_line_south" and not moving_forward:
            # Defensive zone exit (breakout)
            return EventType.BREAKOUT
        
        elif boundary.name == "goal_line_north" and moving_forward:
            # Offensive zone entry
            return EventType.ZONE_ENTRY
        
        elif boundary.name == "goal_line_south" and not moving_forward:
            # Defensive zone exit
            return EventType.ZONE_EXIT
        
        return None
    
    def _determine_attacking_team(self, puck_pos: Tuple[float, float], boundary: ZoneBoundary) -> str:
        """Determine which team is attacking based on puck position and boundary."""
        if not self.current_period:
            return "unknown"
        
        # Determine which direction the puck is moving relative to the boundary
        if boundary.name in ["blue_line_north", "goal_line_north"]:
            # Moving toward north end
            if self.current_period.home_attacking_direction == "north":
                return "home"
            else:
                return "away"
        elif boundary.name in ["blue_line_south", "goal_line_south"]:
            # Moving toward south end
            if self.current_period.home_attacking_direction == "south":
                return "home"
            else:
                return "away"
        
        return "unknown"
    
    def _is_controlled_entry(self, puck_pos: Tuple[float, float], player_id: str) -> bool:
        """Determine if a zone entry is controlled (player near puck)."""
        if player_id not in self.player_positions:
            return False
        
        player_pos = self.player_positions[player_id]
        distance = math.sqrt(
            (puck_pos[0] - player_pos[0])**2 + 
            (puck_pos[1] - player_pos[1])**2
        )
        
        # Consider controlled if player is within 50 pixels of puck
        return distance <= 50.0
    
    def _find_nearest_player(self, position: Tuple[float, float]) -> Optional[str]:
        """Find the player closest to a given position."""
        if not self.player_positions:
            return None
        
        nearest_player = None
        min_distance = float('inf')
        
        for player_id, player_pos in self.player_positions.items():
            distance = math.sqrt(
                (position[0] - player_pos[0])**2 + 
                (position[1] - player_pos[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_player = player_id
        
        return nearest_player
    
    def _detect_shots_team_aware(self, frame_id: int, timestamp: float, goal_zones: List[Dict] = None) -> List[HockeyEvent]:
        """Detect shots on goal with team awareness using GoalZone detection."""
        events = []
        
        if len(self.puck_velocity) < 3:
            return events
        
        current_velocity = self.puck_velocity[-1]
        velocity_magnitude = math.sqrt(current_velocity[0]**2 + current_velocity[1]**2)
        
        if velocity_magnitude > self.velocity_threshold:
            current_puck_pos = self.puck_positions[-1]
            
            # Enhanced shot detection using GoalZone if available
            shot_detected = False
            goal_zone_info = None
            
            if goal_zones:
                # Check if puck is moving toward detected goal zones
                for goal_zone in goal_zones:
                    if self._is_puck_moving_toward_goalzone(current_puck_pos, current_velocity, goal_zone):
                        shot_detected = True
                        goal_zone_info = goal_zone
                        break
            else:
                # Fallback to team-aware detection
                shot_detected = self._is_moving_toward_goal_team_aware(current_puck_pos, current_velocity)
            
            if shot_detected:
                # Determine which team is shooting
                shooting_team = self._determine_shooting_team(current_puck_pos, current_velocity)
                
                # Calculate shot accuracy if goal zone is detected
                shot_accuracy = None
                if goal_zone_info:
                    shot_accuracy = self._calculate_shot_accuracy(current_puck_pos, goal_zone_info)
                
                event = HockeyEvent(
                    event_type=EventType.SHOT_ON_GOAL,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    player_id=None,
                    puck_position=current_puck_pos,
                    player_positions=self.player_positions.copy(),
                    confidence=0.7,
                    metadata={
                        'velocity': velocity_magnitude,
                        'direction': current_velocity,
                        'shooting_team': shooting_team,
                        'period': self.current_period.period_number if self.current_period else 1,
                        'goal_zone_detected': goal_zone_info is not None,
                        'shot_accuracy': shot_accuracy
                    }
                )
                events.append(event)
        
        return events
    
    def _is_puck_moving_toward_goalzone(self, puck_pos: Tuple[float, float], velocity: Tuple[float, float], 
                                        goal_zone: Dict[str, Any]) -> bool:
        """Check if puck is moving toward a detected GoalZone."""
        # Extract goal zone position and dimensions
        goal_x = goal_zone.get('x', 0)
        goal_y = goal_zone.get('y', 0)
        goal_width = goal_zone.get('width', 0)
        goal_height = goal_zone.get('height', 0)
        
        # Calculate goal zone center
        goal_center_x = goal_x
        goal_center_y = goal_y
        
        # Check if puck is moving toward goal zone
        puck_x, puck_y = puck_pos
        vel_x, vel_y = velocity
        
        # Calculate distance to goal zone
        distance_to_goal = math.sqrt((puck_x - goal_center_x)**2 + (puck_y - goal_center_y)**2)
        
        # Puck should be within reasonable distance and moving toward goal
        if distance_to_goal > 300:  # Increased detection range
            return False
        
        # Check if velocity vector points toward goal zone
        if abs(vel_x) > abs(vel_y):  # Horizontal movement
            # Check if moving in right direction
            if goal_center_x > puck_x and vel_x > 0:  # Moving right toward goal
                return True
            elif goal_center_x < puck_x and vel_x < 0:  # Moving left toward goal
                return True
        else:  # Vertical movement
            # Check if moving in right direction
            if goal_center_y > puck_y and vel_y > 0:  # Moving down toward goal
                return True
            elif goal_center_y < puck_y and vel_y < 0:  # Moving up toward goal
                return True
        
        return False
    
    def _calculate_shot_accuracy(self, puck_pos: Tuple[float, float], goal_zone: Dict[str, Any]) -> float:
        """Calculate shot accuracy based on distance to goal zone center."""
        goal_x = goal_zone.get('x', 0)
        goal_y = goal_zone.get('y', 0)
        
        distance = math.sqrt((puck_pos[0] - goal_x)**2 + (puck_pos[1] - goal_y)**2)
        
        # Normalize accuracy (closer = higher accuracy)
        max_distance = 100  # Maximum distance for 100% accuracy
        accuracy = max(0, 100 - (distance / max_distance) * 100)
        
        return min(100, accuracy)
    
    def _is_moving_toward_goal_team_aware(self, puck_pos: Tuple[float, float], velocity: Tuple[float, float]) -> bool:
        """Check if puck is moving toward goal areas with team awareness."""
        if not self.current_period:
            return self._is_moving_toward_goal(puck_pos, velocity)
        
        # Check if moving toward offensive goal area based on current period
        if velocity[1] > 0:  # Moving north
            if self.current_period.home_attacking_direction == "north":
                # Home team attacking north
                return puck_pos[1] > self.rink_height * 0.75
            else:
                # Away team attacking north
                return puck_pos[1] > self.rink_height * 0.75
        
        elif velocity[1] < 0:  # Moving south
            if self.current_period.home_attacking_direction == "south":
                # Home team attacking south
                return puck_pos[1] < self.rink_height * 0.25
            else:
                # Away team attacking south
                return puck_pos[1] < self.rink_height * 0.25
        
        return False
    
    def _determine_shooting_team(self, puck_pos: Tuple[float, float], velocity: Tuple[float, float]) -> str:
        """Determine which team is shooting based on direction and period."""
        if not self.current_period:
            return "unknown"
        
        if velocity[1] > 0:  # Moving north
            if self.current_period.home_attacking_direction == "north":
                return "home"
            else:
                return "away"
        elif velocity[1] < 0:  # Moving south
            if self.current_period.home_attacking_direction == "south":
                return "home"
            else:
                return "away"
        
        return "unknown"
    
    def _detect_passes_team_aware(self, frame_id: int, timestamp: float) -> List[HockeyEvent]:
        """Detect passes with team awareness."""
        events = []
        
        if len(self.puck_positions) < 5 or len(self.player_positions) < 2:
            return events
        
        # Look for puck movement patterns that suggest passing
        if len(self.puck_velocity) >= 3:
            recent_velocities = self.puck_velocity[-3:]
            
            # Calculate velocity changes
            velocity_changes = []
            for i in range(1, len(recent_velocities)):
                v1 = recent_velocities[i-1]
                v2 = recent_velocities[i]
                change = math.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
                velocity_changes.append(change)
            
            # If there are significant velocity changes, it might be a pass
            if any(change > self.velocity_threshold * 0.3 for change in velocity_changes):
                # Determine if it's a low-to-high pass (moving from defensive to offensive zones)
                start_pos = self.puck_positions[-10]  # 10 frames ago
                end_pos = self.puck_positions[-1]
                
                if self._is_low_to_high_movement_team_aware(start_pos, end_pos):
                    # Determine which team made the pass
                    passing_team = self._determine_passing_team(start_pos, end_pos)
                    
                    event = HockeyEvent(
                        event_type=EventType.LOW_TO_HIGH_PASS,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        player_id=None,
                        puck_position=end_pos,
                        player_positions=self.player_positions.copy(),
                        confidence=0.6,
                        metadata={
                            'start_position': start_pos,
                            'end_position': end_pos,
                            'velocity_changes': velocity_changes,
                            'passing_team': passing_team,
                            'period': self.current_period.period_number if self.current_period else 1
                        }
                    )
                    events.append(event)
        
        return events
    
    def _is_low_to_high_movement_team_aware(self, start_pos: Tuple[float, float], 
                                            end_pos: Tuple[float, float]) -> bool:
        """Check if movement is from defensive to offensive zones with team awareness."""
        if not self.current_period:
            return self._is_low_to_high_movement(start_pos, end_pos)
        
        # Determine if this is a low-to-high pass based on current period
        if self.current_period.home_attacking_direction == "north":
            # Home attacking north, so moving from south to north is low-to-high
            return end_pos[1] > start_pos[1] and (end_pos[1] - start_pos[1]) > 50
        else:
            # Home attacking south, so moving from north to south is low-to-high
            return end_pos[1] < start_pos[1] and (start_pos[1] - end_pos[1]) > 50
    
    def _determine_passing_team(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]) -> str:
        """Determine which team made the pass based on direction and period."""
        if not self.current_period:
            return "unknown"
        
        # Determine if this is a low-to-high pass for home or away team
        if self.current_period.home_attacking_direction == "north":
            # Home attacking north
            if end_pos[1] > start_pos[1]:  # Moving north (low-to-high)
                return "home"
            else:  # Moving south (high-to-low)
                return "away"
        else:
            # Home attacking south
            if end_pos[1] < start_pos[1]:  # Moving south (low-to-high)
                return "home"
            else:  # Moving north (high-to-low)
                return "away"
    
    def _is_moving_toward_goal(self, puck_pos: Tuple[float, float], velocity: Tuple[float, float]) -> bool:
        """Check if puck is moving toward goal areas (legacy method)."""
        # Check if moving toward offensive goal area (top of rink)
        if velocity[1] > 0:  # Moving upward in rink coordinates
            # Check if in offensive zone
            return puck_pos[1] > self.rink_height * 0.75
        
        # Check if moving toward defensive goal area (bottom of rink)
        elif velocity[1] < 0:  # Moving downward in rink coordinates
            # Check if in defensive zone
            return puck_pos[1] < self.rink_height * 0.25
        
        return False
    
    def _is_low_to_high_movement(self, start_pos: Tuple[float, float], 
                                 end_pos: Tuple[float, float]) -> bool:
        """Check if movement is from defensive zone to offensive zone (legacy method)."""
        # Moving from lower (defensive) to higher (offensive) y-coordinates
        return end_pos[1] > start_pos[1] and (end_pos[1] - start_pos[1]) > 100
    
    def _detect_possession_changes(self, frame_id: int, timestamp: float, 
                                  center_circles: List[Dict] = None, 
                                  red_circles: List[Dict] = None) -> List[HockeyEvent]:
        """Detect changes in team possession using faceoff circles."""
        events = []
        
        if len(self.puck_positions) < 2:
            return events
        
        current_pos = self.puck_positions[-1]
        previous_pos = self.puck_positions[-2]
        
        # Enhanced possession detection using faceoff circles
        possession_info = self._analyze_faceoff_possession(
            current_pos, previous_pos, center_circles, red_circles
        )
        
        if possession_info:
            event = HockeyEvent(
                event_type=EventType.ZONE_EXIT,  # Use existing event type for now
                frame_id=frame_id,
                timestamp=timestamp,
                player_id=None,
                puck_position=current_pos,
                player_positions=self.player_positions.copy(),
                confidence=0.6,
                metadata={
                    'possession_change': True,
                    'possession_info': possession_info,
                    'period': self.current_period.period_number if self.current_period else 1
                }
            )
            events.append(event)
        
        return events
    
    def _analyze_faceoff_possession(self, current_pos: Tuple[float, float], 
                                   previous_pos: Tuple[float, float],
                                   center_circles: List[Dict] = None,
                                   red_circles: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Analyze possession using faceoff circle positions."""
        possession_info = {
            'zone': 'unknown',
            'faceoff_circle_nearby': False,
            'possession_strength': 0.0
        }
        
        # Check if puck is near center circle (neutral zone possession)
        if center_circles:
            for circle in center_circles:
                circle_x = circle.get('x', 0)
                circle_y = circle.get('y', 0)
                circle_radius = max(circle.get('width', 0), circle.get('height', 0)) / 2
                
                distance_to_center = self._calculate_distance(current_pos, (circle_x, circle_y))
                if distance_to_center < circle_radius * 2:  # Within 2x radius
                    possession_info['zone'] = 'neutral'
                    possession_info['faceoff_circle_nearby'] = True
                    possession_info['possession_strength'] = max(0, 100 - (distance_to_center / circle_radius) * 50)
                    break
        
        # Check if puck is near red circles (offensive/defensive zone possession)
        if red_circles:
            for circle in red_circles:
                circle_x = circle.get('x', 0)
                circle_y = circle.get('y', 0)
                circle_radius = max(circle.get('width', 0), circle.get('height', 0)) / 2
                
                distance_to_circle = self._calculate_distance(current_pos, (circle_x, circle_y))
                if distance_to_circle < circle_radius * 2:
                    # Determine zone based on circle position relative to center
                    if circle_y < self.rink_height / 2:
                        possession_info['zone'] = 'offensive'
                    else:
                        possession_info['zone'] = 'defensive'
                    
                    possession_info['faceoff_circle_nearby'] = True
                    possession_info['possession_strength'] = max(0, 100 - (distance_to_circle / circle_radius) * 50)
                    break
        
        # Calculate possession strength based on player proximity
        if self.player_positions:
            nearest_player_distance = float('inf')
            for player_pos in self.player_positions.values():
                distance = self._calculate_distance(current_pos, player_pos)
                nearest_player_distance = min(nearest_player_distance, distance)
            
            # Adjust possession strength based on player proximity
            if nearest_player_distance < 100:  # Player within 100 pixels
                possession_info['possession_strength'] = min(100, possession_info['possession_strength'] + 30)
            elif nearest_player_distance < 200:  # Player within 200 pixels
                possession_info['possession_strength'] = min(100, possession_info['possession_strength'] + 15)
        
        return possession_info if possession_info['zone'] != 'unknown' else None
    
    def _get_zone_change(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> Optional[str]:
        """Get the type of zone change between two positions."""
        zone1 = self._get_position_zone(pos1)
        zone2 = self._get_position_zone(pos2)
        
        if zone1 != zone2:
            return f"{zone1.value}_to_{zone2.value}"
        return None
    
    def _get_position_zone(self, position: Tuple[float, float]) -> ZoneType:
        """Determine which zone a position is in (legacy method)."""
        y = position[1]
        
        if y < self.rink_height * 0.25:
            return ZoneType.DEFENSIVE
        elif y > self.rink_height * 0.75:
            return ZoneType.OFFENSIVE
        else:
            return ZoneType.NEUTRAL
    
    def _calculate_current_metrics(self, frame_id: int) -> Dict[str, Any]:
        """Calculate current metrics based on tracked events."""
        metrics = {
            'total_events': len(self.events),
            'zone_entries': len([e for e in self.events if e.event_type in [
                EventType.CONTROLLED_ENTRY, EventType.DUMP_IN
            ]]),
            'controlled_entries': len([e for e in self.events if e.event_type == EventType.CONTROLLED_ENTRY]),
            'dump_ins': len([e for e in self.events if e.event_type == EventType.DUMP_IN]),
            'breakouts': len([e for e in self.events if e.event_type == EventType.BREAKOUT]),
            'shots_on_goal': len([e for e in self.events if e.event_type == EventType.SHOT_ON_GOAL]),
            'low_to_high_passes': len([e for e in self.events if e.event_type == EventType.LOW_TO_HIGH_PASS]),
            'zone_exits': len([e for e in self.events if e.event_type == EventType.ZONE_EXIT]),
            'zone_entries': len([e for e in self.events if e.event_type == EventType.ZONE_ENTRY])
        }
        
        # Enhanced metrics using new detection capabilities
        if self.events:
            # Shot accuracy metrics
            shots = [e for e in self.events if e.event_type == EventType.SHOT_ON_GOAL]
            if shots:
                shot_accuracies = [e.metadata.get('shot_accuracy', 0) for e in shots if e.metadata.get('shot_accuracy')]
                if shot_accuracies:
                    metrics['avg_shot_accuracy'] = sum(shot_accuracies) / len(shot_accuracies)
                    metrics['high_accuracy_shots'] = len([acc for acc in shot_accuracies if acc > 80])
            
            # Possession metrics
            possession_events = [e for e in self.events if e.metadata.get('possession_change')]
            if possession_events:
                possession_strengths = [e.metadata.get('possession_info', {}).get('possession_strength', 0) 
                                      for e in possession_events]
                if possession_strengths:
                    metrics['avg_possession_strength'] = sum(possession_strengths) / len(possession_strengths)
                    metrics['strong_possession_events'] = len([s for s in possession_strengths if s > 70])
            
            # Rink element utilization
            rink_elements = [e.metadata.get('rink_elements_detected', {}) for e in self.events 
                           if e.metadata.get('rink_elements_detected')]
            if rink_elements:
                metrics['rink_elements_utilized'] = len(set().union(*[elem.keys() for elem in rink_elements]))
        
        return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.events:
            return {}
        
        # Calculate time-based metrics
        if self.events:
            start_time = self.events[0].timestamp
            end_time = self.events[-1].timestamp
            duration = end_time - start_time
        else:
            duration = 0
        
        # Group events by type
        events_by_type = {}
        for event in self.events:
            event_type = event.event_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        # Calculate rates per minute
        events_per_minute = {}
        if duration > 0:
            for event_type, event_list in events_by_type.items():
                events_per_minute[event_type] = len(event_list) / (duration / 60.0)
        
        # Team-specific metrics
        team_metrics = self._calculate_team_specific_metrics()
        
        summary = {
            'total_duration_seconds': duration,
            'total_events': len(self.events),
            'events_by_type': {k: len(v) for k, v in events_by_type.items()},
            'events_per_minute': events_per_minute,
            'zone_entry_efficiency': self._calculate_zone_entry_efficiency(),
            'possession_metrics': self._calculate_possession_metrics(),
            'shot_metrics': self._calculate_shot_metrics(),
            'team_metrics': team_metrics,
            'periods': [asdict(period) for period in self.periods]
        }
        
        return summary
    
    def _calculate_team_specific_metrics(self) -> Dict[str, Any]:
        """Calculate metrics broken down by team."""
        team_metrics = {
            'home': {'controlled_entries': 0, 'dump_ins': 0, 'shots': 0, 'passes': 0},
            'away': {'controlled_entries': 0, 'dump_ins': 0, 'shots': 0, 'passes': 0}
        }
        
        for event in self.events:
            if event.event_type == EventType.CONTROLLED_ENTRY:
                team = event.metadata.get('attacking_team', 'unknown')
                if team in team_metrics:
                    team_metrics[team]['controlled_entries'] += 1
            elif event.event_type == EventType.DUMP_IN:
                team = event.metadata.get('attacking_team', 'unknown')
                if team in team_metrics:
                    team_metrics[team]['dump_ins'] += 1
            elif event.event_type == EventType.SHOT_ON_GOAL:
                team = event.metadata.get('shooting_team', 'unknown')
                if team in team_metrics:
                    team_metrics[team]['shots'] += 1
            elif event.event_type == EventType.LOW_TO_HIGH_PASS:
                team = event.metadata.get('passing_team', 'unknown')
                if team in team_metrics:
                    team_metrics[team]['passes'] += 1
        
        return team_metrics
    
    def _calculate_zone_entry_efficiency(self) -> Dict[str, float]:
        """Calculate zone entry efficiency metrics."""
        controlled_entries = len([e for e in self.events if e.event_type == EventType.CONTROLLED_ENTRY])
        dump_ins = len([e for e in self.events if e.event_type == EventType.DUMP_IN])
        total_entries = controlled_entries + dump_ins
        
        if total_entries == 0:
            return {'controlled_entry_rate': 0.0, 'dump_in_rate': 0.0}
        
        return {
            'controlled_entry_rate': controlled_entries / total_entries,
            'dump_in_rate': dump_ins / total_entries
        }
    
    def _calculate_possession_metrics(self) -> Dict[str, Any]:
        """Calculate possession-related metrics."""
        # This would be enhanced with actual team identification
        return {
            'possession_changes': len(self.possession_changes),
            'avg_possession_duration': 0.0  # Would need team tracking
        }
    
    def _calculate_shot_metrics(self) -> Dict[str, Any]:
        """Calculate shot-related metrics."""
        shots = [e for e in self.events if e.event_type == EventType.SHOT_ON_GOAL]
        
        if not shots:
            return {'total_shots': 0, 'shots_per_minute': 0.0}
        
        # Calculate shots per minute
        if self.events:
            duration = (self.events[-1].timestamp - self.events[0].timestamp) / 60.0
            shots_per_minute = len(shots) / duration if duration > 0 else 0
        else:
            shots_per_minute = 0
        
        return {
            'total_shots': len(shots),
            'shots_per_minute': shots_per_minute
        }
    
    def export_events_to_csv(self, output_path: str):
        """Export all events to a CSV file."""
        if not self.events:
            print("No events to export")
            return
        
        # Convert events to DataFrame
        events_data = []
        for event in self.events:
            event_dict = asdict(event)
            # Flatten metadata
            if 'metadata' in event_dict:
                for key, value in event_dict['metadata'].items():
                    event_dict[f'metadata_{key}'] = value
                del event_dict['metadata']
            
            events_data.append(event_dict)
        
        df = pd.DataFrame(events_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(events_data)} events to {output_path}")
    
    def export_metrics_to_json(self, output_path: str):
        """Export metrics summary to JSON file."""
        summary = self.get_metrics_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Exported metrics summary to {output_path}")
    
    def reset_tracking(self):
        """Reset all tracking state."""
        self.events.clear()
        self.player_positions.clear()
        self.puck_positions.clear()
        self.puck_velocity.clear()
        self.possession_changes.clear()
        self.current_frame_data = None
        self.previous_frame_data = None
        self.periods.clear()
        self.current_period = None
        self.period_start_detected = False
        print("Tracking state reset")
