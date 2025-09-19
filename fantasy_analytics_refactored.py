#!/usr/bin/env python3
"""
Unified Fantasy Sports Analytics Suite - Yahoo + ESPN Support
Combines advanced analytics for both Yahoo and ESPN fantasy leagues
"""

import streamlit as st
import requests
import xml.etree.ElementTree as ET
from requests_oauthlib import OAuth2Session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from functools import lru_cache
import os
from typing import Dict, List, Optional, Tuple, Any

# ESPN API import
try:
    from espn_api.football import League as ESPNFootballLeague
    ESPN_AVAILABLE = True
except ImportError:
    ESPN_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration management"""
    
    def __init__(self):
        # Yahoo credentials
        self.CLIENT_ID = os.getenv("YAHOO_CLIENT_ID", "dj0yJmk9NDlOM0xEbjdod2VyJmQ9WVdrOWJ6TmhWM1U1UVhnbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTFi")
        self.CLIENT_SECRET = os.getenv("YAHOO_CLIENT_SECRET", "b9a3f55e22aa865a0783edaa4426a21fb83581d8")
        self.REDIRECT_URI = os.getenv("YAHOO_REDIRECT_URI", "https://heyskip.streamlit.app")
        
        # Yahoo API URLs
        self.AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
        self.TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
        self.FANTASY_BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"
        
        # XML namespace
        self.YAHOO_NS = {"y": "http://fantasysports.yahooapis.com/fantasy/v2/base.rng"}
        
        # Cache settings
        self.CACHE_TTL = 1800  # 30 minutes

config = Config()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        # Platform selection
        "platform": None,  # "yahoo" or "espn"
        
        # Yahoo state
        "token": None,
        "league_key": None,
        "selected_game_code": None,
        "selected_game_name": None,
        
        # ESPN state
        "espn_league": None,
        "espn_connected": False,
        "espn_credentials": None,
        
        # Common state
        "run_analytics": False,
        "debug_mode": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state():
    """Clear all session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(value) if value is not None and value != "" else default
    except (ValueError, TypeError):
        return default

def is_baseball_league(game_code: str = None, game_name: str = None) -> bool:
    """Check if league is baseball"""
    if game_code:
        return game_code.lower() == "mlb"
    if game_name:
        return "baseball" in game_name.lower()
    return False

def is_football_league(game_code: str = None, game_name: str = None) -> bool:
    """Check if league is football"""
    if game_code:
        return game_code.lower() == "nfl"
    if game_name:
        return "football" in game_name.lower()
    return False

def handle_api_error(status_code: int) -> str:
    """Generate user-friendly error messages"""
    if status_code == 401:
        return "Authentication expired. Please refresh to re-authenticate."
    elif status_code == 403:
        return "Access forbidden. Check your league permissions."
    elif status_code == 404:
        return "Data not found. May not be available yet."
    elif status_code == 429:
        return "Rate limit exceeded. Please wait before trying again."
    elif status_code >= 500:
        return "Yahoo's servers are having issues. Try again later."
    else:
        return f"API request failed (Status {status_code})"

# =============================================================================
# YAHOO AUTHENTICATION & API
# =============================================================================

class YahooAuth:
    """Handle Yahoo OAuth and API requests"""
    
    def get_auth_url(self) -> str:
        """Get authorization URL with proper parameters"""
        params = {
            'client_id': config.CLIENT_ID,
            'redirect_uri': config.REDIRECT_URI,
            'response_type': 'code',
            'scope': 'openid'
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        auth_url = f"{config.AUTH_URL}?{query_string}"
        
        return auth_url
    
    def exchange_code_for_token(self, code: str) -> Dict:
        """Exchange auth code for access token"""
        oauth = OAuth2Session(config.CLIENT_ID, redirect_uri=config.REDIRECT_URI)
        return oauth.fetch_token(
            config.TOKEN_URL,
            code=code,
            client_secret=config.CLIENT_SECRET
        )
    
    def make_api_request(self, url: str, token: Dict) -> requests.Response:
        """Make authenticated API request"""
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        return requests.get(url, headers=headers)
    
    def get_user_leagues(self, token: Dict) -> List[Dict]:
        """Get all user leagues"""
        url = f"{config.FANTASY_BASE_URL}/users;use_login=1/games;game_keys=nfl,mlb/leagues"
        response = self.make_api_request(url, token)
        
        if response.status_code != 200:
            raise Exception(handle_api_error(response.status_code))
        
        return self._parse_leagues_xml(response.text)
    
    def _parse_leagues_xml(self, xml_text: str) -> List[Dict]:
        """Parse leagues from XML response"""
        root = ET.fromstring(xml_text)
        leagues = []
        
        for game in root.findall(".//y:game", config.YAHOO_NS):
            game_code_el = game.find("y:code", config.YAHOO_NS)
            game_name_el = game.find("y:name", config.YAHOO_NS)
            
            if game_code_el is None or game_name_el is None:
                continue
                
            game_code = game_code_el.text
            game_name = game_name_el.text
            
            for league in game.findall(".//y:league", config.YAHOO_NS):
                league_name_el = league.find("y:name", config.YAHOO_NS)
                league_key_el = league.find("y:league_key", config.YAHOO_NS)
                
                if league_name_el is None or league_key_el is None:
                    continue
                    
                leagues.append({
                    "game_code": game_code,
                    "game_name": game_name,
                    "league_name": league_name_el.text,
                    "league_key": league_key_el.text,
                })
        
        return leagues

# =============================================================================
# ESPN FUNCTIONALITY
# =============================================================================

def connect_to_espn(league_id: int, year: int, espn_s2: str, swid: str):
    """Connect to ESPN private league"""
    if not espn_s2 or not swid:
        st.error("Both ESPN_S2 and SWID cookies are required for private leagues")
        return None
    
    try:
        league = ESPNFootballLeague(
            league_id=league_id,
            year=year,
            espn_s2=espn_s2.strip(),
            swid=swid.strip()
        )
        
        # Test connection
        teams = league.teams
        if not teams:
            st.error("Connected but no teams found. Check your league ID.")
            return None
            
        st.success(f"Connected! Found {len(teams)} teams.")
        return league
        
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")
        return None

def extract_weekly_points(stats: dict, target_week: int, debug_mode: bool = False, player_name: str = "") -> float:
    """Extract weekly points using the correct ESPN API format"""
    
    if not isinstance(stats, dict):
        return 0.0
    
    # Debug specific players
    debug_this_player = debug_mode and any(name.lower() in player_name.lower() 
                                         for name in ['little', 'cam', 'fields', 'justin', 'irving', 'bucky'])
    
    if debug_this_player:
        st.write(f"**EXTRACTING POINTS FOR {player_name} - Week {target_week}:**")
        st.write("Available week keys in stats:")
        for key, value in stats.items():
            st.write(f"  Week {key}: {value}")
            if isinstance(value, dict) and 'points' in value:
                st.write(f"    → Points: {value['points']}")
    
    # Method 1: Try direct week key first (should work for Week 2+)
    if target_week in stats:
        week_data = stats[target_week]
        if isinstance(week_data, dict) and 'points' in week_data:
            points = safe_float(week_data['points'])
            if debug_this_player:
                st.write(f"  → Found direct week {target_week} with {points} points")
            return points
    
    # Method 2: Try string key version
    str_week = str(target_week)
    if str_week in stats:
        week_data = stats[str_week]
        if isinstance(week_data, dict) and 'points' in week_data:
            points = safe_float(week_data['points'])
            if debug_this_player:
                st.write(f"  → Found direct week '{str_week}' with {points} points")
            return points
    
    # Method 3: For Week 1 (since Key 1 is missing), calculate using season total minus other weeks
    if target_week == 1:
        # Look for season totals (Key 0 in your previous findings)
        season_total = 0.0
        for key in [0, '0', 'season', 'total']:
            if key in stats:
                season_data = stats[key]
                if isinstance(season_data, dict) and 'points' in season_data:
                    season_total = safe_float(season_data['points'])
                    if debug_this_player:
                        st.write(f"  → Found season total from key '{key}': {season_total}")
                    break
        
        # Calculate other weeks total
        other_weeks_total = 0.0
        weeks_found = []
        
        for key, value in stats.items():
            try:
                week_num = int(key) if str(key).isdigit() else None
                if week_num is not None and week_num > 1:  # Week 2, 3, 4, etc.
                    if isinstance(value, dict) and 'points' in value:
                        week_points = safe_float(value['points'])
                        other_weeks_total += week_points
                        weeks_found.append((week_num, week_points))
                        if debug_this_player:
                            st.write(f"    → Adding Week {week_num}: {week_points} points")
            except (ValueError, TypeError):
                continue
        
        # Calculate Week 1 as season total minus other weeks
        week1_points = season_total - other_weeks_total
        
        if debug_this_player:
            st.write(f"  → WEEK 1 CALCULATION: {season_total} (season) - {other_weeks_total} (other weeks) = {week1_points}")
            st.write(f"  → Other weeks found: {weeks_found}")
        
        # Sanity check
        if -5 <= week1_points <= 80:
            return week1_points
        else:
            if debug_this_player:
                st.write(f"  → Week 1 calculation failed sanity check: {week1_points}")
    
    # If all methods fail
    if debug_this_player:
        st.write(f"  → No viable extraction method found, returning 0")
    
    return 0.0

def get_espn_roster_data(league, debug_mode: bool = False):
    """Get current roster data with cumulative stats across all completed weeks"""
    all_data = []
    
    if not league:
        st.error("No league connection")
        return all_data
    
    try:
        # Get current week from league
        current_week = getattr(league, 'current_week', 3)
        completed_weeks = list(range(1, current_week))  # All weeks completed so far
        
        if debug_mode:
            st.write(f"Analyzing weeks {completed_weeks} (current week: {current_week})")
        
        teams = league.teams
        for team in teams:
            try:
                team_name = getattr(team, 'team_name', 'Unknown Team')
                roster = getattr(team, 'roster', [])
                
                for player in roster:
                    try:
                        player_name = getattr(player, 'name', 'Unknown Player')
                        position = getattr(player, 'position', 'UNKNOWN')
                        lineup_slot = getattr(player, 'lineupSlot', 20)
                        
                        # Map position
                        position_map = {'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'K': 'K', 'DEF': 'DEF', 'DST': 'DEF'}
                        position = position_map.get(position, position)
                        
                        # Determine current starter status
                        is_starter = lineup_slot not in [20, 21]
                        
                        # Calculate cumulative points across all completed weeks
                        cumulative_points = 0.0
                        stats = getattr(player, 'stats', {})
                        
                        for week in completed_weeks:
                            week_points = extract_weekly_points(stats, week, debug_mode, player_name)
                            cumulative_points += week_points
                        
                        # Debug for key players
                        if debug_mode and any(name.lower() in player_name.lower() 
                                            for name in ['little', 'cam', 'fields', 'justin', 'irving', 'bucky', 'goff', 'jared']):
                            st.write(f"**{player_name} (Current {position}):**")
                            st.write(f"- Team: {team_name}")
                            st.write(f"- Current lineup slot: {lineup_slot}")
                            st.write(f"- Is current starter: {is_starter}")
                            st.write(f"- Cumulative points (weeks {completed_weeks}): {cumulative_points}")
                            st.write("---")
                        
                        player_data = {
                            "Team": team_name,
                            "Player": player_name,
                            "Position": position,
                            "Is_Starter": is_starter,
                            "Lineup_Slot": lineup_slot,
                            "Weeks_Analyzed": f"Weeks {min(completed_weeks)}-{max(completed_weeks)}",
                            "Points": cumulative_points,
                            "Total_Points": getattr(player, 'total_points', 0.0)
                        }
                        
                        all_data.append(player_data)
                        
                    except Exception as player_error:
                        if debug_mode:
                            st.write(f"Error processing player: {player_error}")
                        continue
                        
            except Exception as team_error:
                if debug_mode:
                    st.write(f"Error processing team: {team_error}")
                continue
                
    except Exception as e:
        st.error(f"Error getting roster data: {e}")
    
    return all_data

# =============================================================================
# YAHOO BASEBALL ANALYTICS (keeping existing working version)
# =============================================================================

class BaseballAnalytics:
    """Baseball fantasy analytics with enhanced z-score and strength analysis"""
    
    def __init__(self, league_key: str, oauth_session: OAuth2Session):
        self.league_key = league_key
        self.oauth = oauth_session
        
        # Stat mappings
        self.STAT_MAP = {
            '7': 'Runs', '12': 'Home Runs', '13': 'RBIs', '16': 'Stolen Bases',
            '55': 'OPS', '42': 'Strikeouts', '26': 'ERA', '27': 'WHIP',
            '83': 'Quality Starts', '89': 'Saves + Holds'
        }
        
        self.WANTED_COLS = list(self.STAT_MAP.values())
        self.RATE_COLS = ['OPS', 'ERA', 'WHIP']
        self.LOWER_BETTER = ['ERA', 'WHIP']
        self.COUNT_COLS = [c for c in self.WANTED_COLS if c not in self.RATE_COLS]
    
    def week_has_data(self, week: int) -> bool:
        """Check if week has data"""
        url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/scoreboard;week={week}"
        
        try:
            resp = self.oauth.get(url)
            if resp.status_code != 200:
                return False
                
            root = ET.fromstring(resp.text)
            matchups = root.findall('.//y:matchup', config.YAHOO_NS)
            
            if not matchups:
                return False
            
            # Check for actual stat values
            for matchup in matchups:
                for team in matchup.findall('.//y:team', config.YAHOO_NS):
                    for stat in team.findall('.//y:stat', config.YAHOO_NS):
                        value_el = stat.find('y:value', config.YAHOO_NS)
                        if value_el is not None and value_el.text and value_el.text.strip():
                            return True
            return False
            
        except Exception:
            return False
    
    def get_available_weeks(self, max_weeks: int = 30) -> List[int]:
        """Get weeks with available data"""
        weeks = []
        empty_streak = 0
        
        for week in range(1, max_weeks + 1):
            if self.week_has_data(week):
                weeks.append(week)
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 3 and weeks:
                    break
        
        return weeks
    
    def get_weekly_stats(self, week: int) -> pd.DataFrame:
        """Get weekly stats for all teams"""
        url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/scoreboard;week={week}"
        
        try:
            response = self.oauth.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            rows = []
            
            for matchup in root.findall('.//y:matchup', config.YAHOO_NS):
                for team in matchup.findall('.//y:team', config.YAHOO_NS):
                    name_el = team.find('y:name', config.YAHOO_NS)
                    if name_el is None or not name_el.text:
                        continue
                    
                    row = {"Team": name_el.text}
                    
                    for stat in team.findall('.//y:stat', config.YAHOO_NS):
                        stat_id_el = stat.find('y:stat_id', config.YAHOO_NS)
                        value_el = stat.find('y:value', config.YAHOO_NS)
                        
                        if stat_id_el is None or value_el is None:
                            continue
                        
                        stat_name = self.STAT_MAP.get(stat_id_el.text)
                        if not stat_name:
                            continue
                        
                        row[stat_name] = safe_float(value_el.text, np.nan)
                    
                    rows.append(row)
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            df.set_index("Team", inplace=True)
            
            # Ensure all wanted columns exist
            for col in self.WANTED_COLS:
                if col not in df.columns:
                    df[col] = np.nan
                    
            return df[self.WANTED_COLS]
            
        except Exception as e:
            st.error(f"Error fetching week {week} data: {e}")
            return pd.DataFrame()
    
    def compute_weekly_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores for weekly performance"""
        z_scores = df.copy()
        
        for col in df.columns:
            std = df[col].std(ddof=0)
            if std == 0 or pd.isna(std):
                z_scores[col] = 0
            else:
                z_scores[col] = (df[col] - df[col].mean()) / std
        
        # Invert z-scores for "lower is better" stats
        for col in self.LOWER_BETTER:
            if col in z_scores.columns:
                z_scores[col] *= -1
        
        # Clip extreme values
        return z_scores.clip(lower=-3, upper=3)
    
    def compute_team_strength(self, weeks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute team strength based on z-scores across multiple weeks"""
        all_z_scores = []
        
        for week in weeks:
            df = self.get_weekly_stats(week)
            if df is None or df.empty:
                continue
            
            z_scores = self.compute_weekly_z_scores(df)
            z_scores['Week'] = week
            all_z_scores.append(z_scores)
        
        if not all_z_scores:
            return pd.DataFrame(), pd.DataFrame()
        
        # Combine all weeks
        combined_z = pd.concat(all_z_scores)
        
        # Calculate mean z-scores (overall strength)
        strength_scores = combined_z.groupby(level=0)[self.WANTED_COLS].mean()
        
        # Calculate consistency (lower std = more consistent)
        consistency_scores = combined_z.groupby(level=0)[self.WANTED_COLS].std()
        consistency_scores = consistency_scores.fillna(0)
        
        return strength_scores, consistency_scores
    
    def compute_cumulative_stats(self, weeks: List[int]) -> Tuple[pd.DataFrame, List[str]]:
        """Compute cumulative stats across weeks"""
        teams = None
        rate_means = None
        rate_counts = None
        count_sums = None
        missing_weeks = []
        
        for week in weeks:
            try:
                df = self.get_weekly_stats(week)
                if df is None or df.empty:
                    missing_weeks.append(f"Week {week}: No data")
                    continue
                
                if teams is None:
                    teams = df.index.tolist()
                
                if list(df.index) != teams:
                    df = df.reindex(teams)
                
                df_rate = df[self.RATE_COLS].copy()
                df_count = df[self.COUNT_COLS].copy()
                
                if rate_means is None:
                    rate_means = pd.DataFrame(0.0, index=df.index, columns=self.RATE_COLS)
                    rate_counts = pd.DataFrame(0, index=df.index, columns=self.RATE_COLS)
                if count_sums is None:
                    count_sums = pd.DataFrame(0.0, index=df.index, columns=self.COUNT_COLS)
                
                # Update running averages for rate stats
                valid_mask = df_rate.notna()
                rate_means = rate_means.where(
                    ~valid_mask,
                    (rate_means * rate_counts + df_rate).div(rate_counts + 1)
                )
                rate_counts = rate_counts + valid_mask.astype(int)
                
                # Sum count stats
                count_sums = count_sums.add(df_count.fillna(0), fill_value=0)
                
            except Exception as e:
                missing_weeks.append(f"Week {week}: {str(e)}")
        
        if teams is None:
            return pd.DataFrame(), missing_weeks
        
        rate_final = rate_means.where(rate_counts > 0, np.nan)
        df_total = pd.concat([count_sums, rate_final], axis=1)[self.WANTED_COLS]
        
        return df_total, missing_weeks
    
    def create_heatmap(self, df_total: pd.DataFrame) -> plt.Figure:
        """Create rankings heatmap"""
        # Calculate rankings
        ranked_normal = df_total.drop(columns=self.LOWER_BETTER, errors='ignore').rank(
            ascending=False, method='min'
        )
        ranked_lower = df_total[self.LOWER_BETTER].rank(ascending=True, method='min')
        ranked_df = pd.concat([ranked_normal, ranked_lower], axis=1)
        ranked_df = ranked_df[df_total.columns]
        
        # Color function
        def cell_color(rank_val: float):
            if pd.isna(rank_val):
                return 'white'
            elif rank_val <= 3:
                return 'lightgreen'
            elif rank_val <= 7:
                return 'khaki'
            else:
                return 'lightcoral'
        
        colors = ranked_df.applymap(cell_color)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw heatmap
        for i in range(ranked_df.shape[0]):
            for j in range(ranked_df.shape[1]):
                rank_val = ranked_df.iloc[i, j]
                stat_val = df_total.iloc[i, j]
                
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors.iloc[i, j]))
                
                if pd.notna(rank_val) and pd.notna(stat_val):
                    if ranked_df.columns[j] in self.RATE_COLS:
                        pretty_total = round(stat_val, 3)
                    else:
                        pretty_total = round(stat_val, 0)
                    
                    ax.text(j + 0.5, i + 0.28, f"Rk {int(rank_val)}", 
                           ha='center', va='center', fontsize=8, weight='bold')
                    ax.text(j + 0.5, i + 0.70, f"{pretty_total}", 
                           ha='center', va='center', fontsize=8)
                else:
                    ax.text(j + 0.5, i + 0.50, "—", ha='center', va='center', fontsize=8)
        
        # Labels and formatting
        ax.set_xticks([x + 0.5 for x in range(ranked_df.shape[1])])
        ax.set_xticklabels(ranked_df.columns, rotation=30, ha='right', fontsize=10)
        ax.set_yticks([y + 0.5 for y in range(ranked_df.shape[0])])
        ax.set_yticklabels(ranked_df.index, fontsize=10)
        ax.set_xlim(0, ranked_df.shape[1])
        ax.set_ylim(0, ranked_df.shape[0])
        ax.invert_yaxis()
        ax.set_title("Team Rankings by Stat Category", fontsize=14, weight='bold')
        ax.tick_params(left=False, bottom=False)
        
        plt.grid(False)
        plt.tight_layout()
        
        return fig

# =============================================================================
# UNIFIED HEATMAP FUNCTION
# =============================================================================

def create_unified_heatmap(data, starters_only: bool = False, debug_mode: bool = False):
    """Create positional heatmap with debugging for both Yahoo and ESPN"""
    if not data:
        return None, pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # DEBUG: Show what we're working with
    if debug_mode:
        st.write("**DEBUG - Raw data sample:**")
        sample_df = df[df['Player'].str.contains('Fields|Little|Goff|Jared', case=False, na=False)].copy()
        if not sample_df.empty:
            st.dataframe(sample_df[['Team', 'Player', 'Position', 'Is_Starter', 'Points']])
    
    if starters_only:
        if debug_mode:
            st.write(f"**DEBUG - Filtering for starters only. Before: {len(df)} players**")
        df = df[df['Is_Starter'] == True]
        if debug_mode:
            st.write(f"**DEBUG - After starter filter: {len(df)} players**")
        
        if df.empty:
            st.warning("No starter data found")
            return None, pd.DataFrame()
    
    # Aggregate by team and position
    totals = df.groupby(["Team", "Position"])["Points"].sum().reset_index()
    
    # DEBUG: Show totals after aggregation
    if debug_mode:
        st.write("**DEBUG - After aggregation (sample):**")
        st.dataframe(totals.head(10))
    
    pivot = totals.pivot(index="Team", columns="Position", values="Points").fillna(0)
    
    # Sort by total points
    team_totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot_sorted = pivot.loc[team_totals.index]
    
    # Create rankings
    rankings = {}
    for col in pivot_sorted.columns:
        rankings[col] = pivot_sorted[col].rank(ascending=False, method='min')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_sorted.columns) * 1.2), max(6, len(pivot_sorted.index) * 0.6)))
    
    def get_color_by_rank(rank, num_teams):
        if rank <= 3:
            return 'lightgreen'
        elif rank <= num_teams - 3:
            return 'khaki'
        else:
            return 'lightcoral'
    
    num_teams = len(pivot_sorted.index)
    
    # Draw heatmap
    for i in range(len(pivot_sorted.index)):
        for j in range(len(pivot_sorted.columns)):
            value = pivot_sorted.iloc[i, j]
            position = pivot_sorted.columns[j]
            rank = rankings[position].iloc[i]
            
            color = get_color_by_rank(rank, num_teams)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, edgecolor='white', linewidth=1))
            
            if value > 0:
                ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha='center', va='center', 
                       color='black', weight='bold', fontsize=9)
            else:
                ax.text(j + 0.5, i + 0.5, "0.0", ha='center', va='center', 
                       color='gray', fontsize=8)
    
    # Set properties
    ax.set_xlim(0, len(pivot_sorted.columns))
    ax.set_ylim(0, len(pivot_sorted.index))
    ax.invert_yaxis()
    
    # Labels
    ax.set_xticks([x + 0.5 for x in range(len(pivot_sorted.columns))])
    ax.set_xticklabels(pivot_sorted.columns, fontsize=11, weight='bold')
    ax.set_yticks([y + 0.5 for y in range(len(pivot_sorted.index))])
    ax.set_yticklabels(pivot_sorted.index, fontsize=10)
    
    ax.tick_params(left=False, bottom=False)
    
    title = "Starting Lineup Points by Position" if starters_only else "Total Roster Points by Position"
    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.xlabel("Position", fontsize=12, weight='bold')
    plt.ylabel("Team", fontsize=12, weight='bold')
    
    # Legend
    legend_elements = [
        Patch(facecolor='lightgreen', label='Top 3 Teams'),
        Patch(facecolor='khaki', label='Middle Teams'), 
        Patch(facecolor='lightcoral', label='Bottom 3 Teams')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    return fig, pivot_sorted

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_landing_page():
    """Render enhanced landing page with platform selection"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .logo-section {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%, #f093fb 100%);
        border-radius: 20px;
        margin: 1rem 0 3rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    .main-header h1, .main-header h3 {
        position: relative;
        z-index: 1;
    }
    .platform-card {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.4s ease;
        border: 2px solid transparent;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .platform-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.6s;
    }
    .platform-card:hover::before {
        left: 100%;
    }
    .platform-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: #667eea;
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.2);
    }
    .platform-icon {
        font-size: 72px;
        margin-bottom: 1.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .platform-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 50px;
        text-decoration: none;
        font-weight: bold;
        font-size: 16px;
        display: inline-block;
        margin: 1.5rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .platform-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.2), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .platform-button:hover::before {
        opacity: 1;
    }
    .platform-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .feature-highlight {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .feature-highlight:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateX(5px);
    }
    .stats-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Large centered logo section
    st.markdown('<div class="logo-section">', unsafe_allow_html=True)
    
    # Try to display your custom logo - much larger and centered
    try:
        # Center the logo with columns
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image("skipper_logo.png", width=400)
    except:
        # Fallback if logo file not found - larger and more prominent
        st.markdown("""
        <div style="
            width: 350px; 
            height: 200px; 
            background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%); 
            border-radius: 20px; 
            display: flex; 
            align-items: center; 
            justify-content: center;
            box-shadow: 0 12px 24px rgba(0,0,0,0.2);
            margin: 2rem auto;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -50px;
                right: -50px;
                width: 100px;
                height: 100px;
                background: rgba(255,255,255,0.1);
                border-radius: 50%;
            "></div>
            <div style="text-align: center; z-index: 1;">
                <div style="font-size: 64px; margin-bottom: 10px;">🏆</div>
                <div style="color: white; font-weight: bold; font-size: 28px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">SKIPPER</div>
                <div style="color: white; font-size: 16px; opacity: 0.9; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">Analytics</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 3.5rem; font-weight: 700; letter-spacing: -2px;">Skipper Analytics</h1>
        <h3 style="margin: 1rem 0 0 0; opacity: 0.95; font-weight: 400; font-size: 1.3rem;">Advanced Fantasy Sports Intelligence</h3>
        <div style="margin-top: 1rem;">
            <span class="stats-badge">⚾ Baseball</span>
            <span class="stats-badge">🏈 Football</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Platform selection with enhanced cards
    st.markdown("## Choose Your Fantasy Platform")
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;'>Select your fantasy platform to unlock advanced analytics and insights</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="platform-card">
            <div class="platform-icon">🟣</div>
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;">Yahoo Fantasy</h3>
            <p style="color: #6c757d; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">Advanced analytics for Yahoo Fantasy Baseball and Football leagues. Z-score analysis, team strength metrics, and comprehensive performance tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use regular Streamlit elements for features
        st.markdown("**⚾ Baseball:** Category rankings, Z-scores, consistency analysis")
        st.markdown("**🏈 Football:** Positional heatmaps, starter analysis") 
        st.markdown("**🔐 Security:** Secure OAuth authentication")
        st.markdown("**📈 Trends:** Multi-week trend analysis")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Connect Yahoo Fantasy", key="yahoo_btn", use_container_width=True):
            st.session_state.platform = "yahoo"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="platform-card">
            <div class="platform-icon">🔴</div>
            <h3 style="color: #2c3e50; margin-bottom: 1rem; font-size: 1.8rem;">ESPN Fantasy</h3>
            <p style="color: #6c757d; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;">Deep analytics for ESPN private leagues. Season-to-date performance tracking with precise weekly point extraction and roster depth analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use regular Streamlit elements for features
        st.markdown("**🏈 Football:** Current roster season-to-date analysis")
        st.markdown("**📊 Extraction:** Weekly point extraction & cumulative tracking")
        st.markdown("**🎯 Analysis:** Starter vs bench performance breakdown")
        st.markdown("**🔧 Debug:** Advanced debugging & data validation")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not ESPN_AVAILABLE:
            st.error("📦 ESPN API not installed. Run: `pip install espn-api`")
        else:
            if st.button("🚀 Connect ESPN Fantasy", key="espn_btn", use_container_width=True):
                st.session_state.platform = "espn"
                st.rerun()
    
    # Features overview
    with st.expander("🔍 What Makes Skipper Analytics Different?", expanded=False):
        st.markdown("""
        ### **Advanced Analytics Beyond Basic Apps**
        
        **🎯 Yahoo Fantasy Specialties:**
        - **Z-Score Analysis**: See how many standard deviations above/below league average each team performs
        - **Team Strength Metrics**: Comprehensive performance tracking across all categories
        - **Consistency Analysis**: Identify which teams perform reliably vs inconsistently
        - **Category Rankings**: Visual heatmaps showing team strengths and weaknesses
        
        **🎯 ESPN Fantasy Specialties:**
        - **Precise Weekly Extraction**: Correctly handles ESPN's API quirks for accurate weekly points
        - **Season-to-Date Analysis**: Current roster performance across all completed weeks
        - **Roster Depth Analysis**: Compare starter performance vs bench strength
        - **Debug Mode**: Deep dive into data extraction for validation
        
        **🔐 Security & Privacy:**
        - Yahoo: Secure OAuth 2.0 (never stores passwords)
        - ESPN: Local cookie authentication (data stays private)
        - No data stored permanently
        - Read-only access to your leagues
        """)

def setup_page():
    """Configure Streamlit page"""
    try:
        st.set_page_config(
            page_title="Skipper Fantasy Analytics",
            page_icon="🏆",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        pass

def render_sidebar():
    """Render sidebar with platform-specific controls"""
    with st.sidebar:
        st.markdown("🏆 **Skipper Analytics**")
        st.markdown("---")
        
        # Platform switching
        if st.session_state.platform:
            st.write(f"**Platform:** {st.session_state.platform.title()}")
            if st.button("Switch Platform"):
                clear_session_state()
                st.rerun()
            st.markdown("---")
        
        # Common controls
        st.header("🛠️ Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🚪 Reset", use_container_width=True):
                clear_session_state()
                st.rerun()
        
        # Debug mode
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode", value=st.session_state.debug_mode)
        
        st.markdown("---")
        
        st.header("About")
        st.markdown("""
        **Advanced Fantasy Analytics**
        
        **Supported Platforms:**
        - Yahoo Fantasy (MLB/NFL)
        - ESPN Fantasy (NFL)
        
        **Key Features:**
        - Team strength analysis
        - Positional breakdowns
        - Performance heatmaps
        - Weekly trend tracking
        """)

# =============================================================================
# YAHOO AUTHENTICATION HANDLER
# =============================================================================

def handle_yahoo_authentication():
    """Enhanced authentication handler for Yahoo"""
    auth = YahooAuth()
    
    # Check if we have a token and if it's still valid
    if st.session_state.token:
        try:
            test_url = f"{config.FANTASY_BASE_URL}/users;use_login=1"
            response = auth.make_api_request(test_url, st.session_state.token)
            if response.status_code == 200:
                return True, st.session_state.token, None
            else:
                st.sidebar.warning(f"Token invalid (Status: {response.status_code})")
        except Exception as e:
            st.sidebar.error(f"Token test failed: {e}")
    
    # Check for auth code in URL parameters
    query_params = st.query_params.to_dict()
    
    if "code" in query_params:
        try:
            code = query_params["code"]
            st.sidebar.info(f"Processing auth code: {code[:10]}...")
            
            # Exchange code for token
            token = auth.exchange_code_for_token(code)
            st.session_state.token = token
            st.sidebar.success("Token exchange successful!")
            st.rerun()
        except Exception as e:
            error_msg = f"Token exchange failed: {str(e)}"
            st.sidebar.error(error_msg)
            return False, None, error_msg
    
    # If we get here, we need to authenticate
    if not st.session_state.token:
        try:
            auth_url = auth.get_auth_url()
            return False, None, auth_url
        except Exception as e:
            error_msg = f"Auth URL generation failed: {str(e)}"
            st.sidebar.error(error_msg)
            return False, None, error_msg
    
    return True, st.session_state.token, None

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

def render_yahoo_interface():
    """Render Yahoo Fantasy interface"""
    # Handle authentication
    is_authenticated, token, auth_result = handle_yahoo_authentication()
    
    if not is_authenticated:
        if auth_result and auth_result.startswith("http"):
            st.info("Please authenticate with Yahoo to access your fantasy leagues.")
            st.markdown(f"""
            <div style="text-align: center; margin: 2rem 0;">
                <a href="{auth_result}" target="_blank" style="
                    background: linear-gradient(135deg, #1f77b4, #2e8b57);
                    color: white;
                    padding: 1rem 2rem;
                    border-radius: 50px;
                    text-decoration: none;
                    font-weight: bold;
                    font-size: 18px;
                ">🔐 Authenticate with Yahoo</a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Authentication failed: {auth_result}")
        return
    
    st.success("✅ Successfully authenticated with Yahoo!")
    
    # Get leagues
    auth = YahooAuth()
    try:
        leagues = auth.get_user_leagues(token)
    except Exception as e:
        st.error(f"Failed to load leagues: {str(e)}")
        return
    
    if not leagues:
        st.warning("No fantasy leagues found for your account.")
        return
    
    # League selection
    st.subheader("Your Yahoo Fantasy Leagues")
    
    game_choices = sorted(set(league["game_name"] for league in leagues))
    game_choice = st.selectbox("Choose a sport:", ["Select..."] + game_choices)
    
    if game_choice == "Select...":
        st.info("Select a sport to view your leagues.")
        return
    
    leagues_for_game = [l for l in leagues if l["game_name"] == game_choice]
    league_names = [l["league_name"] for l in leagues_for_game]
    league_choice = st.selectbox("Choose a league:", ["Select..."] + league_names)
    
    if league_choice == "Select...":
        st.info("Select a league to begin analysis.")
        return
    
    # Get selected league info
    selected_league = next(l for l in leagues_for_game if l["league_name"] == league_choice)
    league_key = selected_league["league_key"]
    game_code = selected_league["game_code"]
    game_name = selected_league["game_name"]
    
    # Create OAuth session for API calls
    oauth_session = OAuth2Session(config.CLIENT_ID, token=token)
    
    st.markdown("---")
    
    # Render appropriate analytics
    if is_baseball_league(game_code, game_name):
        render_yahoo_baseball_analytics(league_key, oauth_session)
    elif is_football_league(game_code, game_name):
        render_yahoo_football_analytics(league_key, oauth_session)
    else:
        st.error(f"Unsupported sport: {game_name}")

def render_espn_interface():
    """Render ESPN Fantasy interface"""
    st.subheader("ESPN Fantasy Analytics")
    
    # Connection status
    if not st.session_state.espn_connected:
        st.info("Connect to your ESPN private league to begin analysis.")
        
        with st.expander("How to get ESPN cookies", expanded=False):
            st.markdown("""
            **Steps to get your ESPN cookies:**
            1. Open ESPN Fantasy while logged in
            2. Press F12 to open Developer Tools
            3. Go to Application → Storage → Cookies → espn.com
            4. Find and copy:
               - **ESPN_S2**: Long string starting with "AE"
               - **SWID**: Format like `{12345678-1234-1234-1234-123456789012}`
            """)
        
        with st.form("espn_connect"):
            league_id = st.number_input("League ID", min_value=1, help="From your ESPN league URL")
            year = st.number_input("Year", min_value=2020, max_value=2025, value=2025)
            
            espn_s2 = st.text_area("ESPN_S2 Cookie", height=80, placeholder="Paste the long ESPN_S2 value here")
            swid = st.text_input("SWID Cookie", placeholder="{12345678-1234-1234-1234-123456789012}")
            
            if st.form_submit_button("Connect to League"):
                if league_id and espn_s2 and swid:
                    with st.spinner("Connecting to ESPN..."):
                        league = connect_to_espn(league_id, year, espn_s2, swid)
                        if league:
                            st.session_state.espn_league = league
                            st.session_state.espn_connected = True
                            st.session_state.espn_credentials = {
                                'league_id': league_id,
                                'year': year,
                                'espn_s2': espn_s2,
                                'swid': swid
                            }
                            st.rerun()
                else:
                    st.error("Please fill in all fields")
    else:
        # Connected - show analytics
        league = st.session_state.espn_league
        if league:
            # Show league info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("League", getattr(league, 'name', 'Unknown'))
            with col2:
                st.metric("Year", st.session_state.espn_credentials['year'])
            with col3:
                st.metric("Teams", len(league.teams))
            
            render_espn_analytics(league)

def render_yahoo_baseball_analytics(league_key: str, oauth_session: OAuth2Session):
    """Render Yahoo baseball analytics"""
    analytics = BaseballAnalytics(league_key, oauth_session)
    
    st.success("⚾ Baseball analytics loaded successfully!")
    
    # Get available weeks
    available_weeks = analytics.get_available_weeks()
    if not available_weeks:
        st.warning("No weekly data found yet. Check back once the season starts.")
        return
    
    # Timeframe selection
    timeframe = st.radio(
        "Show data for:", 
        ["Last 5 weeks", "Last 10 weeks", "Full season"], 
        index=2
    )
    
    # Filter weeks
    if timeframe == "Last 5 weeks":
        selected_weeks = available_weeks[-5:]
    elif timeframe == "Last 10 weeks":
        selected_weeks = available_weeks[-10:]
    else:
        selected_weeks = available_weeks
    
    st.caption(f"Weeks included: {selected_weeks}")
    
    # Progress indicator
    with st.spinner("Loading baseball analytics..."):
        # Compute stats
        df_total, missing_weeks = analytics.compute_cumulative_stats(selected_weeks)
        
        if df_total.empty:
            st.warning("No data available for selected timeframe.")
            return
        
        # Compute strength analysis
        strength_scores, consistency_scores = analytics.compute_team_strength(selected_weeks)
    
    # Show any data issues
    if missing_weeks:
        with st.expander("Data Issues"):
            for issue in missing_weeks:
                st.warning(issue)
    
    # Create tabs
    tab_heatmap, tab_strength, tab_data = st.tabs([
        "Rankings Heatmap", 
        "Team Strength", 
        "Raw Data"
    ])
    
    with tab_heatmap:
        st.header("Team Rankings by Stat Category")
        
        fig = analytics.create_heatmap(df_total)
        st.pyplot(fig, use_container_width=True)
        
        st.markdown(
            "**Note:** OPS, ERA, and WHIP are averaged; other stats are totals."
        )
    
    with tab_strength:
        st.header("Team Strength Analysis")
        
        if not strength_scores.empty:
            # Overall strength score
            overall_strength = strength_scores.mean(axis=1).sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Overall Team Strength")
                for i, (team, score) in enumerate(overall_strength.head(10).items()):
                    if i < 3:
                        medal = ["🥇", "🥈", "🥉"][i]
                        st.write(f"{medal} **{team}**: {score:.2f}")
                    else:
                        st.write(f"{i+1}. {team}: {score:.2f}")
            
            with col2:
                st.subheader("Category Leaders")
                for category in analytics.WANTED_COLS:
                    if category in strength_scores.columns:
                        leader = strength_scores[category].idxmax()
                        score = strength_scores[category].max()
                        st.write(f"**{category}**: {leader} ({score:.2f})")
        else:
            st.warning("Not enough data for strength analysis.")
    
    with tab_data:
        st.header("Raw Statistics")
        st.dataframe(df_total.round(3), use_container_width=True)

def render_yahoo_football_analytics(league_key: str, oauth_session: OAuth2Session):
    """Render Yahoo football analytics"""
    st.success("🏈 Football analytics loaded successfully!")
    st.info("Yahoo Football analytics coming soon! Currently focused on ESPN implementation.")

def render_espn_analytics(league):
    """Render ESPN analytics interface"""
    
    # Show approach explanation
    with st.expander("📊 Current Roster Season-to-Date Analytics", expanded=True):
        current_week = getattr(league, 'current_week', 3)
        completed_weeks = list(range(1, current_week))
        
        st.markdown(f"""
        **What This Tool Shows:**
        This heatmap shows the **current roster's cumulative fantasy performance** across all completed weeks this season.
        
        **Current Season Status:**
        - **Weeks {min(completed_weeks)}-{max(completed_weeks)}**: Completed
        - **Week {current_week}**: Currently in progress
        - **Analysis**: Cumulative points from Weeks {min(completed_weeks)}-{max(completed_weeks)}
        
        **Key Points:**
        - Uses **current roster composition** (who's on teams right now)
        - Shows **season-to-date cumulative fantasy points** 
        - "Current Starters" = players currently in starting lineup positions
        - "All Players" = entire current roster including bench
        
        **Perfect For:**
        - Analyzing current team strengths/weaknesses
        - Trade target identification  
        - Understanding positional depth
        - Season performance trends
        """)
    
    # Get league info for display
    current_week = getattr(league, 'current_week', 3)
    completed_weeks = list(range(1, current_week))
    
    st.write(f"**Season Progress:** Currently Week {current_week} | Analyzing cumulative data from Weeks {min(completed_weeks)}-{max(completed_weeks)}")
    
    # Analysis type
    analysis_type = st.radio(
        "Analysis Type:",
        ["All Players (Full Roster)", "Current Starters Only"],
        help="All Players shows entire roster depth. Current Starters shows only players in starting positions."
    )
    
    starters_only = analysis_type.startswith("Current Starters")
    
    # Run analysis
    if st.button("🚀 Run Analysis"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Processing season-to-date data (Weeks {min(completed_weeks)}-{max(completed_weeks)})...")
        all_data = get_espn_roster_data(league, st.session_state.debug_mode)
        progress.progress(1.0)
        
        status_text.text("Creating visualization...")
        
        if all_data:
            # Show debug summary if enabled
            if st.session_state.debug_mode:
                df_debug = pd.DataFrame(all_data)
                debug_players = df_debug[df_debug['Player'].str.contains('Little|Cam|Fields|Justin|Irving|Bucky|Goff|Jared', case=False, na=False, regex=True)]
                if not debug_players.empty:
                    st.write("**🐛 Debug Summary - Key Players:**")
                    debug_display = debug_players[['Player', 'Position', 'Points', 'Total_Points']]
                    st.dataframe(debug_display, use_container_width=True)
            
            # Validation check
            df_all = pd.DataFrame(all_data)
            total_points = df_all['Points'].sum()
            
            st.write(f"**📊 Season-to-Date Analysis (Weeks {min(completed_weeks)}-{max(completed_weeks)}):**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Cumulative Points", f"{total_points:.1f}")
            with col2:
                st.metric("Players Analyzed", len(df_all))
            
            # Create heatmap
            fig, pivot = create_unified_heatmap(all_data, starters_only, st.session_state.debug_mode)
            
            if fig:
                st.pyplot(fig, use_container_width=True)
                
                # Summary metrics
                df = pd.DataFrame(all_data)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Players", len(df))
                with col2:
                    st.metric("Total Points", f"{df['Points'].sum():.1f}")
                with col3:
                    starters = df[df['Is_Starter'] == True]
                    st.metric("Current Starter Points", f"{starters['Points'].sum():.1f}")
                with col4:
                    bench = df[df['Is_Starter'] == False]
                    st.metric("Bench Points", f"{bench['Points'].sum():.1f}")
                
                # Data table
                with st.expander("📋 View Data Table"):
                    st.dataframe(pivot.style.format("{:.1f}"), use_container_width=True)
                    
                # Top performers
                with st.expander("🏆 Top Season-to-Date Performers"):
                    top_performers = df.nlargest(10, 'Points')[['Player', 'Team', 'Position', 'Points', 'Is_Starter']]
                    st.dataframe(top_performers, use_container_width=True)
            else:
                st.error("Failed to create visualization")
                
            # Clear status
            status_text.empty()
            progress_bar.empty()
        else:
            st.warning("No data found")
            status_text.empty()
            progress_bar.empty()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application"""
    # Setup
    setup_page()
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    try:
        # Platform selection or interface
        if not st.session_state.platform:
            render_landing_page()
        elif st.session_state.platform == "yahoo":
            render_yahoo_interface()
        elif st.session_state.platform == "espn":
            if not ESPN_AVAILABLE:
                st.error("ESPN API not installed. Run: `pip install espn-api`")
                if st.button("Back to Platform Selection"):
                    st.session_state.platform = None
                    st.rerun()
            else:
                render_espn_interface()
        else:
            st.error("Unknown platform selected")
            st.session_state.platform = None
            st.rerun()
    
    except Exception as e:
        st.error("An unexpected error occurred.")
        with st.expander("Error Details"):
            st.exception(e)
        if st.button("Restart Application"):
            clear_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
