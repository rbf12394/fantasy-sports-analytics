#!/usr/bin/env python3
"""
Fantasy Sports Analytics Suite - Complete Working Version
Uses the player stats API endpoint for accurate weekly fantasy points
"""

import streamlit as st
import requests
import xml.etree.ElementTree as ET
from requests_oauthlib import OAuth2Session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache
import os
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration management"""
    
    def __init__(self):
        # Get credentials from environment variables (more secure)
        self.CLIENT_ID = os.getenv("YAHOO_CLIENT_ID", "dj0yJmk9NDlOM0xEbjdod2VyJmQ9WVdrOWJ6TmhWM1U1UVhnbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTFi")
        self.CLIENT_SECRET = os.getenv("YAHOO_CLIENT_SECRET", "b9a3f55e22aa865a0783edaa4426a21fb83581d8")
        self.REDIRECT_URI = os.getenv("YAHOO_REDIRECT_URI", "https://ed85bde5dfda.ngrok-free.app")
        
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
        "token": None,
        "run_analytics": False,
        "league_key": None,
        "selected_game_code": None,
        "selected_game_name": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_session_state():
    """Clear all session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

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

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    try:
        return float(value) if value is not None and value != "" else default
    except (ValueError, TypeError):
        return default

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
# AUTHENTICATION & LEAGUE MANAGEMENT
# =============================================================================

class YahooAuth:
    """Handle Yahoo OAuth and API requests"""
    
    def get_auth_url(self) -> str:
        """Get authorization URL"""
        oauth = OAuth2Session(config.CLIENT_ID, redirect_uri=config.REDIRECT_URI)
        auth_url, state = oauth.authorization_url(config.AUTH_URL)
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
# BASEBALL ANALYTICS (keeping existing working version)
# =============================================================================

class BaseballAnalytics:
    """Baseball fantasy analytics with enhanced z-score and strength analysis"""
    
    def __init__(self, league_key: str, oauth_session: OAuth2Session):
        self.league_key = league_key
        self.oauth = oauth_session
        
        # Stat mappings - moved to __init__ to fix scoping issue
        self.STAT_MAP = {
            '7': 'Runs', '12': 'Home Runs', '13': 'RBIs', '16': 'Stolen Bases',
            '55': 'OPS', '42': 'Strikeouts', '26': 'ERA', '27': 'WHIP',
            '83': 'Quality Starts', '89': 'Saves + Holds'
        }
        
        self.WANTED_COLS = list(self.STAT_MAP.values())
        self.RATE_COLS = ['OPS', 'ERA', 'WHIP']
        self.LOWER_BETTER = ['ERA', 'WHIP']
        self.COUNT_COLS = [c for c in self.WANTED_COLS if c not in self.RATE_COLS]
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def week_has_data(_self, week: int) -> bool:
        """Check if week has data"""
        url = f"{config.FANTASY_BASE_URL}/league/{_self.league_key}/scoreboard;week={week}"
        
        try:
            resp = _self.oauth.get(url)
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
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_available_weeks(_self, max_weeks: int = 30) -> List[int]:
        """Get weeks with available data"""
        weeks = []
        empty_streak = 0
        
        for week in range(1, max_weeks + 1):
            if _self.week_has_data(week):
                weeks.append(week)
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 3 and weeks:
                    break
        
        return weeks
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_weekly_stats(_self, week: int) -> pd.DataFrame:
        """Get weekly stats for all teams"""
        url = f"{config.FANTASY_BASE_URL}/league/{_self.league_key}/scoreboard;week={week}"
        
        try:
            response = _self.oauth.get(url)
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
                        
                        stat_name = _self.STAT_MAP.get(stat_id_el.text)
                        if not stat_name:
                            continue
                        
                        row[stat_name] = safe_float(value_el.text, np.nan)
                    
                    rows.append(row)
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            df.set_index("Team", inplace=True)
            
            # Ensure all wanted columns exist
            for col in _self.WANTED_COLS:
                if col not in df.columns:
                    df[col] = np.nan
                    
            return df[_self.WANTED_COLS]
            
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
    
    def create_z_score_heatmap(self, strength_scores: pd.DataFrame) -> plt.Figure:
        """Create z-score strength heatmap"""
        # Color function for z-scores
        def z_color(z_val: float):
            if pd.isna(z_val):
                return 'white'
            elif z_val >= 1.5:
                return 'darkgreen'
            elif z_val >= 0.5:
                return 'lightgreen'
            elif z_val >= -0.5:
                return 'lightyellow'
            elif z_val >= -1.5:
                return 'lightcoral'
            else:
                return 'darkred'
        
        colors = strength_scores.applymap(z_color)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw heatmap
        for i in range(strength_scores.shape[0]):
            for j in range(strength_scores.shape[1]):
                z_val = strength_scores.iloc[i, j]
                
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=colors.iloc[i, j]))
                
                if pd.notna(z_val):
                    # Choose text color based on background
                    text_color = 'white' if abs(z_val) > 1 else 'black'
                    ax.text(j + 0.5, i + 0.5, f"{z_val:.2f}", 
                           ha='center', va='center', fontsize=9, weight='bold', 
                           color=text_color)
                else:
                    ax.text(j + 0.5, i + 0.5, "—", ha='center', va='center', fontsize=9)
        
        # Labels and formatting
        ax.set_xticks([x + 0.5 for x in range(strength_scores.shape[1])])
        ax.set_xticklabels(strength_scores.columns, rotation=30, ha='right', fontsize=10)
        ax.set_yticks([y + 0.5 for y in range(strength_scores.shape[0])])
        ax.set_yticklabels(strength_scores.index, fontsize=10)
        ax.set_xlim(0, strength_scores.shape[1])
        ax.set_ylim(0, strength_scores.shape[0])
        ax.invert_yaxis()
        ax.set_title("Team Strength (Z-Scores)", fontsize=14, weight='bold')
        ax.tick_params(left=False, bottom=False)
        
        plt.grid(False)
        plt.tight_layout()
        
        return fig

# =============================================================================
# FOOTBALL ANALYTICS USING PLAYER STATS API
# =============================================================================

class FootballAnalytics:
    """Football fantasy analytics using the player stats endpoint approach"""
    
    def __init__(self, league_key: str, oauth_session: OAuth2Session):
        self.league_key = league_key
        self.oauth = oauth_session
        self.POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']
        self.POSITION_MAPPING = {
            'QB': 'QB', 'RB': 'RB', 'WR': 'WR', 'TE': 'TE', 'K': 'K',
            'DEF': 'DEF', 'DST': 'DEF', 'D/ST': 'DEF'
        }
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def week_has_data(_self, week: int) -> bool:
        """Check if week has data"""
        url = f"{config.FANTASY_BASE_URL}/league/{_self.league_key}/scoreboard;week={week}"
        try:
            resp = _self.oauth.get(url)
            if resp.status_code != 200:
                return False
            root = ET.fromstring(resp.text)
            matchups = root.findall('.//y:matchup', config.YAHOO_NS)
            return bool(matchups)
        except Exception:
            return False
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_available_weeks(_self, max_weeks: int = 18) -> List[int]:
        """Get available weeks"""
        weeks = []
        empty_streak = 0
        
        for week in range(1, max_weeks + 1):
            if _self.week_has_data(week):
                weeks.append(week)
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 3 and weeks:
                    break
        
        return weeks
    
    @st.cache_data(ttl=config.CACHE_TTL)  
    def get_team_weekly_totals(_self, week: int) -> List[Dict]:
        """Get team weekly totals"""
        url = f"{config.FANTASY_BASE_URL}/league/{_self.league_key}/scoreboard;week={week}"
        
        try:
            resp = _self.oauth.get(url)
            if resp.status_code != 200:
                return []
            
            root = ET.fromstring(resp.text)
            results = []
            
            for matchup in root.findall('.//y:matchup', config.YAHOO_NS):
                for team in matchup.findall('.//y:team', config.YAHOO_NS):
                    team_name_el = team.find('y:name', config.YAHOO_NS)
                    team_points_el = team.find('y:team_points/y:total', config.YAHOO_NS)
                    
                    if team_name_el is not None and team_points_el is not None:
                        try:
                            results.append({
                                "Team": team_name_el.text,
                                "Week": week,
                                "Total_Points": float(team_points_el.text)
                            })
                        except (ValueError, TypeError):
                            pass
            
            return results
            
        except Exception:
            return []
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_team_keys(_self) -> List[Tuple[str, str]]:
        """Get team keys and names"""
        teams_url = f"{config.FANTASY_BASE_URL}/league/{_self.league_key}/teams"
        
        try:
            resp = _self.oauth.get(teams_url)
            if resp.status_code != 200:
                return []
            
            root = ET.fromstring(resp.text)
            team_keys = []
            
            for team in root.findall('.//y:team', config.YAHOO_NS):
                team_key_el = team.find('y:team_key', config.YAHOO_NS)
                team_name_el = team.find('y:name', config.YAHOO_NS)
                
                if team_key_el is not None and team_name_el is not None:
                    team_keys.append((team_key_el.text, team_name_el.text))
            
            return team_keys
            
        except Exception:
            return []
    
    @st.cache_data(ttl=config.CACHE_TTL)
    def get_roster_data(_self, week: int) -> List[Dict]:
        """Get roster data for a SINGLE WEEK ONLY with weekly stats"""
        team_keys = _self.get_team_keys()
        if not team_keys:
            return []
        
        all_player_data = []
        
        for team_key, team_name in team_keys:
            # Step 1: Get roster structure (positions, starter/bench) for THIS WEEK ONLY
            roster_info = _self._get_team_roster_info(team_key, team_name, week)
            
            # Step 2: Get player keys from roster
            player_keys = [p["player_key"] for p in roster_info if p.get("player_key")]
            
            # Step 3: Get stats for THIS SPECIFIC WEEK ONLY
            if player_keys:
                player_stats = _self._get_players_single_week_stats(player_keys, week)
                
                # Step 4: Merge roster info with stats
                merged_data = _self._merge_roster_and_weekly_stats(roster_info, player_stats, team_name, week)
                all_player_data.extend(merged_data)
        
        return all_player_data
    
    def _get_team_roster_info(self, team_key: str, team_name: str, week: int) -> List[Dict]:
        """Get basic roster info (positions, starter/bench status)"""
        roster_url = f"{config.FANTASY_BASE_URL}/team/{team_key}/roster;week={week}"
        
        try:
            resp = self.oauth.get(roster_url)
            if resp.status_code != 200:
                return []
            
            root = ET.fromstring(resp.text)
            roster_info = []
            
            for player in root.findall('.//y:player', config.YAHOO_NS):
                # Extract basic roster information
                player_key_el = player.find('y:player_key', config.YAHOO_NS)
                if player_key_el is None:
                    continue
                    
                name_el = player.find('y:name/y:full', config.YAHOO_NS)
                if name_el is None:
                    name_el = player.find('y:name', config.YAHOO_NS)
                
                selected_pos_el = player.find('y:selected_position/y:position', config.YAHOO_NS)
                if selected_pos_el is None:
                    selected_pos_el = player.find('y:selected_position', config.YAHOO_NS)
                
                pos_el = player.find('y:display_position', config.YAHOO_NS)
                if pos_el is None:
                    pos_el = player.find('y:eligible_positions/y:position', config.YAHOO_NS)
                
                player_info = {
                    "player_key": player_key_el.text,
                    "player_name": name_el.text if name_el is not None else "Unknown",
                    "selected_position": selected_pos_el.text if selected_pos_el is not None else "Unknown",
                    "display_position": pos_el.text if pos_el is not None else "Unknown"
                }
                
                roster_info.append(player_info)
            
            return roster_info
            
        except Exception:
            return []
    
    def _get_players_single_week_stats(self, player_keys: List[str], week: int) -> Dict[str, float]:
        """Get stats for a SINGLE WEEK ONLY - not cumulative"""
        if not player_keys:
            return {}
        
        # The key insight: we need to get ONLY this week's data, not cumulative
        # Try the most specific endpoint format first
        player_keys_str = ",".join(player_keys)
        
        # Try endpoints that specifically request ONLY that week's data
        single_week_endpoints = [
            f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={player_keys_str}/stats;type=week;week={week}",
            f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={player_keys_str}/stats/week/{week}",
        ]
        
        for stats_url in single_week_endpoints:
            try:
                resp = self.oauth.get(stats_url)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    player_stats = {}
                    
                    for player in root.findall('.//y:player', config.YAHOO_NS):
                        player_key_el = player.find('y:player_key', config.YAHOO_NS)
                        if player_key_el is None:
                            continue
                        
                        player_key = player_key_el.text
                        
                        # Extract ONLY this week's fantasy points
                        points = self._extract_single_week_points(player, week)
                        player_stats[player_key] = points
                    
                    return player_stats
                        
            except Exception:
                continue
        
        # Fallback: If the API doesn't support true single-week queries,
        # we'll need to calculate the difference between cumulative totals
        return self._calculate_single_week_from_cumulative(player_keys, week)
    
    def _calculate_single_week_from_cumulative(self, player_keys: List[str], week: int) -> Dict[str, float]:
        """Calculate single week stats by subtracting previous weeks from cumulative"""
        if week == 1:
            # For week 1, cumulative = single week
            return self._get_cumulative_stats_through_week(player_keys, week)
        
        # Get cumulative through this week and previous week
        cumulative_current = self._get_cumulative_stats_through_week(player_keys, week)
        cumulative_previous = self._get_cumulative_stats_through_week(player_keys, week - 1)
        
        # Calculate the difference to get ONLY this week's stats
        single_week_stats = {}
        for player_key in player_keys:
            current_total = cumulative_current.get(player_key, 0.0)
            previous_total = cumulative_previous.get(player_key, 0.0)
            single_week_points = current_total - previous_total
            single_week_stats[player_key] = max(0.0, single_week_points)  # Ensure non-negative
        
        return single_week_stats
    
    def _get_cumulative_stats_through_week(self, player_keys: List[str], through_week: int) -> Dict[str, float]:
        """Get cumulative stats from week 1 through the specified week"""
        player_keys_str = ",".join(player_keys)
        stats_url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={player_keys_str}/stats;week={through_week}"
        
        try:
            resp = self.oauth.get(stats_url)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                player_stats = {}
                
                for player in root.findall('.//y:player', config.YAHOO_NS):
                    player_key_el = player.find('y:player_key', config.YAHOO_NS)
                    if player_key_el is None:
                        continue
                    
                    player_key = player_key_el.text
                    
                    # Get the cumulative points through this week
                    points = self._extract_fantasy_points_from_player_stats(player)
                    player_stats[player_key] = points
                
                return player_stats
        except Exception:
            pass
        
        return {}
    
    def _extract_single_week_points(self, player_element, week: int) -> float:
        """Extract fantasy points for a single week only"""
        # Look for weekly-specific fantasy points
        points_el = player_element.find('.//y:player_points/y:total', config.YAHOO_NS)
        if points_el is not None and points_el.text:
            points = safe_float(points_el.text)
            # Should be reasonable for a single week (0-60 range)
            if 0 <= points <= 60:
                return points
        
        # Look for reasonable decimal values that could be single-week points
        stats = player_element.findall('.//y:player_stats/y:stats/y:stat', config.YAHOO_NS)
        for stat in stats:
            stat_value_el = stat.find('y:value', config.YAHOO_NS)
            if stat_value_el is not None and stat_value_el.text and '.' in stat_value_el.text:
                value = safe_float(stat_value_el.text)
                if 0 <= value <= 60:  # Reasonable single-week range
                    return value
        
        return 0.0
    
    def _extract_fantasy_points_from_player_stats(self, player_element) -> float:
        """Extract fantasy points from player stats XML - targeting weekly fantasy points"""
        # Method 1: Look for player_points/total (most reliable for weekly data)
        points_el = player_element.find('.//y:player_points/y:total', config.YAHOO_NS)
        if points_el is not None and points_el.text:
            weekly_points = safe_float(points_el.text)
            # Weekly fantasy points should be reasonable (0-50 range typically)
            if 0 <= weekly_points <= 60:
                return weekly_points
        
        # Method 2: Look for stats section with fantasy points (stat_id="0" is typically fantasy points)
        stats = player_element.findall('.//y:player_stats/y:stats/y:stat', config.YAHOO_NS)
        fantasy_points_candidates = []
        
        for stat in stats:
            stat_id_el = stat.find('y:stat_id', config.YAHOO_NS)
            stat_value_el = stat.find('y:value', config.YAHOO_NS)
            
            if stat_id_el is not None and stat_value_el is not None:
                stat_id = stat_id_el.text
                stat_value = safe_float(stat_value_el.text)
                
                # stat_id="0" is typically fantasy points in Yahoo's system
                if stat_id == "0" and 0 <= stat_value <= 60:
                    return stat_value
                
                # Collect other potential fantasy points values (reasonable weekly range)
                if 0 <= stat_value <= 60:
                    fantasy_points_candidates.append((stat_id, stat_value))
        
        # Method 3: If we have candidates, try to pick the most likely fantasy points value
        if fantasy_points_candidates:
            # Sort by value and pick a reasonable one (not too high, not zero unless it should be)
            fantasy_points_candidates.sort(key=lambda x: x[1])
            for stat_id, value in fantasy_points_candidates:
                # Look for stat IDs that commonly represent fantasy points
                if stat_id in ["0", "1"]:  # Common fantasy points stat IDs
                    return value
            # If no common IDs found, return the median value as best guess
            if len(fantasy_points_candidates) > 0:
                mid_idx = len(fantasy_points_candidates) // 2
                return fantasy_points_candidates[mid_idx][1]
        
        # Method 4: Debug fallback - return 0 but this indicates extraction failed
        return 0.0
    
    def _merge_roster_and_weekly_stats(self, roster_info: List[Dict], player_stats: Dict[str, float], 
                                     team_name: str, week: int) -> List[Dict]:
        """Merge roster information with weekly stats"""
        merged_data = []
        
        for player_info in roster_info:
            player_key = player_info["player_key"]
            points = player_stats.get(player_key, 0.0)
            
            # Determine if player is starter or bench
            selected_pos = player_info["selected_position"]
            is_starter = selected_pos not in ['BN', 'IR', 'DL'] and selected_pos != "Unknown"
            
            merged_player = {
                "Team": team_name,
                "Player": player_info["player_name"],
                "Position": self._clean_position(player_info["display_position"]),
                "Selected_Position": selected_pos,
                "Is_Starter": is_starter,
                "Week": week,
                "Points": points
            }
            
            merged_data.append(merged_player)
        
        return merged_data
    
    def _clean_position(self, position: str) -> str:
        """Clean position name"""
        position = str(position).upper().strip()
        return self.POSITION_MAPPING.get(position, position)
    
    def create_positional_heatmap(self, data: List[Dict], starters_only: bool = False) -> Tuple[plt.Figure, pd.DataFrame]:
        """Create positional heatmap with starter/bench filtering and ranking-based colors"""
        if not data:
            return None, pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Filter for starters only if requested
        if starters_only:
            df = df[df['Is_Starter'] == True]
            if df.empty:
                st.warning("No starter data found for selected weeks.")
                return None, pd.DataFrame()
        
        # Debug: Show what we're aggregating
        st.write("**Debug - Raw data for DEF position:**")
        if 'DEF' in df['Position'].values:
            def_data = df[df['Position'] == 'DEF'][['Team', 'Player', 'Position', 'Points', 'Week', 'Is_Starter']]
            st.dataframe(def_data)
            
            # Show the Points values being summed
            st.write("**Points being summed for each team's DEF:**")
            for team in def_data['Team'].unique():
                team_def = def_data[def_data['Team'] == team]
                points_list = team_def['Points'].tolist()
                total = sum(points_list)
                st.write(f"- {team}: {points_list} = {total}")
        else:
            st.write("No DEF players found")
            sample_debug = df.head(10)[['Team', 'Player', 'Position', 'Points', 'Week']]
            st.dataframe(sample_debug)
        
        # Aggregate by team and position - sum points across all weeks
        totals = df.groupby(["Team", "Position"])["Points"].sum().reset_index()
        
        # Debug: Show aggregation results for DEF
        st.write("**Debug - Final aggregated totals for DEF:**")
        if 'DEF' in totals['Position'].values:
            def_totals = totals[totals['Position'] == 'DEF']
            st.dataframe(def_totals)
        
        pivot = totals.pivot(index="Team", columns="Position", values="Points").fillna(0)
        
        # Sort by total points
        team_totals = pivot.sum(axis=1).sort_values(ascending=False)
        pivot_sorted = pivot.loc[team_totals.index]
        
        # Create ranking for each position column
        rankings = {}
        for col in pivot_sorted.columns:
            rankings[col] = pivot_sorted[col].rank(ascending=False, method='min')
        
        # Create figure
        fig, ax = plt.subplots(
            figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot.index) * 0.5))
        )
        
        # Function to get color based on ranking
        def get_color_by_rank(rank, num_teams):
            if rank <= 3:
                return 'lightgreen'  # Top 3
            elif rank <= num_teams - 3:
                return 'khaki'  # Middle teams
            else:
                return 'lightcoral'  # Bottom 3
        
        num_teams = len(pivot_sorted.index)
        
        # Draw heatmap cells with ranking-based colors
        for i in range(len(pivot_sorted.index)):
            for j in range(len(pivot_sorted.columns)):
                value = pivot_sorted.iloc[i, j]
                position = pivot_sorted.columns[j]
                rank = rankings[position].iloc[i]
                
                # Get color based on ranking
                color = get_color_by_rank(rank, num_teams)
                
                # Draw cell
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, edgecolor='white', linewidth=1))
                
                # Add text
                if value > 0:
                    text_color = 'black'
                    ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha='center', va='center', 
                           color=text_color, weight='bold', fontsize=9)
                else:
                    ax.text(j + 0.5, i + 0.5, "0.0", ha='center', va='center', 
                           color='gray', fontsize=8)
        
        # Set axis properties
        ax.set_xlim(0, len(pivot_sorted.columns))
        ax.set_ylim(0, len(pivot_sorted.index))
        ax.invert_yaxis()
        
        # Labels
        ax.set_xticks([x + 0.5 for x in range(len(pivot_sorted.columns))])
        ax.set_xticklabels(pivot_sorted.columns, fontsize=11, weight='bold')
        ax.set_yticks([y + 0.5 for y in range(len(pivot_sorted.index))])
        ax.set_yticklabels(pivot_sorted.index, fontsize=10)
        
        # Remove ticks
        ax.tick_params(left=False, bottom=False)
        
        # Dynamic title
        title = "Starting Lineup Fantasy Points by Position" if starters_only else "Total Roster Fantasy Points by Position"
        plt.title(title, fontsize=14, weight='bold', pad=15)
        plt.xlabel("Position", fontsize=12, weight='bold')
        plt.ylabel("Team", fontsize=12, weight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Top 3 Teams'),
            Patch(facecolor='khaki', label='Middle Teams'), 
            Patch(facecolor='lightcoral', label='Bottom 3 Teams')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        return fig, pivot_sorted
    
    def get_league_scoring_settings(self) -> Dict:
        """Get league scoring settings for manual fantasy point calculation"""
        settings_url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/settings"
        
        try:
            resp = self.oauth.get(settings_url)
            if resp.status_code != 200:
                return {}
            
            root = ET.fromstring(resp.text)
            scoring_settings = {}
            
            # Look for stat_categories and their point values
            stat_categories = root.findall('.//y:stat_categories/y:stats/y:stat', config.YAHOO_NS)
            for stat in stat_categories:
                stat_id_el = stat.find('y:stat_id', config.YAHOO_NS)
                points_el = stat.find('.//y:value', config.YAHOO_NS)  # or points_per
                
                if stat_id_el is not None and points_el is not None:
                    scoring_settings[stat_id_el.text] = safe_float(points_el.text)
            
            return scoring_settings
            
        except Exception:
            return {}
    
    def calculate_fantasy_points_from_raw_stats(self, player_stats: Dict[str, str], scoring_settings: Dict[str, float]) -> float:
        """Calculate fantasy points manually from raw stats and league settings"""
        total_points = 0.0
        
        for stat_id, stat_value in player_stats.items():
            if stat_id in scoring_settings:
                stat_points = safe_float(stat_value) * scoring_settings[stat_id]
                total_points += stat_points
        
        return total_points
    
    def debug_player_stats_endpoint(self, sample_player_key: str, week: int) -> Dict:
        """Debug the player stats endpoint to see what data is available"""
        
        # Test the endpoint you found
        stats_url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={sample_player_key}/stats;week={week}"
        
        results = {
            "endpoint_url": stats_url,
            "status_code": None,
            "success": False,
            "xml_tags": [],
            "player_data": {},
            "all_stats_found": {},
            "raw_xml": ""
        }
        
        try:
            resp = self.oauth.get(stats_url)
            results["status_code"] = resp.status_code
            results["success"] = resp.status_code == 200
            results["raw_xml"] = resp.text[:3000]  # First 3000 chars
            
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                
                # Get all XML tags
                all_tags = set()
                for elem in root.iter():
                    if elem.tag:
                        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        all_tags.add(tag)
                results["xml_tags"] = sorted(list(all_tags))
                
                # Extract player data
                player = root.find('.//y:player', config.YAHOO_NS)
                if player is not None:
                    player_data = {}
                    
                    # Get basic info
                    name_el = player.find('y:name/y:full', config.YAHOO_NS)
                    if name_el is not None:
                        player_data["name"] = name_el.text
                    
                    # Get ALL stats with their IDs and values
                    stats = player.findall('.//y:player_stats/y:stats/y:stat', config.YAHOO_NS)
                    all_stats = {}
                    stats_dict = {}
                    for stat in stats:
                        stat_id_el = stat.find('y:stat_id', config.YAHOO_NS)
                        stat_value_el = stat.find('y:value', config.YAHOO_NS)
                        if stat_id_el is not None and stat_value_el is not None:
                            stat_id = stat_id_el.text
                            stat_value = stat_value_el.text
                            stats_dict[stat_id] = stat_value
                            all_stats[f"stat_id_{stat_id}"] = stat_value
                    
                    player_data["stats"] = stats_dict
                    results["all_stats_found"] = all_stats
                    
                    # Look for player_points
                    points_el = player.find('.//y:player_points/y:total', config.YAHOO_NS)
                    if points_el is not None:
                        player_data["player_points"] = points_el.text
                        results["all_stats_found"]["player_points_total"] = points_el.text
                    
                    # Look for other point-related fields
                    for elem in player.iter():
                        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                        if 'point' in tag.lower() and elem.text:
                            results["all_stats_found"][tag] = elem.text
                    
                    results["player_data"] = player_data
                    
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def test_batch_vs_individual_stats(self, player_keys: List[str], week: int) -> Dict:
        """Test batch vs individual player stats requests"""
        results = {
            "batch_request": None,
            "individual_requests": [],
            "comparison": {}
        }
        
        # Test batch request
        if len(player_keys) > 1:
            player_keys_str = ",".join(player_keys[:5])  # Test with first 5 players
            batch_url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={player_keys_str}/stats;week={week}"
            
            try:
                resp = self.oauth.get(batch_url)
                results["batch_request"] = {
                    "url": batch_url,
                    "status_code": resp.status_code,
                    "success": resp.status_code == 200,
                    "player_count": len(player_keys[:5]),
                    "response_length": len(resp.text)
                }
                
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    players_found = len(root.findall('.//y:player', config.YAHOO_NS))
                    results["batch_request"]["players_in_response"] = players_found
                    
            except Exception as e:
                results["batch_request"] = {"error": str(e)}
        
        # Test individual requests
        for i, player_key in enumerate(player_keys[:3]):  # Test first 3 individually
            individual_url = f"{config.FANTASY_BASE_URL}/league/{self.league_key}/players;player_keys={player_key}/stats;week={week}"
            
            try:
                resp = self.oauth.get(individual_url)
                individual_result = {
                    "player_key": player_key,
                    "url": individual_url,
                    "status_code": resp.status_code,
                    "success": resp.status_code == 200,
                    "response_length": len(resp.text)
                }
                
                if resp.status_code == 200:
                    root = ET.fromstring(resp.text)
                    player = root.find('.//y:player', config.YAHOO_NS)
                    if player is not None:
                        points = self._extract_fantasy_points_from_player_stats(player)
                        individual_result["fantasy_points"] = points
                
                results["individual_requests"].append(individual_result)
                
            except Exception as e:
                results["individual_requests"].append({
                    "player_key": player_key,
                    "error": str(e)
                })
        
        return results

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def setup_page():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="Fantasy Sports Analytics Suite",
        page_icon="🏆",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("🛠️ Controls")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🚪 Logout"):
            clear_session_state()
            st.rerun()
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **Fantasy Sports Analytics Suite**
        
        Advanced analytics for Yahoo Fantasy leagues:
        - Team performance heatmaps
        - Z-score strength analysis
        - Weekly trends analysis  
        - Positional breakdowns
        - Customizable timeframes
        
        **Supported Sports:**
        - ⚾ Baseball (MLB)
        - 🏈 Football (NFL)
        """)

def handle_authentication():
    """Handle OAuth authentication"""
    auth = YahooAuth()
    
    if st.session_state.token:
        # Test if token is still valid
        try:
            test_url = f"{config.FANTASY_BASE_URL}/users;use_login=1"
            response = auth.make_api_request(test_url, st.session_state.token)
            if response.status_code == 200:
                return True, st.session_state.token, None
        except Exception:
            pass
    
    # Check for auth code in URL
    query_params = st.query_params.to_dict()
    if "code" in query_params:
        try:
            code = query_params["code"]
            token = auth.exchange_code_for_token(code)
            st.session_state.token = token
            st.rerun()
        except Exception as e:
            return False, None, f"Authentication failed: {str(e)}"
    
    # Need to authenticate
    if not st.session_state.token:
        auth_url = auth.get_auth_url()
        return False, None, auth_url
    
    return True, st.session_state.token, None

def render_baseball_analytics(league_key: str, oauth_session: OAuth2Session):
    """Render enhanced baseball analytics"""
    analytics = BaseballAnalytics(league_key, oauth_session)
    
    st.success("⚾ Running Fantasy Baseball Analytics...")
    
    # Get available weeks
    available_weeks = analytics.get_available_weeks()
    if not available_weeks:
        st.warning("No weekly data found yet. Check back once the season starts.")
        return
    
    # Timeframe selection
    with st.sidebar:
        st.subheader("Timeframe")
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
    tab_heatmap, tab_strength, tab_z_scores, tab_data = st.tabs([
        "📊 Rankings Heatmap", 
        "💪 Team Strength", 
        "📈 Z-Score Analysis", 
        "📋 Raw Data"
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
            # Overall strength score (mean of all z-scores)
            overall_strength = strength_scores.mean(axis=1).sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Overall Team Strength")
                st.write("*Based on average z-scores across all categories*")
                
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
            
            # Consistency analysis
            st.subheader("Team Consistency")
            st.write("*Lower values indicate more consistent performance*")
            
            if not consistency_scores.empty:
                overall_consistency = consistency_scores.mean(axis=1).sort_values()
                
                consistency_df = pd.DataFrame({
                    'Team': overall_consistency.index,
                    'Consistency Score': overall_consistency.values,
                    'Rating': ['Very Consistent' if x < 0.5 else 'Consistent' if x < 1.0 else 'Variable' if x < 1.5 else 'Inconsistent' for x in overall_consistency.values]
                })
                
                st.dataframe(consistency_df, use_container_width=True)
        else:
            st.warning("Not enough data for strength analysis.")
    
    with tab_z_scores:
        st.header("Z-Score Analysis")
        
        if not strength_scores.empty:
            st.write("**Z-scores show how many standard deviations above/below league average each team performs in each category.**")
            st.write("- Green (positive): Above average performance")
            st.write("- Red (negative): Below average performance")
            
            fig_z = analytics.create_z_score_heatmap(strength_scores)
            st.pyplot(fig_z, use_container_width=True)
            
            # Z-score interpretation
            with st.expander("Z-Score Interpretation"):
                st.write("""
                - **Z > 1.5**: Excellent (top 7% of league)
                - **0.5 < Z < 1.5**: Good (top 30% of league)
                - **-0.5 < Z < 0.5**: Average (middle 40% of league)
                - **-1.5 < Z < -0.5**: Below Average (bottom 30% of league)
                - **Z < -1.5**: Poor (bottom 7% of league)
                """)
        else:
            st.warning("Not enough data for z-score analysis.")
    
    with tab_data:
        st.header("Raw Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cumulative Stats")
            st.dataframe(df_total.round(3))
        
        with col2:
            if not strength_scores.empty:
                st.subheader("Z-Score Strength")
                st.dataframe(strength_scores.round(3))

def render_football_analytics(league_key: str, oauth_session: OAuth2Session):
    """Render football analytics with proper starter/bench detection"""
    analytics = FootballAnalytics(league_key, oauth_session)
    
    st.success("🏈 Running Fantasy Football Analytics...")
    
    # Get available weeks
    available_weeks = analytics.get_available_weeks()
    if not available_weeks:
        st.warning("No weekly NFL data found yet.")
        return
    
    st.write(f"Available weeks: {available_weeks}")
    
    # Create tabs
    tab_positional, tab_trends, tab_debug = st.tabs(["🎯 Positional Analysis", "📈 Weekly Trends", "🔍 API Debug"])
    
    with tab_positional:
        st.header("Positional Fantasy Points Analysis")
        
        # Week selection
        weeks_to_analyze = st.multiselect(
            "Select weeks to analyze:",
            available_weeks,
            default=available_weeks[:min(3, len(available_weeks))]
        )
        
        if not weeks_to_analyze:
            st.info("Please select at least one week to analyze.")
            return
        
        # Toggle for starters vs all players
        analysis_type = st.radio(
            "Analysis Type:",
            ["All Players (Roster Depth)", "Starters Only (Actual Lineup)"],
            index=0,
            help="All Players shows total roster assets including bench. Starters Only shows points from actual weekly lineups."
        )
        
        starters_only = analysis_type.startswith("Starters Only")
        
        # Collect data - process each week separately to avoid cumulative issue
        all_data = []
        progress = st.progress(0)
        
        for i, week in enumerate(weeks_to_analyze):
            st.write(f"Processing Week {week}...")
            week_data = analytics.get_roster_data(week)  # This now gets ONLY that week's data
            all_data.extend(week_data)
            progress.progress((i + 1) / len(weeks_to_analyze))
        
        if not all_data:
            st.error("No roster data found.")
            return
        
        # Create visualization
        fig, pivot = analytics.create_positional_heatmap(all_data, starters_only=starters_only)
        
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            
            # Show additional analysis
            df_all = pd.DataFrame(all_data)
            
            if starters_only and 'Is_Starter' in df_all.columns:
                st.subheader("Starting Lineup Analysis")
                
                # Calculate efficiency metrics
                starters_total = df_all[df_all['Is_Starter'] == True]['Points'].sum()
                bench_total = df_all[df_all['Is_Starter'] == False]['Points'].sum()
                total_points = starters_total + bench_total
                
                if total_points > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Starter Points", f"{starters_total:.1f}")
                    with col2:
                        st.metric("Bench Points", f"{bench_total:.1f}")
                    with col3:
                        st.metric("Starter Efficiency", f"{(starters_total/total_points*100):.1f}%")
                
                # Show top performers by position (starters only)
                with st.expander("Top Starting Performers"):
                    starters_df = df_all[df_all['Is_Starter'] == True]
                    
                    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                        pos_players = starters_df[starters_df['Position'] == pos]
                        if not pos_players.empty:
                            top_3 = pos_players.nlargest(3, 'Points')[['Player', 'Team', 'Points', 'Week']]
                            if not top_3.empty:
                                st.write(f"**{pos} Leaders:**")
                                st.dataframe(top_3, use_container_width=True)
            else:
                st.subheader("Total Roster Analysis")
                st.write("Shows all fantasy points available on rosters, including bench players.")
            
            # Data table
            st.subheader("Data Table")
            st.dataframe(pivot.style.format("{:.1f}"))
            
            # Bench analysis if showing all players
            if not starters_only and 'Is_Starter' in df_all.columns:
                with st.expander("Bench Depth Analysis"):
                    bench_df = df_all[df_all['Is_Starter'] == False]
                    if not bench_df.empty:
                        bench_totals = bench_df.groupby(["Team", "Position"])["Points"].sum().reset_index()
                        bench_pivot = bench_totals.pivot(index="Team", columns="Position", values="Points").fillna(0)
                        
                        st.write("**Fantasy points sitting on benches:**")
                        st.dataframe(bench_pivot.style.format("{:.1f}"))
                        
                        # Teams with most bench points
                        bench_team_totals = bench_pivot.sum(axis=1).sort_values(ascending=False)
                        st.write("**Teams with most bench depth:**")
                        for team, points in bench_team_totals.head(5).items():
                            st.write(f"- {team}: {points:.1f} bench points")
        else:
            st.warning("Unable to create visualization. Check the debug data above.")
    
    with tab_trends:
        st.header("Weekly Performance Trends")
        
        # Get team totals
        all_weekly_data = []
        for week in available_weeks:
            weekly_data = analytics.get_team_weekly_totals(week)
            all_weekly_data.extend(weekly_data)
        
        if all_weekly_data:
            df = pd.DataFrame(all_weekly_data)
            pivot = df.pivot(index="Team", columns="Week", values="Total_Points").fillna(0)
            
            st.subheader("Weekly Totals")
            st.dataframe(pivot.style.format("{:.1f}"))
            
            # Simple line chart
            if len(pivot.columns) > 1:
                st.line_chart(pivot.T)
        else:
            st.warning("No weekly team data available.")
    
    with tab_debug:
        st.header("Yahoo Fantasy API Debug Tool")
        st.write("Debug the new player stats endpoint approach.")
        
        # Get team keys
        team_keys = analytics.get_team_keys()
        
        if team_keys:
            selected_team = st.selectbox(
                "Select team for debugging:",
                [f"{name} ({key})" for key, name in team_keys]
            )
            
            debug_week = st.selectbox("Select week:", available_weeks)
            
            team_key = selected_team.split("(")[1].rstrip(")")
            team_name = selected_team.split(" (")[0]
            
            # Step 1: Get roster info to extract player keys
            if st.button("Step 1: Get Roster Info"):
                st.subheader("Step 1: Roster Information")
                
                roster_info = analytics._get_team_roster_info(team_key, team_name, debug_week)
                
                if roster_info:
                    st.success(f"✅ Found {len(roster_info)} players on roster")
                    
                    # Show sample roster data
                    st.write("**Sample roster data:**")
                    df = pd.DataFrame(roster_info[:10])
                    st.dataframe(df)
                    
                    # Extract player keys for next step
                    player_keys = [p["player_key"] for p in roster_info if p.get("player_key")]
                    st.write(f"**Player keys extracted:** {len(player_keys)}")
                    
                    # Store in session state for next steps
                    st.session_state['debug_player_keys'] = player_keys
                    st.session_state['debug_roster_info'] = roster_info
                    
                    # Show a few sample player keys
                    st.write("**Sample player keys:**")
                    for key in player_keys[:5]:
                        st.code(key)
                else:
                    st.error("❌ No roster data found")
            
            # Step 2: Test player stats endpoint
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Step 2: Test Single Player Stats"):
                    if 'debug_player_keys' in st.session_state and st.session_state['debug_player_keys']:
                        st.subheader("Step 2: Single Player Stats Test")
                        
                        # Test with first player
                        test_player_key = st.session_state['debug_player_keys'][0]
                        
                        results = analytics.debug_player_stats_endpoint(test_player_key, debug_week)
                        
                        st.write(f"**Testing player:** {test_player_key}")
                        st.write(f"**Endpoint:** {results['endpoint_url']}")
                        
                        if results['success']:
                            st.success(f"✅ Status {results['status_code']}")
                            
                            # Show player data found
                            if results.get('player_data'):
                                player_data = results['player_data']
                                st.write("**Player found:**")
                                st.write(f"- Name: {player_data.get('name', 'N/A')}")
                                st.write(f"- Fantasy Points: {player_data.get('player_points', 'N/A')}")
                                
                                if player_data.get('stats'):
                                    st.write("**Individual stats:**")
                                    stats_df = pd.DataFrame(list(player_data['stats'].items()), 
                                                          columns=['Stat ID', 'Value'])
                                    st.dataframe(stats_df)
                            
                            # Show XML tags found
                            if results.get('xml_tags'):
                                with st.expander("XML Tags Found"):
                                    tags_of_interest = [tag for tag in results['xml_tags'] 
                                                      if any(keyword in tag.lower() for keyword in 
                                                           ['point', 'stat', 'player'])]
                                    st.write("**Tags of interest:**")
                                    for tag in tags_of_interest:
                                        st.code(tag)
                            
                            with st.expander("Raw XML Response"):
                                st.text(results.get('raw_xml', 'No XML data'))
                        else:
                            st.error(f"❌ Failed: Status {results.get('status_code', 'Unknown')}")
                            if 'error' in results:
                                st.error(f"Error: {results['error']}")
                    else:
                        st.warning("Please run Step 1 first to get player keys")
            
            with col2:
                if st.button("Step 3: Test Batch vs Individual"):
                    if 'debug_player_keys' in st.session_state and st.session_state['debug_player_keys']:
                        st.subheader("Step 3: Batch vs Individual Requests")
                        
                        player_keys = st.session_state['debug_player_keys']
                        
                        results = analytics.test_batch_vs_individual_stats(player_keys, debug_week)
                        
                        # Show batch results
                        if results.get('batch_request'):
                            batch = results['batch_request']
                            st.write("**Batch Request Results:**")
                            if batch.get('success'):
                                st.success(f"✅ Batch request successful")
                                st.write(f"- Requested {batch.get('player_count', 0)} players")
                                st.write(f"- Found {batch.get('players_in_response', 0)} in response")
                            else:
                                st.error(f"❌ Batch request failed: {batch.get('status_code', 'Unknown')}")
                                if 'error' in batch:
                                    st.error(f"Error: {batch['error']}")
                        
                        # Show individual results
                        if results.get('individual_requests'):
                            st.write("**Individual Request Results:**")
                            for req in results['individual_requests']:
                                if req.get('success'):
                                    points = req.get('fantasy_points', 0)
                                    st.write(f"✅ {req['player_key']}: {points} points")
                                else:
                                    st.write(f"❌ {req['player_key']}: Failed")
                    else:
                        st.warning("Please run Step 1 first to get player keys")
            
            # Step 4: Full test
            if st.button("Step 4: Full Integration Test"):
                st.subheader("Step 4: Full Integration Test")
                
                with st.spinner("Testing full data extraction..."):
                    # Test the complete get_roster_data method
                    roster_data = analytics.get_roster_data(debug_week)
                    
                    if roster_data:
                        st.success(f"✅ Successfully extracted {len(roster_data)} player records")
                        
                        df = pd.DataFrame(roster_data)
                        
                        # Analyze results
                        total_points = df['Points'].sum()
                        players_with_points = len(df[df['Points'] > 0])
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Players", len(roster_data))
                        with col_b:
                            st.metric("Players with Points", players_with_points)
                        with col_c:
                            st.metric("Total Points", f"{total_points:.1f}")
                        
                        # Show breakdown by starter/bench
                        if 'Is_Starter' in df.columns:
                            starters = df[df['Is_Starter'] == True]
                            bench = df[df['Is_Starter'] == False]
                            
                            st.write("**Starter vs Bench Breakdown:**")
                            col_d, col_e = st.columns(2)
                            with col_d:
                                st.write(f"Starters: {len(starters)} players, {starters['Points'].sum():.1f} points")
                            with col_e:
                                st.write(f"Bench: {len(bench)} players, {bench['Points'].sum():.1f} points")
                        
                        # Show top scorers
                        if players_with_points > 0:
                            st.write("**Top Scorers:**")
                            top_scorers = df.nlargest(10, 'Points')[['Player', 'Team', 'Position', 'Points', 'Is_Starter']]
                            st.dataframe(top_scorers)
                        
                        # Show sample of all data
                        with st.expander("Sample Data"):
                            st.dataframe(df.head(15))
                    else:
                        st.error("❌ No data extracted")
            
            # Instructions
            with st.expander("How to use this debug tool"):
                st.markdown(f"""
                **Step-by-step debugging process:**
                
                1. **Get Roster Info**: Extracts player keys and basic roster information
                2. **Test Single Player**: Tests the player stats endpoint with one player
                3. **Test Batch vs Individual**: Compares batch requests vs individual requests
                4. **Full Integration Test**: Tests the complete data extraction pipeline
                
                **The new API approach:**
                
                Uses endpoint: `{config.FANTASY_BASE_URL}/league/{{league_key}}/players;player_keys={{player_key}}/stats;week={{week}}`
                
                This should return weekly fantasy points for players instead of season totals.
                
                **What to look for:**
                - `player_points/total` for fantasy points
                - `stats` with `stat_id="0"` (often fantasy points)
                - Individual stat values that could be fantasy points
                """)
        else:
            st.warning("Could not retrieve team information.")

def render_landing_page(auth_url: str):
    """Render enhanced landing page with logo and better design"""
    
    # Header with logo and title
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display the dog mascot logo 
        st.image("https://i.imgur.com/placeholder.jpg", width=150, caption="")  # You'll need to upload your logo
        
    with col2:
        st.markdown("""
        <div style="padding: 20px 0px;">
        <h1 style="color: #1f77b4; margin-bottom: 0px;">Skipper Analytics</h1>
        <h3 style="color: #666; margin-top: 0px; font-weight: 300;">Advanced Fantasy Sports Intelligence</h3>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main value proposition
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px 0px;">
        <h2 style="color: #333;">Take Your Fantasy Game to the Next Level</h2>
        <p style="font-size: 18px; color: #666; line-height: 1.6;">
        Get advanced analytics, team strength analysis, and positional insights 
        that go far beyond basic fantasy apps.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Connect with Yahoo section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 40px; border-radius: 15px; margin: 30px 0px;">
    <div style="text-align: center;">
    <h3 style="color: white; margin-bottom: 20px;">Connect Your Yahoo Fantasy Account</h3>
    """, unsafe_allow_html=True)
    
    # Yahoo Fantasy Sports logo link (centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        yahoo_logo_html = f"""
        <div style="text-align: center;">
        <a href="{auth_url}" target="_self">
        <img src="https://s.yimg.com/rz/d/yahoo_sports_en-US_f_p_142x37_2x.png" 
             style="background: white; padding: 10px; border-radius: 8px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s;"
             onmouseover="this.style.transform='scale(1.05)'"
             onmouseout="this.style.transform='scale(1)'"
             alt="Connect with Yahoo Fantasy Sports">
        </a>
        </div>
        """
        st.markdown(yahoo_logo_html, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: white; text-align: center; margin-top: 20px; opacity: 0.9;">
    Secure OAuth connection - we never store your password
    </p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("""
    <h2 style="text-align: center; margin: 50px 0px 30px 0px; color: #333;">
    What Makes Skipper Analytics Different
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <div style="font-size: 48px; margin-bottom: 15px;">📊</div>
        <h4 style="color: #1f77b4;">Smart Heatmaps</h4>
        <p style="color: #666; font-size: 14px;">
        Color-coded positional rankings that instantly show team strengths and weaknesses
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <div style="font-size: 48px; margin-bottom: 15px;">⚡</div>
        <h4 style="color: #1f77b4;">True Weekly Data</h4>
        <p style="color: #666; font-size: 14px;">
        Accurate week-by-week analysis, not misleading cumulative totals
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
        <div style="font-size: 48px; margin-bottom: 15px;">🎯</div>
        <h4 style="color: #1f77b4;">Multi-Sport Ready</h4>
        <p style="color: #666; font-size: 14px;">
        Baseball z-score analysis and football positional breakdowns in one app
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sports supported
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0px;">
    <h3 style="color: #333; margin-bottom: 20px;">Supported Sports</h3>
    <div style="display: flex; justify-content: center; gap: 40px;">
    <div style="text-align: center;">
    <div style="font-size: 32px; margin-bottom: 8px;">⚾</div>
    <strong>Baseball (MLB)</strong><br>
    <span style="color: #666; font-size: 14px;">Z-score strength analysis<br>Team consistency metrics</span>
    </div>
    <div style="text-align: center;">
    <div style="font-size: 32px; margin-bottom: 8px;">🏈</div>
    <strong>Football (NFL)</strong><br>
    <span style="color: #666; font-size: 14px;">Positional heatmaps<br>Starter vs bench analysis</span>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Setup
    setup_page()
    init_session_state()
    
    # Sidebar
    render_sidebar()
    
    # Show debug info if requested
    if st.checkbox("Show debug info"):
        with st.expander("Debug Info"):
            st.json({
                "client_id": config.CLIENT_ID[:20] + "..." if config.CLIENT_ID else "Not set",
                "redirect_uri": config.REDIRECT_URI,
                "fantasy_base_url": config.FANTASY_BASE_URL
            })
    
    try:
        # Authentication
        is_authenticated, token, auth_result = handle_authentication()
        
        if not is_authenticated:
            if auth_result and auth_result.startswith("http"):
                # Show enhanced landing page
                render_landing_page(auth_result)
            else:
                st.error(f"Authentication failed: {auth_result}")
            return
        
        st.success("✅ Successfully authenticated!")
        
        # Get leagues
        auth = YahooAuth()
        try:
            leagues = auth.get_user_leagues(token)
        except Exception as e:
            st.error(f"Failed to load leagues: {str(e)}")
            if st.button("Clear session and retry"):
                clear_session_state()
                st.rerun()
            return
        
        if not leagues:
            st.warning("No fantasy leagues found for your account.")
            return
        
        # League selection
        st.subheader("📋 Your Fantasy Leagues")
        
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
            render_baseball_analytics(league_key, oauth_session)
        elif is_football_league(game_code, game_name):
            render_football_analytics(league_key, oauth_session)
        else:
            st.error(f"Unsupported sport: {game_name}")
    
    except Exception as e:
        st.error("An unexpected error occurred.")
        with st.expander("Error Details"):
            st.exception(e)
        if st.button("Restart Application"):
            clear_session_state()
            st.rerun()

if __name__ == "__main__":
    main()
