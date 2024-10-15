#%%
import os
import json
import logging
import anthropic
from espn_api.football import League
import gamedaybot.espn.functionality as espn

logger = logging.getLogger(__name__)

def get_anthropic_client():
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    return client

#%%
specific_team_summary_user_prompt = """
You are a witty and opinionated fantasy football analyst. Your task is to analyze the provided league data and create an entertaining summary.

Here is the fantasy football league data:
<league_overview_data>
{{LEAGUE_DATA}}
</league_overview_data>

Here are summaries of the rosters and performance of each team:
<team_specific_summaries>
{{TEAM_SUMMARIES}}
</team_specific_summaries>

Analyze this data carefully, paying attention to player performance, team rankings, scores, and any notable performances or trends.

Create a summary of the league's current state. Your summary can include:

1. The team's current ranking and record
2. Any standout players or disappointing player performances
3. How they compare to other teams in the league

Present your analysis in a humorous and slightly sarcastic tone. Feel free to use puns, pop culture references, or playful jabs at the team's performance (whether good or bad). Your goal is to inform while entertaining the reader.

Structure your response as follows:

<main_analysis>
[Your main analysis goes here. This should be 5-7 paragraphs long.]
</main_analysis>

<one_liner>
[A witty one-liner that encapsulates the team's current situation or performance]
</one_liner>

Remember, your tone should be engaging and entertaining, but make sure your analysis is based on the actual data provided. Don't make up facts, but feel free to speculate or make jokes based on the information you have.
"""

team_roster_summary_prompt = """
You will be given JSON data containing information about a fantasy football league team's roster. Your task is to summarize this data for a specific team. Here's how to proceed:

1. First, you will receive the JSON data for the league's roster:
<team_roster_json>
{{TEAM_ROSTER_JSON}}
</team_roster_json>

2. You will also be given the name of the team to summarize:
<team_name>{{TEAM_NAME}}</team_name>

3. Parse the JSON data to extract information about the specified team. The JSON structure should contain details such as player names, positions, points scored, and total points across all games.

4. Analyze the parsed data and create a summary of the team's roster. Include the following information:
   a. Total number of players on the team
   b. Players in each position (e.g., QB, RB, WR, TE, K, DEF)
   c. Top 3 highest-scoring players on the team (include their names, positions, and total points)

5. Format your summary as follows:
   - Begin your response with <team_summary> tags
   - Use appropriate subheadings for each section of the summary
   - Present numerical data in a clear, easy-to-read format
   - End your response with </team_summary> tags

Here's an example of how your output should be structured:

<team_summary>
Team: [Team Name]

Roster Overview:
- Total players: [Number]
- Positions: QB: [Player Name], RB: [Player Names], WR: [Player Names], TE: [Player Names], K: [Player Names], DEF: [Defense Name]

Players And Points Scored [Players who are not on the bench or IR]:
1. [Player Name] ([Position]) - [This Week's Points] points
2. [Player Name] ([Position]) - [This Week's Points] points
...

Team Performance: [Text Summary of Team Performance, taking into account passing, rushing, lineup slot, active status, position, and points in the dataForWeek field. Account for whether the player is on the bench or IR - 
if they are on the bench or on IR, the points do not count]
</team_summary>

Remember to replace the placeholders in brackets with the actual data from the JSON input. Provide a concise yet informative summary that gives a clear picture of the team's composition and performance.
"""

def team_name_mapping(league: League):
    mappings = [f"\"{team.team_name}\" = \"{team.team_abbrev}\", owner = {team.owners[0]['firstName']}" for team in league.teams]
    return "\n".join(mappings)

def roster_data_for_team(league: League, team, week):
    roster_data = []
    for player in team.roster:
        player_data = {
            "acquisitionType": player.acquisitionType,
            "activeStatus": player.active_status,
            "lineupSlot": player.lineupSlot,
            "name": player.name,
            "position": player.position,
            "proTeam": player.proTeam,
            "totalPoints": player.total_points,
        }
        if week in player.stats:
            player_stats_current_week = player.stats[week]
            player_data_current_week = {
                "projectedPoints": player_stats_current_week["projected_points"]
            }
            if player.active_status == "active" and "breakdown" in player_stats_current_week:
                player_data_current_week["points"] = player_stats_current_week["points"]
                breakdown_keys = [
                    "passingAttempts",
                    "passingTimesSacked",
                    "passingCompletions",
                    "passingIncompletions",
                    "passingYards",
                    "passingTouchdowns",
                    "passing40PlusYardTD",
                    "passingCompletionPercentage",
                    "rushingAttempts",
                    "rushingYards",
                    "pointsScored",
                    "turnovers",
                    "passingInterceptions",
                    "rushingYardsPerAttempt",
                ]
                breakdown_stats = {b_k: player_stats_current_week['breakdown'][b_k] for b_k in breakdown_keys if b_k in player_stats_current_week['breakdown']}
                player_data_current_week["breakdown"] = breakdown_stats
            player_data["dataForWeek"] = player_data_current_week

        roster_data.append(player_data)
    return roster_data

def teams_roster_data(league: League, week: int):
    teams = []
    for team in league.teams:
        team_roster_data = roster_data_for_team(league, team, week)
        teams.append({
            "teamName": team.team_name,
            "teamAbbreviation": team.team_abbrev,
            "rosterData": team_roster_data
        })
    return json.dumps(teams)

def end_of_week_summary_for_prompt(league: League):
    previous_week = league.current_week - 1
    return """
        Team names:
        {team_name_mapping}
        
        Last week's scoreboard:
        {scoreboard}
        
        Optimal Scores:
        {optimal_scores}

        Trophies:
        {trophies}

        Power rankings:
        {power_rankings}

        Current Standings:
        {standings}

        Matchups for next week:
        {matchups}
    """.format(team_name_mapping=team_name_mapping(league),
               scoreboard=espn.get_scoreboard_short(league, week=previous_week),
               optimal_scores=espn.optimal_team_scores(league, week=previous_week, full_report=True),
               trophies=espn.get_trophies(league, week=previous_week),
               power_rankings=espn.get_power_rankings(league),
               standings=espn.get_standings(league),
               matchups=espn.get_matchups(league))

def end_of_week_ai_summary(league: League):
    league_data_text = end_of_week_summary_for_prompt(league)
    team_summaries = []
    for team in league.teams:
        team_summaries.append(team_roster_summary(league, team))
    prompt_with_data = specific_team_summary_user_prompt.replace("{{LEAGUE_DATA}}", league_data_text).replace("{{TEAM_SUMMARIES}}", "\n".join(team_summaries))
    message = get_anthropic_client().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_with_data
                    }
                ]
            }
        ]
    )
    content = message.content[0].text
    main_analysis = content.split("<main_analysis>")[1].split("</main_analysis>")[0]
    one_liner = content.split("<one_liner>")[1].split("</one_liner>")[0]
    return main_analysis, one_liner

def team_roster_summary(league, team):
    logger.info("Creating team summary for team_name={}", team.team_name)
    roster_data = roster_data_for_team(league, team, league.current_week - 1)
    prompt_with_data = team_roster_summary_prompt.replace("{{TEAM_ROSTER_JSON}}", json.dumps(roster_data)).replace("{{TEAM_NAME}}", team.team_name)
    message = get_anthropic_client().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_with_data
                    }
                ]
            }
        ]
    )
    content = message.content[0].text
    return content