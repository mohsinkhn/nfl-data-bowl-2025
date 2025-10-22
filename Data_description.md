Description

Note: This Big Data Bowl 2026 has two competitions. This competition is the Prediction competition. Learn more about the Analytics competition here.

The downfield pass is the crown jewel of American sports. When the ball is in the air, anything can happen, like a touchdown, an interception, or a contested catch. The uncertainty and the importance of the outcome of these plays is what helps keep audiences on the edge of its seat.

The 2026 Big Data Bowl is designed to help the National Football League better understand player movement during the pass play, starting with when the ball is thrown and ending when the ball is either caught or ruled incomplete. For the offensive team, this means focusing on the targeted receiver, whose job is to move towards the ball landing location in order to complete a catch. For the defensive team, who could have several players moving towards the ball, their jobs are to both prevent the offensive player from making a catch, while also going for the ball themselves. This year's Big Data Bowl asks our fans to help track the movement of these players.

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player.

Competition specifics

In the NFL's tracking data, there are 10 frames per second. As a result, if a ball is in the air for 2.5 seconds, there will be 25 frames of location data to predict.
Quick passes (less than half a second), deflected passes, and throwaway passes are dropped from the competition.
Evaluation for the training data is based on historical data. Evaluation for the leaderboard is based on data that hasn't happened yet. Specifically, we will be doing a live leaderboard covering the last five weeks of the 2025 NFL season.

Evaluation
Submissions are evaluated using the Root Mean Squared Error between the predicted and the observed target. The evaluation metric for this contest can be found here.


Submission File
For each id in the test set, you must predict a value for the x, y variable. The file should contain a header and have the following format:

id,x,y
'1_1_1_1',0,0
'1_1_1_2',0,0
'1_1_1_3',0,0
etc.
NOTE: id is a concatenated identifier in format {game_id}_{play_id}_{nfl_id}_{frame_id}.

Timeline
This is a forecasting competition with an active training phase and a separate forecasting phase where models will be run against player data from future NFL games.
Training Timeline:
September 25, 2025 - Competition Start

November 26, 2025 - Team Merger deadline. This is the last day participants may join or merge teams.

November 26, 2025 - Entry deadline. You must accept the competition rules before this date in order to compete.

December 3, 2025 - Final submission deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Forecasting Timeline:
December 4, 2025 - January 5, 2026 - Watch your results play out! The Kaggle Team will refresh the leaderboard throughout the competition at the conclusion of the prior week's NFL games (NFL games are typically played on Thursdays, Sundays, and Mondays). The first week of games scored begins with those played on December 4, December 7, and December 8. The final scheduled NFL game of the 2025-2026 regular season is Sunday, January 4, 2026.

January 6, 2026  - Competition end, results published

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Summary of data
Here, you'll find a summary of each data set in the 2026 NFL Big Data Bowl, a list of key variables to join on, and a description of each variable. The tracking data is provided by the NFL Next Gen Stats team.

Files
train/
input_2023_w[01-18].csv
The input data contains tracking data before the pass is thrown

game_id: Game identifier, unique (numeric)
play_id: Play identifier, not unique across games (numeric)
player_to_predict: whether or not the x/y prediction for this player will be scored (bool)
nfl_id: Player identification number, unique across players (numeric)
frame_id: Frame identifier for each play/type, starting at 1 for each game_id/play_id/file type (input or output) (numeric)
play_direction: Direction that the offense is moving (left or right)
absolute_yardline_number: Distance from end zone for possession team (numeric)
player_name: player name (text)
player_height: player height (ft-in)
player_weight: player weight (lbs)
player_birth_date: birth date (yyyy-mm-dd)
player_position: the player's position (the specific role on the field that they typically play)
player_side: team player is on (Offense or Defense)
player_role: role player has on play (Defensive Coverage, Targeted Receiver, Passer or Other Route Runner)
x: Player position along the long axis of the field, generally within 0 - 120 yards. (numeric)
y: Player position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)
s: Speed in yards/second (numeric)
a: Acceleration in yards/second^2 (numeric)
o: orientation of player (deg)
dir: angle of player motion (deg)
num_frames_output: Number of frames to predict in output data for the given game_id/play_id/nfl_id. (numeric)
ball_land_x: Ball landing position position along the long axis of the field, generally within 0 - 120 yards. (numeric)
ball_land_y: Ball landing position along the short axis of the field, generally within 0 - 53.3 yards. (numeric)
output_2023_w[01-18].csv
The output data contains tracking data after the pass is thrown.

game_id: Game identifier, unique (numeric)
play_id: Play identifier, not unique across games (numeric)
nfl_id: Player identification number, unique across players. (numeric)
frame_id: Frame identifier for each play/type, starting at 1 for each game_id/play_id/ file type (input or output). The maximum value for a given game_id, play_id and nfl_id will be the same as the num_frames_output value from the corresponding input file. (numeric)
x: Player position along the long axis of the field, generally within 0-120 yards. (TARGET TO PREDICT)
y: Player position along the short axis of the field, generally within 0 - 53.3 yards. (TARGET TO PREDICT)
test_input.csv
Player tracking data before the pass is thrown for the held-out test plays. Same structure as the input files. Provided without ground truth outputs.

test.csv
A mock test set representing the structure of the unseen test set.
Contains the prediction targets as rows with columns (game_id, play_id, nfl_id, frame_id) representing each position that needs to be predicted.

sample_submission.csv
The required submission format showing the expected structure for predictions. Contains columns (id, x, y) where: id is a concatenated identifier in format {game_id}_{play_id}_{nfl_id}_{frame_id}

