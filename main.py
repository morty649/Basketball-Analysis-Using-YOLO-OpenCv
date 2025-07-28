from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from drawers import(
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer,
    PassInterceptionDrawer
)
from team_assigner import TeamAssigner
from ball_aquisition import BallAquisitionDetector
from pass_and_interception_detection import PassAndInterceptionDetection
def main():
    #Read video
    video_frames = read_video("input_videos/video_3.mp4")

    #Initialize Trackers
    player_tracker = PlayerTracker("models/player_detection.pt")
    ball_tracker = BallTracker("models/ball_detection_model.pt")

    #Run trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/player_tracks_stub.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/ball_tracks_stub.pkl")

    #Remove Wrong ball Detections And also Interpolate missing ones
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    #Player_team_assigner
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                player_tracks,
                                                                read_from_stub=True,
                                                                stub_path="stubs/team_assigner_stub.pkl")
    

    #Ball Acquisition
    ball_acquisition_detector = BallAquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks,ball_tracks)


    #Detect Passes and Interceptions
    passes_and_interceptions = PassAndInterceptionDetection()
    passes = passes_and_interceptions.detect_passes(ball_acquisition,player_assignment)
    interceptions = passes_and_interceptions.detect_interception(ball_acquisition,player_assignment)
    

    #Initialize drawers
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()
    pass_and_interception_drawer = PassInterceptionDrawer()


    #draw things 
    output_video_frames = player_drawer.draw(video_frames,player_tracks,
                                             player_assignment,ball_acquisition)
    output_video_frames = ball_drawer.draw(output_video_frames,ball_tracks)
    #print(player_tracks)


    #Drawing team ball Control 

    output_video_frames = team_ball_control_drawer.draw(output_video_frames,
                                                        player_assignment,
                                                        ball_acquisition)
    
    #Draw Passes and Interceptions
    
    output_video_frames = pass_and_interception_drawer.draw(output_video_frames,passes,interceptions)

    #save video
    save_video(output_video_frames,"outputs/output_video.avi")

    #command line printing of saving
    print("Video Saved")

if __name__ == "__main__":
    main()