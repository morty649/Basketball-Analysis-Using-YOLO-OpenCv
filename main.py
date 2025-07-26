from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from drawers import PlayerTracksDrawer,BallTracksDrawer
from team_assigner import TeamAssigner
def main():
    #Read video
    video_frames = read_video("input_videos/video_1.mp4")

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

    #Initialize drawers
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()


    #draw things 
    output_video_frames = player_drawer.draw(video_frames,player_tracks,
                                             player_assignment)
    output_video_frames = ball_drawer.draw(output_video_frames,ball_tracks)
    #print(player_tracks)


    #save video
    save_video(output_video_frames,"outputs/output_video.avi")

    #command line printing of saving
    print("Video Saved")

if __name__ == "__main__":
    main()