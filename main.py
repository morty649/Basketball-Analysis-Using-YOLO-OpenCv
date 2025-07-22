from utils import read_video,save_video
from trackers import PlayerTracker,BallTracker
from drawers import PlayerTracksDrawer
def main():
    #Read video
    video_frames = read_video("input_videos/video_1.mp4")

    #Initialize Trackers
    player_tracker = PlayerTracker("models/player_detection.pt")
    ball_tracker = BallTracker("models/ball_detection_model.pt")

    #Run trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/player_tracks_stub.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path="stubs/ball_tracks_stub.pkl")

    #Drawing ellipses
    player_drawer = PlayerTracksDrawer()
    output_video_frames = player_drawer.draw(video_frames,player_tracks)
    #print(player_tracks)

    #save video
    save_video(output_video_frames,"outputs/output_video.avi")

if __name__ == "__main__":
    main()