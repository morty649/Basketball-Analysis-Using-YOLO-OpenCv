from .utils import draw_ellipse

class PlayerTracksDrawer:
    def __init__(self):
        pass

    def draw(self,video_frames,tracks):
        output_video_frames = []
        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks[frame_num]

            #drawing players
            for track_id,player in player_dict.items():
                frame = draw_ellipse(frame,player["bbox"],(0,0,255),track_id)
            output_video_frames.append(frame)

        return output_video_frames

'''np.int64(104): {'bbox': [678.0634765625, 118.88836669921875, 716.885498046875, 266.1722717285156]}}, 
 {np.int64(122): {'bbox': [393.4752502441406, 205.33209228515625, 455.4018859863281, 329.59368896484375]},
 np.int64(4): {'bbox': [550.6361083984375, 292.02197265625, 673.9376220703125, 452.0999755859375]}, 
 np.int64(115): {'bbox': [218.43032836914062, 214.66799926757812, 286.4392395019531, 366.0453186035156]},
   np.int64(104): {'bbox': [675.6800537109375, 118.98727416992188, 715.1744384765625, 266.2636413574219]}}]
   in drawing players track_id is np.something given and player is bbox that is xyxy --- well this is output i tried before drawing 
   '''