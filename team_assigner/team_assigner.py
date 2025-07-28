from PIL import Image
import cv2
import os
import sys
sys.path.append("../")
from utils import read_stub,save_stub
from transformers import CLIPProcessor, CLIPModel

class TeamAssigner:
    def __init__(self,
                 team1_class_name = "white shirt",
                 team2_class_name = "dark red shirt"):
        self.team1_class_name = team1_class_name
        self.team2_class_name = team2_class_name

        self.player_team_dict = {} 

    def load_model(self):
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image) #this helps in expected format of pytorch

        classes = [self.team1_class_name,self.team2_class_name]

        inputs = self.processor(text=classes,images=pil_image,return_tensors="pt",padding=True)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name
    
    def get_player_team(self,frame,player_bbox,player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)

        team_id = 2
        if player_color==self.team1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id #dictionary with player_id as key and which team he belongs as key

        return team_id
    
    def get_player_teams_across_frames(self,video_frames,player_tracks,read_from_stub=False,stub_path=None):
        player_team_assign = read_stub(read_from_stub,stub_path)
        if player_team_assign is not None:
            if len(player_team_assign)==len(video_frames):
                return player_team_assign
             
        self.load_model()

        player_team_assign = []
        for frame_num,player_track in enumerate(player_tracks):
            player_team_assign.append({})

            if frame_num %50==0:                    #Stops misclassification 
                self.player_team_dict={}            #clears for every 50 frames
            
            for player_id,track in player_track.items():
                team = self.get_player_team(video_frames[frame_num],track["bbox"],player_id)
                player_team_assign[frame_num][player_id] = team

        save_stub(stub_path,player_team_assign)
        
        return player_team_assign
            

             


        