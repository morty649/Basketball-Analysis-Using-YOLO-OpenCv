import cv2
import os
import sys
sys.path.append("../")
from utils import get_center_of_bbox,get_bbox_width
def draw_ellipse(frame,bbox,color,track_id=None):
    y2 = bbox[3]
    x_center,_ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(frame,(int(x_center),int(y2)),axes=(int(width),int(0.35*width)),angle=0,startAngle=45.0,endAngle=235.0,color=color,thickness=2,lineType=cv2.LINE_4)
    
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = int(x_center-rectangle_width//2)
    x2_rect = int(x_center+rectangle_width//2)
    y1_rect = int((y2-rectangle_height//2)+15)
    y2_rect = int((y2+rectangle_height//2)+15)
    if track_id is not None:
        cv2.rectangle(frame,(x1_rect,y1_rect),(x2_rect,y2_rect),color,cv2.FILLED)
        x1_text = x1_rect+12
        if track_id>99:
            track_id-=10
        cv2.putText(frame,str(track_id),(x1_text,y1_rect+15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
    return frame
    