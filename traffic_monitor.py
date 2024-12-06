import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

class TrafficMonitor:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        self.vehicle_count = defaultdict(int)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.frame_height = None
        self.frame_width = None
        self.previous_detections = []
        self.wrong_way_count = 0

    def initialize_regions(self, frame):
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.left_region = int(self.frame_width * 0.3)
        self.right_region = int(self.frame_width * 0.7)

    def process_frame(self, frame):
        if self.frame_height is None:
            self.initialize_regions(frame)

        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 480))
        
        results = self.model(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                if cls in self.vehicle_classes and conf > 0.3:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'class': cls,
                        'conf': conf
                    })
                    
                    # Count vehicles
                    self.vehicle_count[cls] += 1
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{self.model.names[cls]}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.check_wrong_way(detections)
        self.add_info_to_frame(frame)
        self.previous_detections = detections
        
        return frame

    def check_wrong_way(self, current_detections):
        for det in current_detections:
            center_x = det['center'][0]
            if center_x > self.right_region:
                self.wrong_way_count += 1

    def add_info_to_frame(self, frame):
        # Add vehicle counts
        y_position = 30
        for cls in self.vehicle_classes:
            text = f"{self.model.names[cls]}: {self.vehicle_count[cls]}"
            cv2.putText(frame, text, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_position += 30
        
        # Add wrong way count
        cv2.putText(frame, f"Wrong Way: {self.wrong_way_count}", 
                   (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    # Initialize traffic monitor
    monitor = TrafficMonitor()
    
    # Replace 'traffic_video.mp4' with your video file name
    video_path = 'traffic_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = 'output_traffic.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame = monitor.process_frame(frame)
        
        # Write frame to output video
        out.write(processed_frame)
        
        # Display the processed frame
        cv2.imshow('Traffic Monitoring', processed_frame)
        
        # Break on 'q' key or ESC
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()