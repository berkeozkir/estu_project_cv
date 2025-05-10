import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os
import sys
from collections import defaultdict
from ultralytics import YOLO
import threading

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global path prefix relative to script location
DATA_PREFIX = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
YOLO_DIR = os.path.join(DATA_PREFIX, "yolo")

# Create necessary directories if they don't exist
os.makedirs(YOLO_DIR, exist_ok=True)

class VideoGroundTruthVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ground Truth Video Visualizer")
        self.root.geometry("1280x800")
        
        # Initialize color mapping for bounding box visualization
        self.colors = {
            'motorbike': (255, 0, 0),    # Red
            'DHelmet': (0, 255, 0),      # Green
            'DNoHelmet': (0, 0, 255),    # Blue
            'P1Helmet': (255, 255, 0),   # Yellow
            'P1NoHelmet': (255, 0, 255), # Magenta
            'P2Helmet': (0, 255, 255),   # Cyan
            'P2NoHelmet': (128, 0, 128), # Purple
            'P0Helmet': (255, 165, 0),   # Orange
            'P0NoHelmet': (165, 42, 42)  # Brown
        }
        
        # YOLO model variables
        self.yolo_model = None
        self.yolo_results = {}  # Store YOLO results by frame
        self.yolo_model_path = None
        self.show_yolo = True
        self.show_gt = True
        self.available_models = {
            "YOLOv8n": os.path.join(YOLO_DIR, "yolov8n.pt"),
            "YOLOv8s": os.path.join(YOLO_DIR, "yolov8s.pt"),
            "YOLOv8m": os.path.join(YOLO_DIR, "yolov8m.pt"),
            "YOLOv8l": os.path.join(YOLO_DIR, "yolov8l.pt"),
            "YOLOv8x": os.path.join(YOLO_DIR, "yolov8x.pt")
        }
        self.loading_model = False
        self.model_download_progress = 0
        
        # Initialize video variables
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 0
        self.playing = False
        self.delay = 100  # milliseconds
        self.photo = None
        self.video_id = None
        self.width = 640
        self.height = 480
        
        # Load ground truth data
        try:
            self.load_data()
            # Setup UI components
            self.setup_ui()
            
            # Add copyright with absolute positioning that stays in bottom right
            self.copyright_label = tk.Label(
                self.root, 
                text="Developed by Berke Özkır and John Doe © 2024", 
                font=("Arial", 12),
                bg="light gray",
                fg="black",
                padx=5,
                pady=2
            )
            # Use relative positioning that adapts to window size
            self.copyright_label.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
    
    def load_data(self):
        """Load ground truth data from gt.txt"""
        try:
            # Load ground truth data
            columns = ['video_id', 'frame', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'class_id']
            self.gt_data = pd.read_csv(os.path.join(DATA_PREFIX, 'gt.txt'), header=None, names=columns)
            
            # Load class labels
            self.labels = {}
            with open(os.path.join(DATA_PREFIX, 'labels.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split(', ')
                    if len(parts) == 2:
                        self.labels[int(parts[0])] = parts[1]
            
            # Map class_id to label names
            self.gt_data['class_name'] = self.gt_data['class_id'].map(self.labels)
            
            # Get list of available videos
            self.video_ids = sorted(self.gt_data['video_id'].unique())
            
            # Organize annotations by video_id and frame
            self.annotations = defaultdict(lambda: defaultdict(list))
            for _, row in self.gt_data.iterrows():
                video_id = row['video_id']
                frame = row['frame']
                self.annotations[video_id][frame].append({
                    'bb_left': int(row['bb_left']),
                    'bb_top': int(row['bb_top']),
                    'bb_width': int(row['bb_width']),
                    'bb_height': int(row['bb_height']),
                    'class_id': int(row['class_id']),
                    'class_name': row['class_name']
                })
            
            print(f"Loaded {len(self.gt_data)} annotations for {len(self.video_ids)} videos")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def setup_ui(self):
        """Setup the UI components"""
        # Main frame - make it responsive to window resizing
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel - fixed height but responsive width
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Video selection dropdown
        controls_left = ttk.Frame(control_frame)
        controls_left.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(controls_left, text="Select Video:").pack(side=tk.LEFT, padx=(0, 5))
        self.video_combo = ttk.Combobox(
            controls_left, 
            values=[f"{vid:03d}.mp4" for vid in self.video_ids],
            width=10
        )
        self.video_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.video_combo.bind("<<ComboboxSelected>>", self.on_video_selected)
        
        # Browse button
        self.browse_btn = ttk.Button(
            controls_left, 
            text="Browse...", 
            command=self.browse_video
        )
        self.browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # YOLO Model Selection
        ttk.Label(controls_left, text="YOLO Model:").pack(side=tk.LEFT, padx=(10, 5))
        self.model_combo = ttk.Combobox(
            controls_left,
            values=list(self.available_models.keys()),
            width=10
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.model_combo.current(0)  # Select first model as default
        
        # Load model button
        self.load_model_btn = ttk.Button(
            controls_left,
            text="Load Model",
            command=self.load_yolo_model
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Toggle buttons for YOLO and GT
        self.yolo_var = tk.BooleanVar(value=True)
        self.gt_var = tk.BooleanVar(value=True)
        
        self.yolo_toggle = ttk.Checkbutton(
            controls_left,
            text="YOLO",
            variable=self.yolo_var,
            command=self.toggle_yolo
        )
        self.yolo_toggle.pack(side=tk.LEFT, padx=(10, 5))
        
        self.gt_toggle = ttk.Checkbutton(
            controls_left,
            text="Ground Truth",
            variable=self.gt_var,
            command=self.toggle_gt
        )
        self.gt_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        # Model loading progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            variable=self.progress_var
        )
        
        # Playback controls - centered
        controls_center = ttk.Frame(control_frame)
        controls_center.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        
        playback_frame = ttk.Frame(controls_center)
        playback_frame.pack(anchor=tk.CENTER)
        
        self.play_btn = ttk.Button(
            playback_frame, 
            text="Play", 
            command=self.toggle_play,
            state=tk.DISABLED,
            width=8
        )
        self.play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.prev_btn = ttk.Button(
            playback_frame, 
            text="<< Prev", 
            command=self.prev_frame,
            state=tk.DISABLED,
            width=8
        )
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.next_btn = ttk.Button(
            playback_frame, 
            text="Next >>", 
            command=self.next_frame,
            state=tk.DISABLED,
            width=8
        )
        self.next_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Frame info - right aligned
        controls_right = ttk.Frame(control_frame)
        controls_right.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.frame_label = ttk.Label(controls_right, text="Frame: 0/0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Create paned window for resizable sections
        paned_window = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Video display - takes up most space and resizes with window
        self.canvas_frame = ttk.Frame(paned_window)
        paned_window.add(self.canvas_frame, weight=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", width=640, height=480)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bottom panels contained in another frame
        bottom_frame = ttk.Frame(paned_window)
        paned_window.add(bottom_frame, weight=1)
        
        # Timeline slider
        slider_frame = ttk.LabelFrame(bottom_frame, text="Timeline")
        slider_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.slider.pack(fill=tk.X, padx=10, pady=5)
        self.slider.state(['disabled'])
        
        # Create a horizontal paned window for the bottom panels
        bottom_paned = ttk.PanedWindow(bottom_frame, orient=tk.HORIZONTAL)
        bottom_paned.pack(fill=tk.BOTH, expand=True)
        
        # Annotation display
        info_frame = ttk.LabelFrame(bottom_paned, text="Annotations")
        bottom_paned.add(info_frame, weight=2)
        
        # Make the text widget resize with the container
        self.annotation_text = tk.Text(info_frame, height=5, wrap=tk.WORD)
        self.annotation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar for annotations
        annotation_scroll = ttk.Scrollbar(info_frame, command=self.annotation_text.yview)
        annotation_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.annotation_text.config(yscrollcommand=annotation_scroll.set)
        
        # Color legend frame
        legend_frame = ttk.LabelFrame(bottom_paned, text="Color Legend")
        bottom_paned.add(legend_frame, weight=1)
        
        # Create a grid for the legend in a scrolled frame
        legend_canvas = tk.Canvas(legend_frame)
        legend_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        legend_scroll = ttk.Scrollbar(legend_frame, orient=tk.VERTICAL, command=legend_canvas.yview)
        legend_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        legend_canvas.configure(yscrollcommand=legend_scroll.set)
        legend_canvas.bind('<Configure>', lambda e: legend_canvas.configure(scrollregion=legend_canvas.bbox('all')))
        
        legend_grid = ttk.Frame(legend_canvas)
        legend_canvas.create_window((0,0), window=legend_grid, anchor=tk.NW)
        
        # Add legend items - Convert BGR colors to RGB for display
        row, col = 0, 0
        for class_name, color_bgr in self.colors.items():
            # Convert BGR to RGB for display
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            
            # Create a colored box with proper RGB color
            color_box = tk.Canvas(legend_grid, width=15, height=15, bg='#%02x%02x%02x' % color_rgb, highlightthickness=1)
            color_box.grid(row=row, column=col*2, padx=(5, 0), pady=2)
            
            # Create a label with same color as bounding box
            label = ttk.Label(legend_grid, text=class_name, foreground='#%02x%02x%02x' % color_rgb)
            label.grid(row=row, column=col*2+1, padx=(2, 10), pady=2, sticky='w')
            
            col += 1
            if col > 1:  # 2 columns to fit better
                col = 0
                row += 1
        
        # Add YOLO detection color
        yolo_color_rgb = (0, 255, 0)  # Green in RGB
        color_box = tk.Canvas(legend_grid, width=15, height=15, bg='#%02x%02x%02x' % yolo_color_rgb, highlightthickness=1)
        color_box.grid(row=row, column=col*2, padx=(5, 0), pady=2)
        
        label = ttk.Label(legend_grid, text="YOLO Detection", foreground='#%02x%02x%02x' % yolo_color_rgb)
        label.grid(row=row, column=col*2+1, padx=(2, 10), pady=2, sticky='w')
    
    def on_video_selected(self, event=None):
        """Handler for video selection"""
        self.load_selected_video()
    
    def load_selected_video(self):
        """Load the selected video from the dropdown"""
        selected = self.video_combo.get()
        if selected:
            try:
                video_id = int(selected.split('.')[0])
                video_path = os.path.join(DATA_PREFIX, 'videos', selected)
                if os.path.exists(video_path):
                    self.load_video(video_path, video_id)
                else:
                    print(f"Video file {video_path} not found")
            except Exception as e:
                print(f"Error loading selected video: {e}")
                import traceback
                traceback.print_exc()
    
    def browse_video(self):
        """Open file dialog to browse for a video"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov")]
            )
            if file_path:
                # Extract video_id from filename
                filename = os.path.basename(file_path)
                if filename.endswith('.mp4') and filename[:-4].isdigit():
                    video_id = int(filename[:-4])
                    self.load_video(file_path, video_id)
                else:
                    print("Unable to determine video ID from filename")
        except Exception as e:
            print(f"Error browsing for video: {e}")
            import traceback
            traceback.print_exc()
    
    def load_video(self, video_path, video_id):
        """Load the video and initialize the player"""
        try:
            # Close any open video
            if self.cap is not None:
                self.cap.release()
            
            self.video_path = video_path
            self.video_id = video_id
            
            # Open the video
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                print(f"Error opening video file {video_path}")
                return
            
            # Get video properties
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Reset playback state
            self.current_frame = 0
            self.playing = False
            
            # Clear previous YOLO results
            self.yolo_results = {}
            
            # Update UI
            self.slider.config(to=self.frame_count-1)
            self.slider.set(0)
            self.slider.state(['!disabled'])
            
            self.play_btn.config(text="Play", state=tk.NORMAL)
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            
            # Run YOLO detection on first frame if model is loaded
            if self.yolo_model is not None:
                self.process_yolo_detection()
            
            # Display the first frame
            self.show_frame()
            
            print(f"Loaded video {self.video_id} with {self.frame_count} frames at {self.fps} FPS")
        except Exception as e:
            print(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
    
    def show_frame(self):
        """Display the current frame with annotations"""
        if self.cap is None:
            return
        
        try:
            # Seek to the current frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error reading frame {self.current_frame}")
                return
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Track annotations for display in text box
            all_annotations = []
            
            # Draw ground truth annotations on the frame if enabled
            if self.show_gt:
                frame_annotations = self.annotations[self.video_id].get(self.current_frame + 1, [])
                all_annotations.extend(frame_annotations)
                
                for annotation in frame_annotations:
                    x = annotation['bb_left']
                    y = annotation['bb_top']
                    w = annotation['bb_width']
                    h = annotation['bb_height']
                    class_name = annotation['class_name']
                    
                    # Get color for this class
                    color = self.colors.get(class_name, (255, 255, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label
                    cv2.putText(
                        display_frame, class_name, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )
            
            # Process and draw YOLO detections if model is loaded and enabled
            if self.show_yolo and self.yolo_model is not None:
                # Process current frame if needed
                if self.current_frame not in self.yolo_results:
                    self.process_yolo_detection()
                
                # Get results for current frame
                results = self.yolo_results.get(self.current_frame)
                
                if results is not None:
                    # Process YOLO detections
                    boxes = results.boxes
                    yolo_annotations = []
                    
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        cls_name = results.names[cls]
                        
                        # Create a YOLO annotation
                        yolo_annotation = {
                            'bb_left': x1,
                            'bb_top': y1,
                            'bb_width': x2 - x1,
                            'bb_height': y2 - y1,
                            'class_name': f"YOLO:{cls_name}",
                            'confidence': conf
                        }
                        yolo_annotations.append(yolo_annotation)
                        
                        # Draw bounding box with confidence
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(
                            display_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                        )
                    
                    all_annotations.extend(yolo_annotations)
            
            # Convert frame to format suitable for tkinter
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Ensure we have valid dimensions (use defaults if canvas not yet properly sized)
            if canvas_width < 10:
                canvas_width = 640
            if canvas_height < 10:
                canvas_height = 480
            
            # Calculate aspect ratio
            img_ratio = self.width / self.height
            canvas_ratio = canvas_width / canvas_height
            
            # Resize to fit canvas while maintaining aspect ratio
            if img_ratio > canvas_ratio:
                # Width constrained
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                # Height constrained
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=img)
            
            # Clear canvas and update with new image
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
            
            # Update frame info
            self.frame_label.config(text=f"Frame: {self.current_frame + 1}/{self.frame_count}")
            
            # Update annotation text
            self.update_annotation_text(all_annotations)
            
            # Update slider without triggering the callback
            self.slider.set(self.current_frame)
        
        except Exception as e:
            print(f"Error showing frame: {e}")
            import traceback
            traceback.print_exc()
    
    def update_annotation_text(self, annotations):
        """Update the annotation text display"""
        try:
            self.annotation_text.delete(1.0, tk.END)
            
            if not annotations:
                self.annotation_text.insert(tk.END, "No annotations for this frame")
                return
            
            text = f"Frame {self.current_frame + 1} Annotations:\n"
            
            # Separate ground truth and YOLO annotations
            gt_annotations = [ann for ann in annotations if not ann.get('class_name', '').startswith('YOLO:')]
            yolo_annotations = [ann for ann in annotations if ann.get('class_name', '').startswith('YOLO:')]
            
            # Count objects by class
            class_counts = defaultdict(int)
            for ann in gt_annotations:
                class_counts[ann['class_name']] += 1
                
            yolo_class_counts = defaultdict(int)
            for ann in yolo_annotations:
                yolo_class_counts[ann['class_name'].replace('YOLO:', '')] += 1
            
            # Display summary for ground truth
            if gt_annotations and self.show_gt:
                text += "Ground Truth Summary: "
                summary_parts = [f"{count} {class_name}" for class_name, count in class_counts.items()]
                text += ", ".join(summary_parts) + "\n"
            
            # Display summary for YOLO
            if yolo_annotations and self.show_yolo:
                text += "YOLO Detection Summary: "
                summary_parts = [f"{count} {class_name}" for class_name, count in yolo_class_counts.items()]
                text += ", ".join(summary_parts) + "\n"
            
            text += "\n"
            
            # Display detailed ground truth annotations
            if gt_annotations and self.show_gt:
                text += "Ground Truth:\n"
                for i, ann in enumerate(gt_annotations[:10]):
                    text += (f"{i+1}. {ann['class_name']} at "
                            f"({ann['bb_left']}, {ann['bb_top']}, "
                            f"width: {ann['bb_width']}, height: {ann['bb_height']})\n")
                
                if len(gt_annotations) > 10:
                    text += f"... and {len(gt_annotations) - 10} more GT annotations\n"
                
                text += "\n"
            
            # Display detailed YOLO annotations
            if yolo_annotations and self.show_yolo:
                text += "YOLO Detections:\n"
                for i, ann in enumerate(yolo_annotations[:10]):
                    text += (f"{i+1}. {ann['class_name'].replace('YOLO:', '')} "
                            f"(conf: {ann['confidence']:.2f}) at "
                            f"({ann['bb_left']}, {ann['bb_top']}, "
                            f"width: {ann['bb_width']}, height: {ann['bb_height']})\n")
                
                if len(yolo_annotations) > 10:
                    text += f"... and {len(yolo_annotations) - 10} more YOLO detections"
            
            self.annotation_text.insert(tk.END, text)
        except Exception as e:
            print(f"Error updating annotation text: {e}")
            import traceback
            traceback.print_exc()
    
    def toggle_play(self):
        """Toggle between play and pause"""
        if self.cap is None:
            return
        
        try:
            self.playing = not self.playing
            
            if self.playing:
                self.play_btn.config(text="Pause")
                self.play_video()
            else:
                self.play_btn.config(text="Play")
        except Exception as e:
            print(f"Error toggling play: {e}")
    
    def play_video(self):
        """Auto-advance frames during playback"""
        if not self.playing or self.cap is None:
            return
            
        try:
            if self.current_frame < self.frame_count - 1:
                self.current_frame += 1
                self.show_frame()
                # Calculate delay based on FPS, but ensure it's not too fast
                delay = max(1, int(1000 / self.fps))
                self.root.after(delay, self.play_video)
            else:
                # Reached the end
                self.playing = False
                self.play_btn.config(text="Play")
        except Exception as e:
            print(f"Error during playback: {e}")
            self.playing = False
            self.play_btn.config(text="Play")
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.cap is None:
            return
            
        try:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.show_frame()
        except Exception as e:
            print(f"Error going to previous frame: {e}")
    
    def next_frame(self):
        """Go to next frame"""
        if self.cap is None:
            return
            
        try:
            if self.current_frame < self.frame_count - 1:
                self.current_frame += 1
                self.show_frame()
        except Exception as e:
            print(f"Error going to next frame: {e}")
    
    def on_slider_change(self, value):
        """Handle slider position change"""
        if self.cap is None:
            return
            
        try:
            # Get value as float and convert to int
            new_frame = int(float(value))
            
            # Only update if the frame has actually changed
            if new_frame != self.current_frame:
                # Pause playback when manually scrubbing
                if self.playing:
                    self.playing = False
                    self.play_btn.config(text="Play")
                
                # Update current frame and display
                self.current_frame = new_frame
                self.show_frame()
        except Exception as e:
            print(f"Error handling slider change: {e}")
    
    def cleanup(self):
        """Release resources on exit"""
        try:
            if self.cap is not None:
                self.cap.release()
                print("Released video capture")
            
            # Clear YOLO results to free memory
            if hasattr(self, 'yolo_results'):
                self.yolo_results = {}
                print("Cleared YOLO results")
                
            # Clear YOLO model
            if hasattr(self, 'yolo_model'):
                self.yolo_model = None
                print("Released YOLO model")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
    
    def load_yolo_model(self):
        """Load the selected YOLO model"""
        if self.loading_model:
            return
            
        try:
            selected_model = self.model_combo.get()
            if not selected_model:
                print("No model selected")
                return
                
            model_filename = self.available_models[selected_model]
            
            # Disable button during loading
            self.load_model_btn.config(state=tk.DISABLED, text="Loading...")
            self.loading_model = True
            self.progress_bar.pack(side=tk.RIGHT, padx=(10, 10), before=self.frame_label)
            
            # Load model in a separate thread to avoid freezing UI
            def load_model_thread():
                try:
                    self.yolo_model = YOLO(model_filename)
                    self.yolo_model_path = model_filename
                    
                    # Clear previous results
                    self.yolo_results = {}
                    
                    # If video is loaded, run inference on current frame
                    if self.cap is not None and self.current_frame >= 0:
                        self.process_yolo_detection()
                        self.show_frame()
                    
                    self.root.after(0, lambda: self.load_model_btn.config(
                        state=tk.NORMAL, 
                        text=f"{selected_model} Loaded"
                    ))
                    self.root.after(0, lambda: self.progress_bar.pack_forget())
                    
                except Exception as e:
                    print(f"Error loading YOLO model: {e}")
                    self.root.after(0, lambda: self.load_model_btn.config(
                        state=tk.NORMAL, 
                        text="Load Failed"
                    ))
                    self.root.after(0, lambda: self.progress_bar.pack_forget())
                    
                finally:
                    self.loading_model = False
            
            # Start the thread
            threading.Thread(target=load_model_thread, daemon=True).start()
            
        except Exception as e:
            print(f"Error initiating model loading: {e}")
            self.load_model_btn.config(state=tk.NORMAL, text="Load Failed")
            self.progress_bar.pack_forget()
            self.loading_model = False
    
    def process_yolo_detection(self):
        """Process YOLO detection on the current frame"""
        if self.yolo_model is None or self.cap is None:
            return None
            
        # Check if we already have results for this frame
        if self.current_frame in self.yolo_results:
            return self.yolo_results[self.current_frame]
            
        try:
            # Seek to the current frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            
            # Read the frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error reading frame {self.current_frame} for YOLO processing")
                return None
                
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Store results
            self.yolo_results[self.current_frame] = results[0]
            
            return results[0]
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def toggle_yolo(self):
        """Toggle YOLO detection display"""
        self.show_yolo = self.yolo_var.get()
        if self.cap is not None:
            self.show_frame()
    
    def toggle_gt(self):
        """Toggle ground truth display"""
        self.show_gt = self.gt_var.get()
        if self.cap is not None:
            self.show_frame()

def main():
    """Main function to start the application"""
    try:
        root = tk.Tk()
        app = VideoGroundTruthVisualizer(root)
        
        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
        
        root.mainloop()
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 