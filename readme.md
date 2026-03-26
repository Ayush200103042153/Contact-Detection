# Bat-Ball Contact Detection
Cricket Video Analysis - CSC 481 Final Project

**Team:** Ayush Patel & Krunal Kathiriya  
**Course:** CSC 481 – Intro to Image Processing  
**Term:** 2025-26 Winter  
**Instructor:** Kenny Davila Castellanos

---

## What this project does

This project tries to detect whether a cricket ball made contact with the bat in a short video clip. The idea came from the fact that professional cricket uses a system called Hot Spot which uses thermal cameras to detect contact, but those cameras cost like $50,000+ and most cricket matches don't have access to that. So we wanted to see if we could do something similar just using regular video and basic image processing.

The way it works is: you give it a video, draw a box around the bat in the first frame, and then it goes through every frame and tries to detect if there was a sudden motion change inside that region (which would mean the ball hit the bat). At the end it saves a new video with "CONTACT" or "NO CONTACT" written on each frame.

---

## Requirements

- Python 3.8 or higher
- opencv-python
- numpy
- matplotlib

To install everything just run:
```
pip install opencv-python numpy matplotlib
```

We used Python 3.10 when testing this so that should definitely work. Don't include your virtual environment folder in the zip if you're submitting this somewhere.

---

## How to run

1. Open `bat_ball_contact.py` and scroll to the bottom
2. Change the `input_video` path to wherever your video file is
3. Change the `output_video` path to where you want the result saved
4. Run it:

```
python bat_ball_contact.py
```

When it starts, a window will pop up showing the first frame of the video. Use your mouse to draw a rectangle around the bat area, then press **ENTER** or **SPACE** to confirm. After that it will process the whole video automatically.

When it's done you'll see something like:
```
[INFO] Saved output video to: output/output_contact_result.mp4
[INFO] Contact detected at frames: [42, 43, 44, 45]
Final decision: CONTACT occurred.
```

If you want to use it in your own code instead:

```python
from bat_ball_contact import detect_bat_ball_contact

contact, frames = detect_bat_ball_contact(
    video_path="clip.mp4",
    output_path="result.mp4",
    diff_thresh=35,
    min_contact_pixels=80,
    roi_blur_ksize=5
)
```

---

## How the pipeline works

We broke it into 5 steps:

**Step 1 - Frame Extraction**  
Just reads the video frame by frame using OpenCV. Stores each frame as a numpy array.

**Step 2 - Preprocessing**  
Converts each frame to grayscale and applies a Gaussian blur to reduce noise. This is important before doing frame differencing otherwise small random pixel changes mess everything up.

**Step 3 - ROI Selection**  
The user draws a bounding box around the bat in the first frame. All the detection only happens inside this box. This helped a lot with reducing false positives from the background (like fielders moving, crowd, etc.)

**Step 4 - Frame Differencing**  
Takes the absolute difference between two consecutive frames inside the ROI. Then thresholds it to get a binary image. After that we apply morphological opening to clean up noise and then dilate to strengthen the real motion areas.

**Step 5 - Contact Decision**  
Counts the non-zero pixels in the cleaned-up difference image. If that number is above `min_contact_pixels` then we say it's a contact frame. Draws a red overlay on the motion area and writes the annotated frame to the output video.

---

## Parameters you can tune

These three are the main ones that affect how sensitive the detection is:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| diff_thresh | 35 | How big a pixel change needs to be to count. Lower = more sensitive but more false positives |
| min_contact_pixels | 80 | Minimum number of changed pixels to call it a contact. Higher = stricter |
| roi_blur_ksize | 5 | Gaussian blur kernel size. Has to be an odd number |

We tested a bunch of different values for these and found 35/80/5 worked best on our dataset. If you're getting too many false positives try increasing diff_thresh. If it's missing contacts try lowering it.

---

## Dataset

We used three kinds of videos:

- YouTube clips of cricket highlights (edge catches, no-ball decisions, etc.)
- Some broadcast footage from professional matches
- Videos we recorded ourselves with a tennis ball and bat (this helped with ground truth since we knew exactly when contact happened)

Total was around 20-30 clips, each about 2-5 seconds long. We didn't include the videos in the zip because of size/copyright reasons.

---

## Results

On our custom controlled videos we got around 85% accuracy. The system worked pretty well when the camera was not moving and lighting was decent.

It struggled more on broadcast videos because:
- Motion blur from fast deliveries
- Camera panning around causes the whole frame to change which creates false positives
- Sometimes the ball is too small to create enough of a motion signature
- If the bat swings way outside the ROI we drew it won't detect it

---

## Known issues / limitations

- The ROI is fixed after the first frame. If the bat moves a lot it goes outside the box and we miss it
- We don't actually track the ball separately, just look at motion in the bat region
- Lighting changes (like floodlights flickering) can cause false positives
- We thought about adding audio analysis for the snick sound but didn't get to that

---


## Project Structure
```text
Bat-Ball_Contact_Detection/
├── src/
│   └── bat_ball_contact.py
├── input/
│   └── video.MP4
├── output/
│   └── output_contact_result.mp4
├── README.md
├── requirements.txt
└── Self_evaluation.pdf
```

---

## Future improvements

If we had more time we would have liked to:
- Add automatic bat detection so the ROI updates each frame instead of being static
- Try using optical flow instead of just frame differencing
- Add audio analysis for the snick sound
- Maybe try a simple ML classifier on the motion features

---

## Authors

Ayush Patel  
Krunal Kathiriya  
CSC 481 - Winter 2025-26