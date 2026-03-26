import cv2
import numpy as np
import os


def detect_bat_ball_contact(
    video_path,
    output_path="output_contact_result.mp4",
    diff_thresh=35,
    min_contact_pixels=80,
    roi_blur_ksize=5
):


    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: " + video_path)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame for processing")

    # Ask user for ROI around bat on first frame
    print("[INFO] Draw ROI around bat and press ENTER (or SPACE) to confirm.")
    x, y, w, h = cv2.selectROI(
        "Select Bat ROI", prev_frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select Bat ROI")

    # Video writer configuration
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(prev_frame.shape[1])
    height = int(prev_frame.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Preprocess first frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (roi_blur_ksize, roi_blur_ksize), 0)

    frame_idx = 0
    contact_frames = []
    global_contact_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (roi_blur_ksize, roi_blur_ksize), 0)

        # Extract ROI around bat
        roi_prev = prev_gray[y:y + h, x:x + w]
        roi_curr = gray[y:y + h, x:x + w]

        # Frame differencing in ROI
        diff = cv2.absdiff(roi_prev, roi_curr)
        _, diff_thresh_img = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # Morphological filtering to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        diff_clean = cv2.morphologyEx(diff_thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
        diff_clean = cv2.dilate(diff_clean, kernel, iterations=1)

        # Motion measure
        motion_pixels = cv2.countNonZero(diff_clean)

        # Contact decision (simple threshold rule)
        contact_now = motion_pixels > min_contact_pixels
        if contact_now:
            contact_frames.append(frame_idx)
            global_contact_flag = True

        # Visualization
        vis_frame = frame.copy()

        # Draw ROI rectangle
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Build red mask inside ROI for motion
        color_mask = np.zeros_like(vis_frame[y:y + h, x:x + w])
        color_mask[:, :, 2] = diff_clean  # red channel

        # Overlay mask onto ROI area
        vis_frame[y:y + h, x:x + w] = cv2.addWeighted(
            vis_frame[y:y + h, x:x + w], 1.0, color_mask, 0.6, 0
        )

        # Text label
        label = "CONTACT" if contact_now else "NO CONTACT"
        color = (0, 0, 255) if contact_now else (0, 255, 0)
        cv2.putText(
            vis_frame,
            label,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

        # Write annotated frame
        out.write(vis_frame)

        # Update previous
        prev_gray = gray

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("[INFO] Saved output video to:", output_path)
    if global_contact_flag:
        print("[INFO] Contact detected at frames:", contact_frames)
    else:
        print("[INFO] No contact detected in this clip.")

    return global_contact_flag, contact_frames


if __name__ == "__main__":
    # input video path
    input_video = r"C:\Users\AYUSH PATEL\Desktop\MASTER\IMAGE PROCESSING\Bat-Ball_Contact_Detection\input\video.MP4"

    # Output folder and file
    output_folder = r"C:\Users\AYUSH PATEL\Desktop\MASTER\IMAGE PROCESSING\Bat-Ball_Contact_Detection\output"
    os.makedirs(output_folder, exist_ok=True)

    output_video = os.path.join(output_folder, "output_contact_result.mp4")

    contact, frames = detect_bat_ball_contact(
        input_video,
        output_path=output_video,
        diff_thresh=35,
        min_contact_pixels=80,
        roi_blur_ksize=5
    )

    if contact:
        print("Final decision: CONTACT occurred.")
    else:
        print("Final decision: NO CONTACT.")

