import cv2
import numpy as np
import torch
import json
import pyvirtualcam
from test_i3d import run_on_tensor  # ensure the function is importable

# Set these as in your current setup
num_classes = 2000
weights = r"C:\Users\Ashwa\hackathon\WLASL\code\I3D\archived\asl2000\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
json_path = r"C:\Users\Ashwa\hackathon\WLASL\start_kit\WLASL_v0.3.json"
window_size = 100  # number of frames per inference window

# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

frames_list = []
predicted_word = None  # Holds the latest predicted gloss

# Set the virtual camera resolution and fps
virtual_width = 448
virtual_height = 448
virtual_fps = 20

print("Press 'q' to exit.")

with pyvirtualcam.Camera(
    width=virtual_width,
    height=virtual_height,
    fps=virtual_fps,
    fmt=pyvirtualcam.PixelFormat.BGR,
) as cam:
    print(f"Virtual camera running: {cam.device}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame: resize to 224x224 and normalize to [-1, 1]
        frame = cv2.resize(frame, (224, 224))
        frame = (frame / 255.0) * 2 - 1

        # Convert normalized frame back to displayable BGR in [0, 255]
        display_frame = ((frame + 1) / 2) * 255
        display_frame = display_frame.astype(np.uint8)

        # Convert to HSV and adjust saturation
        hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
        saturation_factor = 0.8
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 255).astype(np.uint8)
        adjusted_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Overlay predicted word (if available) as a text box in the middle right side
        if predicted_word is not None:
            text = f"{predicted_word}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7  # reduced font scale
            thickness = 1  # reduced thickness
            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
            text_w, text_h = text_size
            height, width, _ = adjusted_frame.shape
            margin = 10
            # Calculate bottom-left corner so that text is centered vertically on the right
            x = width - margin - text_w
            y = height // 2 + text_h // 2
            text_org = (x, y)
            # Draw background rectangle for better readability
            cv2.rectangle(
                adjusted_frame,
                (x - 5, y - text_h - 5),
                (x + text_w + 5, y + 5),
                (0, 0, 0),
                cv2.FILLED,
            )
            # Overlay the text in white color
            cv2.putText(
                adjusted_frame,
                text,
                text_org,
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Resize adjusted_frame for virtual camera transmission
        virtual_frame = cv2.resize(adjusted_frame, (virtual_width, virtual_height))
        # Send the frame to the virtual camera
        cam.send(virtual_frame)
        cam.sleep_until_next_frame()

        # Optionally show the processed frame locally
        cv2.imshow("Webcam", virtual_frame)
        frames_list.append(frame)

        # Check if we've accumulated enough frames
        if len(frames_list) == window_size:
            # Convert list to tensor (shape: T x H x W x C)
            frames_np = np.asarray(frames_list, dtype=np.float32)
            frames_tensor = torch.tensor(frames_np)
            # Rearrange dimensions to match model input: 1 x C x T x H x W
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # T x C x H x W
            frames_tensor = frames_tensor.unsqueeze(0)  # 1 x T x C x H x W
            frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)  # 1 x C x T x H x W

            print("Processing window of 100 frames...")
            predicted_labels = run_on_tensor(weights, frames_tensor, num_classes)

            # Load mapping from indices to gloss words
            with open(json_path, "r") as f:
                data = json.load(f)
            index_to_gloss = {i: entry["gloss"] for i, entry in enumerate(data)}

            print("Predicted glosses:")
            for idx in predicted_labels[-5:]:
                gloss = index_to_gloss.get(idx, "Unknown")
                print(f"Class {idx}: {gloss}")

            # Use the first predicted label for the overlay text box
            predicted_word = index_to_gloss.get(predicted_labels[0], "Unknown")

            # Reset the frame list to start the next window
            frames_list = []

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
