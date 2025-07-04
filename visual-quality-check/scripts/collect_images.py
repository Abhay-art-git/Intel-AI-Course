import cv2
import os

# Ask user which label to collect
label = input("Enter label name (good/defective): ").strip().lower()
save_dir = f"../datasets/{label}"
os.makedirs(save_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
img_count = 0

print("Press 's' to save image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show preview
    cv2.imshow("Image Collector", frame)

    key = cv2.waitKey(1)

    # Save image on key press
    if key == ord('s'):
        img_path = os.path.join(save_dir, f"{label}_{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        img_count += 1

    # Quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
