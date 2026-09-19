import cv2

# 0 usually means your default/built-in webcam.
# If you have multiple cameras, you may need to try 1 or 2 instead.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    cv2.imshow("Webcam Test", frame)

    # Wait 1 millisecond for a key press; if it's 'q', break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()