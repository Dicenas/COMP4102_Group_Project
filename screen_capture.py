import cv2
import os
import shutil

def main():
    # Set up the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Create or clear the output folder
    output_folder = './data_classifier/captures'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    img_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Define the rectangle for the capture area
        x, y, w, h = 400, 100, 200, 200 

        key = cv2.waitKey(1)
        if key == ord('q'):
            # Capture the defined square from the frame
            cropped_frame = frame[y:y+h, x:x+w]

            # Save the cropped frame
            img_filename = f"{output_folder}/captured_image{img_count}.jpg"
            cv2.imwrite(img_filename, cropped_frame)
            print(f"Image saved: {img_filename}")
            img_count += 1

        # Draw a rectangle around the capture area (moved this line here)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display instructions
        instructions = "Press 'q' to capture, 'esc' to exit"
        cv2.putText(frame, instructions, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Display the resulting frame with the rectangle
        cv2.imshow('Webcam', frame)

        if key == 27:  # Esc key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
