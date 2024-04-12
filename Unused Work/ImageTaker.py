import os
import cv2

DATA_DIR = './background_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100
square_size = 480  # Desired square resolution

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        # Determine dimensions for cropping to square
        height, width = frame.shape[:2]
        min_dim = min(height, width)
        # Adjust square_size if it's larger than the camera resolution
        crop_size = min(square_size, min_dim)
        
        start_x = width // 2 - crop_size // 2
        start_y = height // 2 - crop_size // 2

        cv2.putText(frame, 'Press q to collect images', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        # Crop to square
        cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
        cv2.imshow('frame', cropped_frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), cropped_frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
