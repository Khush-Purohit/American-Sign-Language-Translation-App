import cv2
import os

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 500

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    
    print(f'Get ready for collecting data for class {j}')
    print('Press k to start capturing images and q to quit')
    counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        
        if key == ord('k'):
            print(f'Starting capture for class {j}')
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)
        counter += 1
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print(f'Completed collecting {counter} images for class {j}')

cap.release()
cv2.destroyAllWindows()