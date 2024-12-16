import os
import cv2
import time

# Set up data directory
DATA_DIR = './alphabet_data'
os.makedirs(DATA_DIR, exist_ok=True)

# Alphabet range (excluding 'j' and 'z')
alphabets = [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in ['j', 'z']]
dataset_size = 20

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

for letter in alphabets:
    letter_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_dir, exist_ok=True)

    print(f'Collecting data for letter "{letter.upper()}". Get ready!')

    # Ready screen
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        cv2.putText(frame, f'Get ready for "{letter.upper()}". Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            continue

        cv2.putText(frame, f'Capturing "{letter.upper()}" {counter + 1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):  # Capture on pressing 's'
            filename = os.path.join(letter_dir, f'{counter}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            counter += 1

        if cv2.waitKey(1) & 0xFF == ord('e'):  # Exit capturing for this letter
            print(f"Exiting data collection for letter {letter.upper()}")
            break

cap.release()
cv2.destroyAllWindows()
