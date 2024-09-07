import cv2
from keras.models import load_model

# Load the pre-trained model
model = load_model('image_caption_model.h5')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and generate caption
    features = extract_features(frame)
    caption = model.predict(features)

    # Display the result
    cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Live Captioning', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
