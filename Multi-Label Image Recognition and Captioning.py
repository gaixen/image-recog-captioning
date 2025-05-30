from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the pre-trained model
model = MobileNetV2(weights='imagenet')

def predict_objects(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return decode_predictions(preds, top=5)[0]

# Simple caption generator based on detected objects
def generate_caption(predictions):
    caption = "This image contains "
    for obj in predictions:
        caption += obj[1] + ", "
    return caption.strip(", ")
