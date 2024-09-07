import pyttsx3

def generate_voice_caption(model, image_features):
    caption = model.predict(image_features)
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()
    return caption
