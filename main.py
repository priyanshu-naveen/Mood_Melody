import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import tkinter as tk
import random

model = load_model("C:/Users/HP/Desktop/Python/Projects/Major(MoodMelody)/emotion_model.keras")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emoji_dict = {
    'angry': "😠",
    'disgust': "🤢",
    'fear': "😨",
    'happy': "😃",
    'neutral': "😐",
    'sad': "😢",
    'surprise': "😲"
}


song_dict = {
    'angry': ["Bhaag Dhoondh -- Divine",
    "Kaala Jaadu -- Brahmāstra",
    "Sher Aaya Sher -- Gully Boy",
    "Jhoome Jo Pathaan -- Pathaan",
    "Zinda Banda -- Jawan",
    "Raataan Lambiyan (Remix) -- DJ Shadow",
    "Karma Theme -- Animal",
    "Bezubaan -- ABCD",
    "Apna Time Aayega -- Gully Boy",
    "Bholenath -- Emiway Bantai"
],
    'disgust': [  "Besharam Rang -- Pathaan",
    "Pyaar Hota Kayi Baar Hai -- Tu Jhoothi Main Makkaar",
    "Character Dheela 2.0 -- Shehzada",
    "Naach Meri Rani -- Guru Randhawa",
    "Alcoholia -- Vikram Vedha",
    "Chedkhaniyaan -- Heeramandi",
    "Nasha -- Badshah",
    "Kudiyee Ni Teri -- Selfiee",
    "Yentamma -- Kisi Ka Bhai Kisi Ki Jaan",
    "Jee Ni Karda -- Satyameva Jayate 2"
],
    'fear': ["Rakt Charitra Theme -- RGV",
    "Azaadi -- Gully Boy",
    "Fear Song -- Devara",
    "Darkhast -- Shivaay",
    "Tandav Theme -- Tandav",
    "Laal Ishq -- Ram-Leela",
    "Aigiri Nandini (Rock Version) -- Mahakaal",
    "Shaitan Ka Saala -- Housefull 4",
    "Khwabon Ke Parindey (Haunting Reprise) -- ZNMD",
    "Jee Le Zaraa (Instrumental) -- Talaash"
],
    'happy': [ "What Jhumka? -- Rocky Aur Rani",
    "Dil Se Dil Tak -- Bawaal",
    "Tere Vaaste -- Zara Hatke Zara Bachke",
    "Heeriye -- Arijit Singh",
    "Jab Se Ishq Hua Hai -- Pooja Verma Studio",
    "Tune Gale Lagaya -- Pooja Verma Studio",
    "Vaaste -- Dhvani Bhanushali",
    "Dance Ka Bhoot -- Brahmāstra",
    "Sauda Khara Khara -- Good Newwz",
    "Gallan Goodiyan -- Dil Dhadakne Do"
],
    'neutral': [ "Qaafirana -- Kedarnath",
    "Kalank Title Track -- Kalank",
    "Jaan Ban Gaye -- Khuda Haafiz",
    "Jitni Dafa -- Parmanu",
    "Nazm Nazm -- Bareilly Ki Barfi",
    "Zaalima -- Raees",
    "Dil Maang Raha Hai -- Ghost",
    "Dariya -- Baar Baar Dekho",
    "Naino Ne Baandhi -- Gold",
    "Aaj Se Teri -- Padman"
],
    'sad': [ "O Bedardeya -- Tu Jhoothi Main Makkaar",
    "Phir Na Milenge -- Broken",
    "Hamari Adhuri Kahani -- Title Track",
    "Toot Gaya -- Broken",
    "Dil Ka Kya -- Metro In Dino",
    "Agar Tum Saath Ho -- Tamasha",
    "Baaton Ko Teri -- All Is Well",
    "Sunn Raha Hai -- Aashiqui 2",
    "Bekhayali -- Kabir Singh",
    "Main Dhoondne Ko Zamaane Mein -- Heartless"
],
    'surprise': ["Chand Baaliyan -- Aditya",
    "Doobey -- Gehraiyaan",
    "Apna Bana Le -- Bhediya",
    "Naatu Naatu -- RRR (Hindi)",
    "Kesariya (Dance Mix) -- Brahmāstra",
    "Gann Deva -- Street Dancer 3D",
    "Deva Shree Ganesha -- Agneepath",
    "Sadda Dil Vi Tu -- ABCD",
    "Mourya Re -- Don",
    "Shree Ganeshay Dheemahi -- Viruddh"
]
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

root = tk.Tk()
root.title("MoodMelody")

video_label = tk.Label(root)
video_label.pack()

emoji_display = tk.Label(root, text="", font=("Arial", 50))
emoji_display.pack()

song_label = tk.Label(root, text="Songs will appear here", font=("Arial", 14), justify="left")
song_label.pack(pady=10)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_count = 0
current_emotion = "neutral"
last_emotion = None

def update_frame():
    global frame_count, current_emotion, last_emotion

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        
        if frame_count % 10 == 0:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            prediction = model.predict(face, verbose=0)
            current_emotion = emotion_labels[np.argmax(prediction)]

    
            if current_emotion != last_emotion:
                if current_emotion in song_dict:
                    songs = random.sample(song_dict[current_emotion], min(3, len(song_dict[current_emotion])))
                    song_label.config(
                        text=f"Recommended Songs for {current_emotion.capitalize()}:\n" + "\n".join(songs)
                    )
                last_emotion = current_emotion

    frame_small = cv2.resize(frame, (400, 300))
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)


    emoji_display.config(text=emoji_dict[current_emotion])

    root.after(30, update_frame)

update_frame()
root.mainloop()
cap.release()