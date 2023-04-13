# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:22:58 2023

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, Response, make_response
import cv2
import datetime
from keras.models import load_model
import keras.utils as image
import matplotlib.pyplot as plt
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="b7e545bdc1cd425f90111f7de9709c8f",client_secret="e52cc05f937f4f76b1ba5bc38ea849e4"))
playlist_limit = 5
song_limit_per_playlist = 20


model = load_model('SavedModel1.h5')
app = Flask(__name__,static_url_path='/static')
emotion_detected = ''

@app.route('/')
def home():
    now = datetime.datetime.now()
    hour = now.hour
    return render_template('index.html',hour=hour)



def screenshot():
    print("hello")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    #img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 27:
         # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
         # SPACE pressed
           img_name = "test.jpg"
           cv2.imwrite(img_name, frame)
           print("Picture Captured!")
    cam.release()
    cv2.destroyAllWindows()
    
def facecrop(image): 
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)
    if img is None:
        print("Error: Image not loaded.")
        return
    faces = cascade.detectMultiScale(img)
    if len(faces) == 0:
        print("Error: No faces detected.")
        return
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        sub_face = img[y:y+h, x:x+w]
        if sub_face is not None:
            img_name = "capture1.jpg"
            cv2.imwrite(img_name, sub_face)
            print("Face cropped and saved to capture1.jpg")
        else:
            print("Error: Sub-face not cropped.")
    cv2.imshow(image, img)
    cv2.destroyAllWindows()

def emotion_analysis(emotions):
    objects = ('angry','disgust','fear','happy','sad','surprise','neutral')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos,emotions,align='center',alpha=0.5)
    plt.xticks(y_pos,objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    res = max(emotions)
    j = 0
    for i in emotions:
        if i==res:
            break
        else:
            j+=1
    Emotion=str(objects[j])
    print('Emotion Detected : '+ Emotion)
    global emotion_detected
    emotion_detected = Emotion
    print('Accuracy : '+ str(res*100))
    plt.savefig('static\\emotion_plot.jpg')
    plt.close()
    
    with open('static\\emotion_plot.jpg','rb') as f:
        image_data = f.read()
    response = Response(image_data,content_type='image/jpeg')
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['content-Type'] = 'image/png'
    return response

@app.route('/capture')
def capture():
    # Call the screenshot() function to capture an image
    screenshot()

    # Load the captured image from file and crop the face
    img_path = 'test.jpg'
    facecrop(img_path)

    # Load the cropped image and process it with Keras
    img = image.load_img('capture1.jpg',
                            grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = model.predict(x)
    final_emotion = emotion_analysis(custom[0])

    # Display the image using matplotlib
    true_image = image.load_img('capture1.jpg')
    plt.imshow(true_image)
    plt.show()
    plt.close()

    # Return the emotion predicted by the model as a JSON object
    return 'capture.jpg'


def songs_by_emotion(emotion):
    results = sp.search(q=emotion,type='playlist',limit=playlist_limit)
    gs = []
    for el in results['playlists']['items']:
        temp = {}
        temp['playlist_name'] = el['name']
        temp['playlist_href'] = el['href']
        temp['playlist_id'] = el['id']
        temp['playlist_spotify_link'] = el['external_urls']['spotify']
        gs.append(temp)
    final_playlist_songs = gs
    for i in range(len(gs)):
        res = sp.playlist(playlist_id = gs[i]['playlist_id'])
        srn = res['tracks']['items'][0:song_limit_per_playlist]
        tlist = []
        for el in srn:
            tlist.append(el['track']['name'])
        final_playlist_songs[i]['playlist_songs'] = tlist
    return final_playlist_songs

def format_songs(final_playlist_songs):
    output = ''
    for el in final_playlist_songs:
        output += f"<strong>Playlist Name:</strong> {el['playlist_name']}<br>"
        output += f"<strong>Playlist Link:</strong> <a href='{el['playlist_href']}'>{el['playlist_href']}</a><br>"
        output += f"<strong>Playlist Spotify Link:</strong> {el['playlist_spotify_link']}<br>"
        output += "<strong>Playlist Songs:</strong> <br>"
        for i, song in enumerate(el['playlist_songs']):
            output += f"{i+1}) {song}<br>"
        output += "<br>"
    return output



@app.route('/playlist')
def playlist():
    print("hello")
    print(emotion_detected)
    final_playlist_songs = songs_by_emotion(emotion_detected)
    playlist_html = format_songs(final_playlist_songs)
    return render_template('playlist.html', final_playlist_songs=final_playlist_songs, emotion=emotion_detected)


if __name__ == "__main__":
    app.run()