from flask import Flask, jsonify, render_template, request, redirect, url_for, flash  
import os                                   
import tensorflow as tf                     
from PIL import Image                       
import numpy as np                     
import pickle                               
from sklearn.feature_extraction.text import TfidfVectorizer  
import re                                   
import nltk
import werkzeug                                 
nltk.download('popular')                    
from nltk.stem import WordNetLemmatizer     
lemmatizer = WordNetLemmatizer()            
import json                                 
import random                               
from keras.models import load_model         
from flask_mysqldb import MySQL             

app = Flask(__name__)                       
UPLOAD_FOLDER = 'static/uploads'            
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['MYSQL_HOST'] = 'localhost'      
app.config['MYSQL_USER'] = 'root'           
app.config['MYSQL_PASSWORD'] = ''           
app.config['MYSQL_DB'] = 'capstone_user_input'  

mysql = MySQL(app)                          

# Rute untuk tampilan form login
@app.route('/', methods=['GET', 'POST'])    
def login():                                
    if request.method == 'POST':            
        username = request.form['username'] 
        password = request.form['password'] 
        
        cur = mysql.connection.cursor()     
        cur.execute("SELECT * FROM register WHERE username=%s AND password=%s", (username, password))  
        result = cur.fetchall()             
        cur.close()                         

        if len(result) == 1:                
            return redirect(url_for('home'))
        else:
            return "Login gagal. Coba lagi."

    return render_template('login.html')    

# Rute untuk API upload gambar flutter
@app.route('/upload', methods=["POST"])
def upload():
    if(request.method == "POST"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uplodedimages/" + filename)
        return jsonify({
            "message": "Gambar Berhasil di Upload"
        })


# Rute untuk tampilan form register
@app.route('/register', methods=['GET', 'POST'])  
def register():                             
    if request.method == 'POST':            
        username = request.form['username'] 
        password = request.form['password'] 
        
        cur = mysql.connection.cursor()     
        cur.execute("INSERT INTO register (username, password) VALUES (%s, %s)", (username, password))
        mysql.connection.commit()           
        cur.close()                         
        
        return redirect(url_for('login'))   

    return render_template('register.html') 

# Rute untuk tampilan menu home
@app.route('/home')                         
def home():                                 
    return render_template('home.html')     

# Rute untuk tampilan menu deteksi
# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')    
interpreter.allocate_tensors()                                 
input_details = interpreter.get_input_details()                 
output_details = interpreter.get_output_details()               


def detect_wound(image_path):                                   
    image = Image.open(image_path).resize((320, 320))           
    image_array = np.array(image) / 255.0                       
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)    
    interpreter.set_tensor(input_details[0]['index'], image_array)          
    interpreter.invoke()                                        
    output = interpreter.get_tensor(output_details[0]['index']) 
    return output                                               


@app.route('/deteksi', methods=['GET', 'POST'])                 
def deteksi():
    result = None                                               
    uploaded_image = None                                       
    if request.method == 'POST':                                
        if 'image' not in request.files:                        
            return render_template('deteksi.html', result='Error: No file part')  
        file = request.files['image']                           
        if file.filename == '':
            return render_template('deteksi.html', result='Error: No selected file')
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  
            file.save(filename)                                

            relative_path = filename.replace(app.config['UPLOAD_FOLDER'] + '/', '')  
            if relative_path.startswith('static/'):
                relative_path = relative_path.replace('static/', '', 1)

            output = detect_wound(filename)                     
            class_id = np.argmax(output[0])                     
            classes = ['luka_bakar', 'luka_lecet', 'luka_memar', 'luka_robek', 'luka_sayatan', 'luka_tusuk']  
            result = classes[class_id]                          
            uploaded_image = relative_path                      
        
            # Simpan informasi deteksi ke dalam tabel deteksi_history
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO deteksi_history (image_path, result) VALUES (%s, %s)", (relative_path, result))
            mysql.connection.commit()
            cur.close()

    return render_template('deteksi.html', result=result, uploaded_image=uploaded_image)


# Rute untuk tampilan menu deteksi_history
@app.route('/deteksi_history')                                  
def deteksi_history():
    # Ambil data histori deteksi dari tabel deteksi_history
    cur = mysql.connection.cursor()                             
    cur.execute("SELECT image_path, result, timestamp FROM deteksi_history ORDER BY timestamp DESC")  
    deteksi_history_list = cur.fetchall()                       
    cur.close()                                                 

    print(deteksi_history_list)                                 

    return render_template('deteksi_history.html', deteksi_history_list=deteksi_history_list)  

# Rute untuk tampilan menu deteksi_history untuk diakses dari Flutter
@app.route('/deteksi_history_api', methods=['GET'])
def deteksi_history_flutter():
    # Ambil data histori deteksi dari tabel deteksi_history
    cur = mysql.connection.cursor()                             
    cur.execute("SELECT image_path, result, timestamp FROM deteksi_history ORDER BY timestamp DESC")  
    deteksi_history_list = cur.fetchall()                       
    cur.close()                                                 

    # Konversi hasil ke format JSON
    history_json = []
    for history in deteksi_history_list:
        history_dict = {
            'image_path': history[0],
            'result': history[1],
            'timestamp': history[2].strftime('%Y-%m-%d %H:%M:%S')  # Ubah format timestamp sesuai kebutuhan
        }
        history_json.append(history_dict)
    
    # Mengembalikan data dalam format JSON
    return jsonify(history_json)

# Rute untuk tampilan menu chatbot
# Load Chatbot model
with open("Train_Bot.json", "r") as json_file:                  
    intents = json.load(json_file)                              

model = load_model("chatbot.h5")                                
words = pickle.load(open("words.pkl", "rb"))                    
classes = pickle.load(open("classes.pkl", "rb"))               

def clean_up_sentence(sentence):
    # tokenisasi pola - membagi kata-kata menjadi sebuah larik.
    sentence_words = nltk.word_tokenize(sentence)
    # stem setiap kata - membuat bentuk kata pendek               
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  
    return sentence_words

# mengembalikan bag of words array: 0 atau 1 untuk setiap kata dalam tas kata yang ada dalam kalimat.
def bow(sentence, words, show_details=True):
    # tokenisasi pola                    
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix                
    bag = [0]*len(words)                                        
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # memberikan nilai 1 jika kata saat ini berada pada posisi kosakata. 
                bag[i] = 1                                      
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # menghilangkan prediksi di bawah ambang batas.
    p = bow(sentence, words,show_details=False)                 
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # mengurutkan berdasarkan kekuatan probabilitas.
    results.sort(key=lambda x: x[1], reverse=True)              
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')                          
    return chatbot_response(userText)                           

# Rute untuk tampilan menu feedback
# Load model SVM
with open('model_svm.pkl', 'rb') as model_file:                 
    svm_model = pickle.load(model_file)                         

# Load TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:     
    tfidf_vectorizer = pickle.load(vectorizer_file)             

@app.route('/feedback')                                         
def feedback():
    return render_template('feedback.html')                     

@app.route('/predict', methods=['POST'])                        
def predict():
    if request.method == 'POST':                                
        feedback = request.form['feedback']                     

        # Lakukan analisis sentimen
        sentiment = predict_sentiment(feedback)

        # Simpan ke database
        save_to_database(feedback, sentiment)                   

        # Hitung persentase hasil sentimen
        positive_percentage, neutral_percentage, negative_percentage = calculate_sentiment_percentage()  
        
        # Ambil data dari database untuk ditampilkan di HTML
        cur = mysql.connection.cursor()                         
        cur.execute("SELECT feedback_user, hasil_sentimen_analisis FROM feedback ORDER BY id DESC LIMIT 1")
        result = cur.fetchone()
        cur.close()

        user_input = result[0]
        sentiment_result = result[1]

        return render_template('feedback.html', user_input=user_input, sentiment_result=sentiment_result, positive_percentage=positive_percentage, neutral_percentage=neutral_percentage, negative_percentage=negative_percentage)
        
    return render_template('feedback.html')                     

def predict_sentiment(feedback):
    # Lakukan pra-pemrosesan pada feedback (sesuai dengan langkah preprocessing)
    transformed_feedback = tfidf_vectorizer.transform([feedback])  

    # Lakukan prediksi sentimen menggunakan model
    prediction = svm_model.predict(transformed_feedback)        

    if prediction[0] in [1, 2]:
        return 'negatif'
    elif prediction[0] == 3:
        return 'netral'
    elif prediction[0] in [4, 5]:
        return 'positif'

def save_to_database(feedback, sentiment):
    # Simpan data ke database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO feedback (feedback_user, hasil_sentimen_analisis) VALUES (%s, %s)", (feedback, sentiment))
    mysql.connection.commit()
    cur.close()

def calculate_sentiment_percentage():
    # Ambil jumlah feedback untuk masing-masing sentimen
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='positif'")
    positive_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='netral'")
    neutral_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM feedback WHERE hasil_sentimen_analisis='negatif'")
    negative_count = cur.fetchone()[0]

    total_count = positive_count + neutral_count + negative_count

    # Hitung persentase
    positive_percentage = (positive_count / total_count) * 100
    neutral_percentage = (neutral_count / total_count) * 100
    negative_percentage = (negative_count / total_count) * 100

    return positive_percentage, neutral_percentage, negative_percentage


if __name__ == '__main__':
    app.run(debug=True)