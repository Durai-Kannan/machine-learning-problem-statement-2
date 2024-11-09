from flask import Flask, request, jsonify, render_template, send_from_directory,redirect,url_for,send_from_directory,flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def accident_detection():
    tr_data_dir = os.path.join("/content/drive/My Drive/dataset/train")
    tr_data = tf.keras.utils.image_dataset_from_directory(
                                tr_data_dir,image_size=(256, 256),
                                seed = 12332
                                ) 
    tr_data_iterator = tr_data.as_numpy_iterator()
    tr_batch = tr_data_iterator.next()
    def label_to_category(label):
        if(label == 1):
            return "No Accident"
        elif label == 0:
            return "Accident"
        else :
            return "error"
    tr_data = tr_data.map(lambda x,y: (x/255, y))

    print("Max pixel value : ",tr_batch[0].max())
    print("Min pixel value : ",tr_batch[0].min())
    val_data_dir = os.path.join("/content/drive/MyDrive/dataset/test")
    val_data = tf.keras.utils.image_dataset_from_directory(val_data_dir)
    val_data_iterator = val_data.as_numpy_iterator()
    val_batch = val_data_iterator.next()
    val_data = val_data.map(lambda x,y: (x/255, y))
    val_batch = val_data.as_numpy_iterator().next()
    
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    # Adding neural Layer
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(tr_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])
    model.save("//content/drive/MyDrive/accidents.keras")                                                     
    import cv2

    # load random samples from samples directory
    #random_data_dirname = os.path.join("/content/drive/MyDrive/dataset/test/Accident")
    #pics = [os.path.join(random_data_dirname, filename) for filename in os.listdir(random_data_dirname)]

    # load first file from samples
    sample = cv2.imread("/content/drive/MyDrive/dataset/sam/accident.jpeg", cv2.IMREAD_COLOR)
    sample = cv2.resize(sample, (256, 256))

    prediction = 1 - model.predict(np.expand_dims(sample/255, 0))

    if prediction >= 0.5: 
        label = 'Predicted class is Accident'
    else:
        label = 'Predicted class is Not Accident'

    plt.title(label)
    plt.imshow(sample)
    plt.show()

def traffic_detection():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Set the detection confidence threshold
    model.conf = 0.4  # Confidence threshold (adjust as needed)

    # Traffic jam threshold
    traffic_jam_threshold = 12

    # Capture video
    cap = cv2.VideoCapture('video.mp4')

    # Process video at 1 FPS
    fps = 1
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current timestamp in seconds
        current_time = time()
        
        # Only process the frame if 1 second has passed
        if current_time - prev_time >= 1.0:
            prev_time = current_time

            # Resize frame for faster processing (optional)
            frame = cv2.resize(frame, (800, 600))

            # Perform vehicle detection with YOLO
            results = model(frame)
            detections = results.pred[0]

            # Count vehicles in the current frame
            vehicle_count = 0

            for det in detections:
                # Extract bounding box and confidence
                x1, y1, x2, y2, confidence, cls = map(int, det[:6])

                # Filter for vehicles (e.g., cars, trucks)
                if cls in [2, 5, 7]:  # Class IDs for cars, buses, and trucks in COCO dataset
                    vehicle_count += 1
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'Vehicle {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check for traffic jam
            if vehicle_count > traffic_jam_threshold:
                traffic_status = "Traffic Jam"
            else:
                traffic_status = "Normal Traffic"

            # Display vehicle count and traffic status
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, traffic_status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the result
            cv2.imshow("Vehicle Detection and Counting", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def peak_hrs_detection():
    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Simulate a dataset (Replace this with your actual dataset)
data = {
    'Date': pd.date_range(start='2023-01-01', periods=365, freq='D'),
    'Bus_Count': np.random.randint(50, 200, size=365),
    'Event_Type': np.random.choice(['None', 'Concert', 'Sports', 'Festival'], size=365),
    'Festival': np.random.choice([0, 1], size=365),
    'Traffic_Level': np.random.randint(1, 5, size=365),  # Scale: 1 (Low) to 4 (High)
    'Weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=365)
}

df = pd.DataFrame(data)

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Preprocess the data
# Encode categorical variables
label_enc_event = LabelEncoder()
label_enc_weather = LabelEncoder()
df['Event_Type'] = label_enc_event.fit_transform(df['Event_Type'])
df['Weather'] = label_enc_weather.fit_transform(df['Weather'])

# Scale numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['Bus_Count', 'Event_Type', 'Festival', 'Traffic_Level', 'Weather']])
scaled_df = pd.DataFrame(scaled_features, columns=['Bus_Count', 'Event_Type', 'Festival', 'Traffic_Level', 'Weather'], index=df.index)

# Step 3: Prepare data for LSTM model
def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, 0])  # Predicting 'Bus_Count'
        return np.array(X), np.array(y)

        # Define sequence length
        SEQ_LENGTH = 7  # Using 7 days of past data to predict the next day
        X, y = create_sequences(scaled_df.values, SEQ_LENGTH)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Build the LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Step 5: Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        # Step 6: Evaluate the model
        train_loss = model.evaluate(X_train, y_train)
        test_loss = model.evaluate(X_test, y_test)
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Testing Loss: {test_loss:.4f}')

        # Step 7: Make predictions
        predictions = model.predict(X_test)

        # Inverse transform predictions and true values
        predictions_inversed = scaler.inverse_transform(np.concatenate(
            (predictions, np.zeros((predictions.shape[0], X.shape[2] - 1))), axis=1))[:, 0]
        y_test_inversed = scaler.inverse_transform(np.concatenate(
            (y_test.reshape(-1, 1), np.zeros((y_test.shape[0], X.shape[2] - 1))), axis=1))[:, 0]

        # Step 8: Visualize the results
        plt.figure(figsize=(10, 6))
        plt.plot(df.index[-len(y_test):], y_test_inversed, label='Actual Bus Count')
        plt.plot(df.index[-len(predictions):], predictions_inversed, label='Predicted Bus Count', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Bus Count')
        plt.title('Actual vs Predicted Bus Count')
        plt.legend()
        plt.show()




# Initialize the app
app = Flask(__name__)
app.secret_key = 'user1111'

# Configurations
app.config['SECRET_KEY'] = 'your_secret_key'  # Change this to a real secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking for performance

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect to login page if user is not authenticated

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    score = db.Column(db.Integer, default=0)  # New score column with default value 0


# Create the database (run once)
@app.got_first_request
def create_tables():
    db.create_all()


@app.route('/',methods=['GET', 'POST'])
def home():
   return render_template('home.html')

@app.route('/get_data',methods=['GET'])
def get_data():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    # Check if the user exists
    user = User.query.filter_by(username=username).first()
    email = user.query.filter_by(email = email)
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        flash('Login Successful!', 'success')
        return redirect(url_for('dashboard.html'))
    else:
        flash('Login Unsuccessful. Please check username and password.', 'danger')

# Login route
@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():        
    return render_template('sign_in.html')

# Sign-up route
@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Check if username or email already exists
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))

        # Create new user
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('sign_in'))

@app.route('/log_up', methods=['GET', 'POST'])
def log_up():
    return render_template('log_up.html')

@app.route('/dashboard')
def dashboard():
    return render_template('/dashboard,html',user = user)

@app.route('/get_travel_data', methods=['GET'])
def get_travel_data():
  start = request.args.get('start')
  end = request.args.get('end')



@app.route('/adminhome.html')
def adminhome():
    return render_template('adminhome.html')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/get_accident_data",methods=['GET'])
def get_accident_data():
  data = request.get_json()
  latitude = data.get('latitude')
  longitude = data.get('longitude')

  # Check if the 'image' file is part of the request
  if 'image' not in request.files:
      return 'No file part', 400

  file = request.files['image']

  # If no file is selected
  if file.filename == '':
      return 'No selected file', 400

  # Save the file with a secure filename
  if file:
      filename = file.filename  # You could also use `secure_filename(file.filename)` here for security
      filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      file.save(filepath)
      return f'Image successfully uploaded to {filepath}'




# Set a directory for saving uploaded images
if __name__ == '__main__':
    app.run(debug=True)
