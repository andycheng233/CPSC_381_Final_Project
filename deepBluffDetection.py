import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Dropout
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import mediapipe as mp
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FacialDeceptionDetector:
    def __init__(self, model_path=None):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # for cnn facial feature extraction
        self.features_model = None

        # for lstm classification on truth or lie
        self.sequence_model = None

        # for standardizing features
        self.scaler = StandardScaler()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_models()
    
    def _build_models(self):
        
        # Build CNN feature extraction model
        cnn_input = Input(shape=(224, 224, 3))  # Input image size
        
        # CNN layers
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        
        # Get feature vector
        features = Flatten()(x)
        features = Dense(512, activation='relu')(features)
        features = Dropout(0.5)(features)
        features = Dense(256, activation='relu')(features)
        
        self.features_model = Model(inputs=cnn_input, outputs=features)
        
        # LSTM model for sequence classification
        sequence_input = Input(shape=(None, 256))  # Variable sequence length, 256 features per step
        x = LSTM(128, return_sequences=True)(sequence_input)
        x = Dropout(0.4)(x)
        x = LSTM(64)(x)
        x = Dropout(0.4)(x)
        output = Dense(1, activation='sigmoid')(x)  # Binary classification: lying or not
        
        self.sequence_model = Model(inputs=sequence_input, outputs=output)
        self.sequence_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
    
    def preprocess_face_for_cnn(self, image):
        """Preprocess face for CNN input"""
        height, width = image.shape[:2]

        # get face points from mediapipe
        results = self.face_mesh(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # crop the image so it only focuses on the face
        if results.multi_face_landmarks:
            # converting to image points
            coordinates = [(int(point.x * width), int(point.y * height)) for point in results.multi_face_landmarks[0].landmark]
            
            x_min, x_max = min(coordinates[0]), max(coordinates[0])
            y_min, y_max = min(coordinates[1]), max(coordinates[1])

            # adding some padding on all sides in case
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(width, x_max + padding)
            y_max = min(height, y_max + padding)

            # crop/resize the image and normalize pixel values
            face_crop = cv2.resize(image[y_min:y_max, x_min:x_max], (224,224)) / 255.0
            
            return face_crop
        
        else:
            print("I don't see face :(, we take whole image instead and pray its there")
            image_crop = cv2.resize(image, (224,224)) / 255.0

            return image_crop

    def extract_features(self, image):
        """Extract facial features using CNN"""
        # Process image through MediaPipe to get face
        face_image = self.preprocess_face_for_cnn(image)
        # Expand dimensions for batch processing
        face_image = np.expand_dims(face_image, axis=0)
        # Extract features using CNN
        features = self.features_model.predict(face_image)
        return features[0]  # Return feature vector
    
    def process_video(self, video_path, sequence_length=30):
        """Process video and extract sequence of facial features"""
        cap = cv2.VideoCapture(video_path)
        features_sequence = []
        
        while len(features_sequence) < sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            features = self.extract_features(frame)
            features_sequence.append(features)
            
        cap.release()
        
        # Pad sequence if needed
        if len(features_sequence) < sequence_length:
            # Pad with zeros if video is shorter than required sequence length
            padding = np.zeros((sequence_length - len(features_sequence), 256))
            features_sequence = np.vstack((np.array(features_sequence), padding))
        else:
            features_sequence = np.array(features_sequence[:sequence_length])
            
        return np.array([features_sequence])  # Return as batch
    
    def predict(self, video_path):
        """Predict if person is lying or not"""
        features_sequence = self.process_video(video_path)
        prediction = self.sequence_model.predict(features_sequence)
        return prediction[0][0]  # Return probability of lying
    
    def train(self, train_videos, train_labels, validation_split=0.2, epochs=50, batch_size=16):
        """Train the model with video data"""
        # Process all training videos
        X_sequences = []
        for video_path in train_videos:
            features_sequence = self.process_video(video_path)
            X_sequences.append(features_sequence[0])
        
        X_train = np.array(X_sequences)
        y_train = np.array(train_labels)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.sequence_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping]
        )
        
        return history
    
    def save_model(self, path):
        """Save both models"""
        self.features_model.save(f"{path}_features.h5")
        self.sequence_model.save(f"{path}_sequence.h5")
    
    def load_model(self, path):
        """Load both models"""
        self.features_model = tf.keras.models.load_model(f"{path}_features.h5")
        self.sequence_model = tf.keras.models.load_model(f"{path}_sequence.h5")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = FacialDeceptionDetector()
    
    # Example: how to load training data
    # Assuming you have a CSV with video paths and labels
    # data = pd.read_csv('deception_dataset.csv')
    # train_videos = data['video_path'].tolist()
    # train_labels = data['is_lying'].tolist()  # 1 for lying, 0 for truth
    
    # # Train the model
    history = detector.train(train_videos, train_labels, epochs=50)
    
    # # Save the trained model
    detector.save_model('deception_detector')
    
    # # Later, load and predict
    # detector = FacialDeceptionDetector(model_path='deception_detector')
    # probability = detector.predict('test_video.mp4')
    # print(f"Probability of lying: {probability:.2f}")