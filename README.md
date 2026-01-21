# Human Action Detection (Video Action Recognition)

Human Action Detection is a deep learning and computer vision project that identifies and classifies human actions from video files or live webcam streams. The system learns both spatial information (what is visible in each frame) and temporal information (how motion changes over time) to predict actions such as walking, running, jumping, sitting, standing, clapping, and waving.

## Project Goals
- Detect and classify human actions from videos
- Support inference on video files and real-time webcam input
- Provide a clean and modular pipeline for training and deployment

## Approach
1. Video Input: Read frames from a video file or webcam.
2. Preprocessing: Resize frames, normalize pixel values, and create fixed-length frame sequences.
3. Model: Use a video-based deep learning model such as CNN+LSTM, 3D-CNN, or Transformer-based architecture.
4. Prediction: Output the predicted action label with confidence score.
5. Visualization: Display predictions on frames and optionally save the output video.

## Dataset
Common datasets that can be used:
- UCF101
- HMDB51
- Kinetics (subset recommended for training)
You can also create a custom dataset by collecting labeled action videos and extracting frame sequences.

## Folder Structure
- data/           Dataset videos or extracted frames
- notebooks/      Experiments and exploration
- src/            Training and inference scripts
- models/         Saved trained models
- outputs/        Predicted videos, logs, and results

## Installation
1. Clone the repository:
   git clone <repo_url>
   cd <repo_name>

2. Install dependencies:
   pip install -r requirements.txt

## Usage
Train the model:
python src/train.py --data_path data/ --epochs 20 --batch_size 8

Test on a video:
python src/predict.py --video_path sample.mp4

Run on webcam:
python src/webcam.py

## Evaluation
Model performance can be measured using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## Output
The system displays the predicted action label and confidence score on the video frames. Optionally, results can be saved to the outputs/ folder.

## Future Improvements
- Improve accuracy using larger datasets and augmentation
- Add multi-person action detection
- Deploy as a web app using FastAPI/Streamlit
- Optimize inference speed using ONNX/TensorRT

## License
This project is intended for learning and academic use.
