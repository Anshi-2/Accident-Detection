import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video_path = r"C:\Users\Nitish Maurya\Downloads\Accident-Detection-System-main (3)\Accident-Detection-System-main\Accident-Detection-System-main\Demo2.mp4"
    output_folder = r"C:\Users\admin\Downloads\wetransfer_zip_2024-02-22_0704\Accident-Detection-System-main\Accident-Detection-System-main\accident detected"
    video = cv2.VideoCapture(video_path)  # for camera use video = cv2.VideoCapture(0)
    
    # Get video details
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer for saving clips
    video_writer = None
    clip_counter = 1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob_percentage = round(prob[0][0] * 100, 2)

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob_percentage}", (20, 30), font, 1, (255, 255, 0), 2)

            # Save video clip when accident probability is greater than 97%
            if prob_percentage > 97:
                if video_writer is None:
                    clip_path = os.path.join(output_folder, f"accident_{clip_counter}_detected.avi")
                    video_writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

                video_writer.write(frame)
            else:
                # Close video writer when accident probability is below 97%
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    clip_counter += 1

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        cv2.imshow('Video', frame)

    # Release the video writer at the end
    if video_writer is not None:
        video_writer.release()

    video.release()
    cv2.destroyAllWindows()

if _name_ == '_main_':
    startapplication()