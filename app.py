from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import torch
import cv2
import os
import pathlib
import os
import numpy as np
# Load YOLOv5 model

WHITELISTED_CLASSES = ["bird", "dog", "cat", "bear", "elephant", "giraffe", "zebra", "horse", "cow", "sheep"]

pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Extract predictions
    predictions = results.pred[0]  # Tensor of detections

    # Filter predictions by confidence > 0.25 and class name
    filtered_predictions = []
    for det in predictions:
        x1, y1, x2, y2, conf, class_id = det.tolist()  # Convert tensor to list
        class_name = model.names[int(class_id)]
        if conf > 0.25 and class_name in WHITELISTED_CLASSES:
            filtered_predictions.append([x1, y1, x2, y2, conf, class_id])

    # Convert filtered predictions to a tensor if needed
    if filtered_predictions:
        filtered_predictions = torch.tensor(filtered_predictions)

    # Handle case where no predictions meet the threshold
    if len(filtered_predictions) == 0:
        return {
            "result": "No objects detected with confidence > 0.25",
            "output": "",
            "class_ids": [],
            "class_names": []
        }

    # Extract class IDs and names
    class_ids = filtered_predictions[:, 5].cpu().numpy().astype(int).tolist()
    class_names = [model.names[cid] for cid in class_ids]

    # Update `results.pred[0]` with filtered predictions
    results.pred[0] = filtered_predictions

    # Render and save the filtered output
    output_img = results.render()[0]  # Render now uses filtered predictions
    output_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(image_path)}")
    output_path = output_path.replace("\\", "/")  # Ensure forward slashes for URLs
    output_url = f"/static/results/{os.path.basename(output_path)}"
    cv2.imwrite(output_path, output_img)

    return {
        "result": "Image processed successfully",
        "output": output_url,
        "class_ids": class_ids,
        "class_names": class_names
    }

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    output_path = os.path.join(RESULTS_FOLDER, f"output_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference with YOLOv5
        results = model(frame)
        predictions = results.pred[0]  # YOLOv5 predictions for the current frame

        # Filter predictions
        filtered_predictions = []
        for det in predictions:
            x1, y1, x2, y2, conf, class_id = det.tolist()  # Convert tensor to list
            class_name = model.names[int(class_id)]
            if conf > 0.25 and class_name in WHITELISTED_CLASSES:
                filtered_predictions.append([x1, y1, x2, y2, conf, class_id])

        if filtered_predictions:
            filtered_predictions = torch.tensor(filtered_predictions)

        # Annotate frame with filtered predictions
        annotated_frame = frame.copy()
        for det in filtered_predictions:
            x1, y1, x2, y2, conf, class_id = det
            label = f"{model.names[int(class_id)]} {conf:.2f}"
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    return {"result": "Video processed successfully", "output": f"/static/results/output_{os.path.basename(video_path)}"}



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Initialize the results
        results = {}

        # Process image or video
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            detections = process_image(file_path)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            detections = process_video(file_path)
        else:
            return jsonify({"error": "Unsupported file format"})

        # Extract detected class IDs and names
        class_ids = detections.get("class_ids", [])
        class_names = detections.get("class_names", [])

        # Include detected classes in the response
        results.update(detections)
        results["detected_classes"] = [{"id": cid, "name": cname} for cid, cname in zip(class_ids, class_names)]

        return jsonify(results)
    
@app.route('/upload-image', methods=['POST'])
def upload_file1():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Initialize the results
        results = {}

        # Process image or video
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            detections = process_image(file_path)
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            detections = process_video(file_path)
        else:
            return jsonify({"error": "Unsupported file format"})

        # Extract detected class IDs and names
        class_ids = detections.get("class_ids", [])
        class_names = detections.get("class_names", [])

        # Include detected classes in the response
        results.update(detections)
        results["detected_classes"] = [{"id": cid, "name": cname} for cid, cname in zip(class_ids, class_names)]

        return jsonify(results)




# for change the box of predict
def plot_one_box(xyxy, img, label=None, color=(255, 0, 0), line_thickness=2):
    # Draw bounding box
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)
    if label:
        # Add label text
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = cv2.getTextSize(label, font, font_scale, 1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # Filled box
        cv2.putText(img, label, (c1[0], c1[1] - 2), font, font_scale, [225, 255, 255], 1, cv2.LINE_AA)


def webcam_feed():
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference with YOLOv5
            results = model(frame)
            predictions = results.pred[0]  # YOLOv5 predictions for the current frame

            # Filter predictions
            filtered_predictions = []
            for det in predictions:
                x1, y1, x2, y2, conf, class_id = det.tolist()  # Convert tensor to list
                class_name = model.names[int(class_id)]
                if conf > 0.25 and class_name in WHITELISTED_CLASSES:
                    filtered_predictions.append([x1, y1, x2, y2, conf, class_id])

            if filtered_predictions:
                filtered_predictions = torch.tensor(filtered_predictions)

            # Annotate frame with filtered predictions
            annotated_frame = frame.copy()
            for det in filtered_predictions:
                x1, y1, x2, y2, conf, class_id = det
                label = f"{model.names[int(class_id)]} {conf:.2f}"
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        cap.release()  # Release the webcam when the stream stops


@app.route('/webcam')
def webcam():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/results/<path:filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
