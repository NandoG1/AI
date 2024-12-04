from flask import Flask, request, jsonify, render_template, Response, send_from_directory
import torch
import cv2
import os
import pathlib
import os
import numpy as np
# Load YOLOv5 model

pathlib.PosixPath = pathlib.WindowsPath
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

# def process_image(image_path):
#     img = cv2.imread(image_path)
#     results = model(img)
    
#     # Access predictions
#     predictions = results.pred[0]  # Tensor of detections
    
#     # Filter by confidence > 50%
#     filtered_predictions = predictions[predictions[:, 4] > 0.5]
    
#     # Extract class IDs and names if there are valid predictions
#     if len(filtered_predictions) > 0:
#         class_ids = filtered_predictions[:, 5].cpu().numpy().astype(int).tolist()  # Convert to list
#         class_names = [model.names[cid] for cid in class_ids]
#     else:
#         class_ids = []  # No detections
#         class_names = []  # No detections

#     # Render only the filtered predictions
#     for idx, det in enumerate(filtered_predictions):
#         # Re-create detection tensor for rendering
#         results.pred[0] = filtered_predictions

#     # Save output image with rendered detections
#     output_img = results.render()[0]
#     output_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(image_path)}")
#     cv2.imwrite(output_path, output_img)
    
#     return {
#         "result": "Image processed successfully",
#         "output": output_path,
#         "class_ids": class_ids,
#         "class_names": class_names
#     }

def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Extract predictions
    predictions = results.pred[0]  # Tensor of detections

    # Filter predictions by confidence > 50%
    filtered_predictions = predictions[predictions[:, 4] > 0.5]

    # Handle case where no predictions meet the threshold
    if len(filtered_predictions) == 0:
        return {
            "result": "No objects detected with confidence > 50%",
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

        results = model(frame)

        # Filter predictions by confidence
        predictions = results.pred[0]
        filtered_predictions = predictions[predictions[:, 4] > 0.5]

        # Annotate frame
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

##

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     output_path = os.path.join(RESULTS_FOLDER, f'output_{os.path.basename(video_path)}')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
#                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

#     detected_classes = set()  # Use a set to store unique class IDs

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         results = model(frame)  # YOLOv5 inference
        
#         # Filter detections based on confidence
#         predictions = results.pred[0]  # Access predictions tensor
#         filtered_predictions = predictions[predictions[:, 4] > 0.5]  # Confidence > 50%
        
#         # Extract class IDs for this frame
#         class_ids = filtered_predictions[:, 5].cpu().numpy()
        
#         # Add class IDs to the set (ensures uniqueness)
#         detected_classes.update(class_ids)

#         # Render the results on the frame (optional)
#         results.pred[0] = filtered_predictions
#         rendered_frame = results.render()[0]
#         out.write(rendered_frame)  # Write the frame to the output video
        
#         frame_count += 1

#     cap.release()
#     out.release()

#     # Convert set to list for easier processing in frontend
#     detected_classes_list = list(detected_classes)

#     # Construct the URL for the processed video
#     output_url = f"/static/results/{os.path.basename(output_path)}"
#     output_url = output_url.replace("\\", "/")  # Ensure forward slashes for URLs

#     return {
#         "result": "Video processed successfully",
#         "output": output_url,
#         "detected_classes": detected_classes_list  # Send back the unique detected classes
#     }


def webcam_feed():
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference with YOLOv5
            results = model(frame)

            # Get predictions: format [x1, y1, x2, y2, confidence, class]
            predictions = results.pred[0]  # YOLOv5 predictions for the current frame
            
            # Filter predictions with confidence > 50%
            filtered_predictions = predictions[predictions[:, 4] > 0.5]  # confidence is in the 5th column

            # Create an annotated frame (but don't use render() yet)
            annotated_frame = frame.copy()

            # Draw bounding boxes and labels manually for filtered predictions
            for det in filtered_predictions:
                x1, y1, x2, y2, confidence, class_id = det
                label = f"{model.names[int(class_id)]} {confidence:.2f}"
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text

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

######

# def process_image(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
    
#     # Perform inference using YOLOv5
#     results = model(img)
    
#     # Access predictions as a tensor
#     predictions = results.pred[0]  # This is a tensor
    
#     # Filter predictions where confidence > 50%
#     filtered_predictions = predictions[predictions[:, 4] > 0.5]
    
#     # Update the results object with the filtered detections
#     results.pred[0] = filtered_predictions
    
#     # Save output image with filtered detections
#     output_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(image_path)}")
#     cv2.imwrite(output_path, results.render()[0])
    
#     return {"result": "Image processed successfully", "output": output_path}

# #ini awal
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
#     if file:
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(file_path)

#         # Process image or video
#         if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             results = process_image(file_path)
#         elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
#             results = process_video(file_path)
#         else:
#             return jsonify({"error": "Unsupported file format"})
#         return jsonify(results)

# ini awal
# def process_image(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
#     results = model(img)  # Inference using YOLOv5
#     output_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(image_path)}")
#     cv2.imwrite(output_path, results.render()[0])  # Save output image
#     return {"result": "Image processed successfully", "output": output_path}


# ini new

# def get_class_ids(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
    
#     # Perform inference using YOLOv5
#     results = model(img)
    
#     # Access predictions
#     predictions = results.pred[0]  # Predictions tensor (shape: [num_detections, 6])
    
#     # Extract class IDs
#     class_ids = predictions[:, 5].cpu().numpy()  # Class IDs as a NumPy array
    
#     # Map class IDs to names using model.names
#     class_names = model.names  # List or dictionary of class names
#     detected_classes = [class_names[int(cls_id)] for cls_id in class_ids]
    
#     return class_ids, detected_classes





# yg lama ini
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     output_path = os.path.join(RESULTS_FOLDER, f'output_{os.path.basename(video_path)}')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
#                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         results = model(frame)  # YOLOv5 inference
        
#         # Filter detections based on confidence
#         predictions = results.pred[0]  # Access predictions tensor
#         filtered_predictions = predictions[predictions[:, 4] > 0.5]  # Confidence > 50%
        
#         # Update results with filtered predictions
#         results.pred[0] = filtered_predictions
        
#         # Render frame with updated results
#         rendered_frame = results.render()[0]
#         out.write(rendered_frame)  # Write frame to output video
        
#         frame_count += 1
    
#     cap.release()
#     out.release()
#     return {"result": "Video processed successfully", "output": output_path}



# ini awal
# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     output_path = os.path.join(RESULTS_FOLDER, f'output_{os.path.basename(video_path)}')
    
#     # Gunakan codec 'mp4v' untuk format .mp4
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Pastikan video writer diinisialisasi dengan benar
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         results = model(frame)  # YOLOv5 inference
#         out.write(results.render()[0])  # Tulis frame hasil deteksi
        
#     cap.release()
#     out.release()

#     # Verifikasi file output
#     print(f"Video saved at: {output_path}")
#     return {"result": "Video processed successfully", "output": f"/static/results/{os.path.basename(output_path)}"}



# broke
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
#     if file:
#         file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         file.save(file_path)

#         # Process image or video
#         if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             results = process_image(file_path)
#         elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
#             results = process_video(file_path)
#         else:
#             return jsonify({"error": "Unsupported file format"})
        
#         # Return the HTML page with detected classes and video/image output
#         return render_template('index.html', result=results["result"], output=results["output"], detected_classes=results["detected_classes"])

# def process_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     output_path = os.path.join(RESULTS_FOLDER, f'output_{os.path.basename(video_path)}')
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
#                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
#     all_detected_classes = set()  # Set to store all detected classes across frames

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         results = model(frame)  # YOLOv5 inference
        
#         # Filter out detections with confidence < 50%
#         for det in results.xyxy[0]:
#             confidence = det[4].item()
#             class_id = int(det[5].item())
#             if confidence > 0.5:
#                 all_detected_classes.add(model.names[class_id])  # Add class name to the set
        
#         out.write(results.render()[0])  # Write output frames to video
#         frame_count += 1
    
#     cap.release()
#     out.release()

#     # Return the classes detected and the path to the output video
#     return {"result": "Video processed successfully", "output": output_path, "detected_classes": list(all_detected_classes)}

# def process_image(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)
#     results = model(img)  # YOLOv5 inference

#     # Filter out detections with confidence < 50%
#     detected_classes = []
#     for det in results.xyxy[0]:  # Each detection is in the form [x1, y1, x2, y2, confidence, class_id]
#         confidence = det[4].item()  # Confidence score
#         class_id = int(det[5].item())  # Class ID
#         if confidence > 0.5:  # Only keep detections with confidence > 50%
#             detected_classes.append(model.names[class_id])  # Get the class name from the model

#     # Remove duplicates in the list of detected classes
#     detected_classes = list(set(detected_classes))

#     # Render the image with detections
#     output_path = os.path.join(RESULTS_FOLDER, f"result_{os.path.basename(image_path)}")
#     img_with_detections = results.render()[0]  # Render the image with bounding boxes
#     cv2.imwrite(output_path, img_with_detections)  # Save output image

#     # Return the classes detected and the path to the output image
#     return {"result": "Image processed successfully", "output": output_path, "detected_classes": detected_classes}

@app.route('/static/results/<path:filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
