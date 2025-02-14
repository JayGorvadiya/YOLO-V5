 #import torch
# import cv2
# import warnings
# import os

# # Suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Check and set the device (use MPS for Apple Silicon if available)
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# print(f"Using device: {device}")

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # Use 'yolov5s' for a lightweight model
# model.eval()  # Set the model to evaluation mode

# # Function to perform face detection and save the output image
# def detect_faces_and_save(image_path, output_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not load image from {image_path}")
#         return
    
#     original_image = image.copy()
    
#     # Convert BGR to RGB for the model
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Perform inference
#     results = model(image_rgb)
    
#     # Parse the results
#     detections = results.xyxy[0].cpu().numpy()  # Ensure results are on the CPU
#     for detection in detections:
#         x_min, y_min, x_max, y_max, confidence, class_id = detection
#         if int(class_id) == 0 and confidence > 0.5:  # Class ID 0 corresponds to "person"
#             # Draw bounding boxes on the original image
#             cv2.rectangle(original_image, 
#                           (int(x_min), int(y_min)), 
#                           (int(x_max), int(y_max)), 
#                           (0, 255, 0), 2)
#             label = f"Face: {confidence:.2f}"
#             cv2.putText(original_image, label, (int(x_min), int(y_min) - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Save the resulting image
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the output directory exists
#     cv2.imwrite(output_path, original_image)
#     print(f"Saved image with bounding boxes to {output_path}")

# # Example usage
# image_path = '/Users/sharmaparth/Downloads/Iphone_Images_4_Dec_2024/IMG_0926/IMG_0926.DNG'  # Replace with the path to your input image
# output_path = '/Users/sharmaparth/Downloads/Iphone_Images_4_Dec_2024/IMG_0926/IMG_output.jpg'       # Replace with the desired output file path
# detect_faces_and_save(image_path, output_path)


# 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# import torch
# import cv2
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Check and set the device (use MPS for Apple Silicon if available)
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# print(f"Using device: {device}")

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # Use 'yolov5s' for a lightweight model
# model.eval()  # Set the model to evaluation mode

# # Get class names from the model
# class_names = model.names

# # Function to preprocess the image (resize to 640x640)
# def preprocess_image(image):
#     # Convert BGR to RGB for the model
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image_rgb

# # Function to detect objects from the webcam
# def detect_objects_from_camera():
#     # Open the webcam (use 0 for the default webcam)
#     cap = cv2.VideoCapture(0)

#     while True:
#         # Read a frame from the webcam
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
        
#         # Preprocess the frame
#         image_rgb = preprocess_image(frame)
        
#         # Perform inference
#         results = model(image_rgb)
        
#         # Parse the results
#         detections = results.xyxy[0].cpu().numpy()  # Ensure results are on the CPU
        
#         # Draw bounding boxes and labels for detected objects
#         for detection in detections:
#             x_min, y_min, x_max, y_max, confidence, class_id = detection
#             if confidence > 0.5:  # Filter by confidence score
#                 class_name = class_names[int(class_id)]  # Get class label
#                 label = f"{class_name}: {confidence:.2f}"
                
#                 # Draw bounding box on the frame
#                 cv2.rectangle(frame, 
#                               (int(x_min), int(y_min)), 
#                               (int(x_max), int(y_max)), 
#                               (0, 255, 0), 2)
#                 cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Display the resulting frame
#         cv2.imshow('Object Detection (Press q to exit)', frame)

#         # Exit loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the webcam and close the window
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the object detection from the camera
# detect_objects_from_camera()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ without prun & Quantize 

# import torch
# import cv2
# import time
# import psutil
# import os
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Check and set the device (use MPS for Apple Silicon)
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# print(f"Using device: {device}")

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
# model.eval()

# # Initialize measurement variables
# total_inference_time = 0
# frame_count = 0
# process = psutil.Process(os.getpid())

# # Get class names
# class_names = model.names

# def get_memory_usage():
#     """Get current memory usage in MB"""
#     return process.memory_info().rss / (1024 ** 2)

# def get_cpu_load():
#     """Get current CPU load percentage"""
#     return psutil.cpu_percent(interval=0.1)

# def detect_objects_from_camera():
#     global total_inference_time, frame_count
    
#     cap = cv2.VideoCapture(0)
#     start_time = time.time()
    
#     try:
#         while True:
#             # Memory measurement before processing
#             mem_before = get_memory_usage()
#             cpu_before = get_cpu_load()
            
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Preprocessing
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
#             # Inference with timing
#             inference_start = time.time()
#             results = model(image_rgb)
#             inference_end = time.time()
            
#             # Measurements
#             total_inference_time += (inference_end - inference_start)
#             frame_count += 1
#             mem_after = get_memory_usage()
#             cpu_after = get_cpu_load()
            
#             # Parse results
#             detections = results.xyxy[0].cpu().numpy()
            
#             # Draw bounding boxes (same as before)
#             for detection in detections:
#                 x_min, y_min, x_max, y_max, confidence, class_id = detection
#                 if confidence > 0.5:
#                     class_name = class_names[int(class_id)]
#                     label = f"{class_name}: {confidence:.2f}"
#                     cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
#                     cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
#             # Display metrics on frame
#             cv2.putText(frame, f"Memory: {mem_after:.2f}MB", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"CPU Load: {cpu_after:.1f}%", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Frame Time: {(inference_end - inference_start)*1000:.1f}ms", (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             cv2.imshow('Object Detection', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     finally:
#         # Final calculations
#         total_time = time.time() - start_time
#         avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        
#         # Print summary
#         print("\n--- Performance Summary ---")
#         print(f"Total Frames Processed: {frame_count}")
#         print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
#         print(f"Peak Memory Usage: {get_memory_usage():.2f}MB")
#         print(f"Average CPU Load: {psutil.cpu_percent(interval=1):.1f}%")
#         print(f"Total Runtime: {total_time:.2f} seconds")
        
#         # Power estimation (macOS specific)
#         print("\nNote: For precise power measurements on macOS:")
#         print("Run 'sudo powermetrics --samplers smc' in terminal during inference")
        
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_objects_from_camera()


# ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ with pruning implementation code  

import torch
import cv2
import time
import psutil
import os
import warnings
import torch.nn.utils.prune as prune

warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("mps")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)

# ----------------- Pruning Function -----------------
def apply_pruning(model, pruning_amount=0.2):
    """Applies L1 unstructured pruning to convolutional layers"""
    parameters_to_prune = []
    
    # Select convolutional layers in backbone and head
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    # Global pruning across all selected layers
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount
    )
    
    # Remove pruning reparameterization to make it permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    
    print(f"Pruned {len(parameters_to_prune)} conv layers ({pruning_amount*100}% sparsity)")
    return model

# Apply pruning to the model (20% sparsity)
model = apply_pruning(model)
model.eval()

# Save the pruned model
pruned_model_path = "best.pt"
torch.save(model.state_dict(), pruned_model_path)
print(f"Pruned model saved as {pruned_model_path}")
# ----------------------------------------------------

# Measurement setup
process = psutil.Process(os.getpid())
metrics = {
    'total_time': 0.0,
    'frame_count': 0,
    'max_memory': 0.0,
    'cpu_loads': []
}

def get_metrics():
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    cpu = psutil.cpu_percent(interval=0.1)
    return mem, cpu

def detect_objects_from_camera():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    try:
        while True:
            mem_before, cpu_before = get_metrics()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            inference_start = time.time()
            results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inference_time = time.time() - inference_start
            
            # Update metrics
            mem_after, cpu_after = get_metrics()
            metrics['total_time'] += inference_time
            metrics['frame_count'] += 1
            metrics['max_memory'] = max(metrics['max_memory'], mem_after)
            metrics['cpu_loads'].append(cpu_after)
            
            # Display detections
            detections = results.xyxy[0].cpu().numpy()
            for detection in detections:
                x_min, y_min, x_max, y_max, conf, cls_id = detection
                if conf > 0.5:
                    label = f"{model.names[int(cls_id)]}: {conf:.2f}"
                    cv2.rectangle(frame, (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), (0,255,0), 2)
                    cv2.putText(frame, label, (int(x_min), int(y_min)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Display metrics
            cv2.putText(frame, f"Mem: {mem_after:.1f}MB", (10,30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"CPU: {cpu_after:.1f}%", (10,60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Frame: {inference_time*1000:.1f}ms", (10,90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            cv2.imshow('Pruned YOLOv5', frame)
            if cv2.waitKey(1) == ord('q'):
                break
                
    finally:
        total_duration = time.time() - start_time
        avg_cpu = sum(metrics['cpu_loads'])/len(metrics['cpu_loads']) if metrics['cpu_loads'] else 0
        
        print("\n\033[1mPruned Model Metrics:\033[0m")
        print(f"Frames Processed: {metrics['frame_count']}")
        print(f"Average FPS: {metrics['frame_count']/total_duration:.1f}")
        print(f"Peak Memory: {metrics['max_memory']:.1f}MB")
        print(f"Average CPU: {avg_cpu:.1f}%")
        print(f"Total Duration: {total_duration:.1f}s")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects_from_camera()




# ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ with Quantization implementation code



# import torch
# import cv2
# import time
# import psutil
# import os
# import warnings

# warnings.filterwarnings("ignore")

# # Device configuration (Quantization works best on CPU)
# device = torch.device("cpu")  # Forced to CPU for quantization
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # ----------------- Quantization Function -----------------
# def apply_quantization(model):
#     """Applies dynamic quantization to linear and convolutional layers"""
#     quantized_model = torch.quantization.quantize_dynamic(
#         model,  # Original model
#         {torch.nn.Conv2d, torch.nn.Linear},  # Layers to quantize
#         dtype=torch.qint8  # 8-bit quantization
#     )
#     print("Applied dynamic quantization to Conv2d and Linear layers")
#     return quantized_model

# # Apply quantization
# model = apply_quantization(model)
# model.eval()
# # ---------------------------------------------------------

# # Measurement setup
# process = psutil.Process(os.getpid())
# metrics = {
#     'total_time': 0.0,
#     'frame_count': 0,
#     'max_memory': 0.0,
#     'cpu_loads': []
# }

# def get_metrics():
#     mem = process.memory_info().rss / (1024 ** 2)  # MB
#     cpu = psutil.cpu_percent(interval=0.1)
#     return mem, cpu

# def detect_objects_from_camera():
#     cap = cv2.VideoCapture(0)
#     start_time = time.time()
    
#     try:
#         while True:
#             mem_before, cpu_before = get_metrics()
            
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Preprocessing with CPU tensor
#             input_tensor = torch.from_numpy(
#                 cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             ).permute(2,0,1).float().div(255).unsqueeze(0)
            
#             # Inference
#             inference_start = time.time()
#             with torch.no_grad():
#                 results = model(input_tensor)
#             inference_time = time.time() - inference_start
            
#             # Update metrics
#             mem_after, cpu_after = get_metrics()
#             metrics['total_time'] += inference_time
#             metrics['frame_count'] += 1
#             metrics['max_memory'] = max(metrics['max_memory'], mem_after)
#             metrics['cpu_loads'].append(cpu_after)
            
#             # Process results
#             detections = results.pred[0]
#             for *xyxy, conf, cls in detections:
#                 if conf > 0.5:
#                     label = f"{model.names[int(cls)]}: {conf:.2f}"
#                     x_min, y_min, x_max, y_max = map(int, xyxy)
#                     cv2.rectangle(frame, (x_min, y_min), 
#                                 (x_max, y_max), (0,255,0), 2)
#                     cv2.putText(frame, label, (x_min, y_min-10),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
#             # Display metrics
#             cv2.putText(frame, f"Mem: {mem_after:.1f}MB", (10,30), 
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(frame, f"CPU: {cpu_after:.1f}%", (10,60),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
#             cv2.putText(frame, f"Frame: {inference_time*1000:.1f}ms", (10,90),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
#             cv2.imshow('Quantized YOLOv5', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break
                
#     finally:
#         total_duration = time.time() - start_time
#         avg_cpu = sum(metrics['cpu_loads'])/len(metrics['cpu_loads']) if metrics['cpu_loads'] else 0
        
#         print("\n\033[1mQuantized Model Metrics:\033[0m")
#         print(f"Frames Processed: {metrics['frame_count']}")
#         print(f"Average FPS: {metrics['frame_count']/total_duration:.1f}")
#         print(f"Peak Memory: {metrics['max_memory']:.1f}MB")
#         print(f"Average CPU: {avg_cpu:.1f}%")
#         print(f"Total Duration: {total_duration:.1f}s")
        
#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_objects_from_camera()