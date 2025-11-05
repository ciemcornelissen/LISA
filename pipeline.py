"""Legacy entry-point kept for backwards compatibility.

Importing from this module forwards to the maintained helpers under
``lisa.inference`` so existing notebooks that still do ``import pipeline``
keep running. For new code prefer ``from lisa import inference`` directly.
"""

from lisa.inference import (  # noqa: F401
    bunchTracker,
    create_pseudo_rgb,
    data_finder,
    extract_patches_from_bbox,
    interpolate_gps,
    load_models_data,
    preprocess_and_predict_quality,
    preprocess_and_predict_quality_onnx,
    process_single_window,
    snv_tensor,
)

__all__ = [
    "bunchTracker",
    "create_pseudo_rgb",
    "data_finder",
    "extract_patches_from_bbox",
    "interpolate_gps",
    "load_models_data",
    "preprocess_and_predict_quality",
    "preprocess_and_predict_quality_onnx",
    "process_single_window",
    "snv_tensor",
]
from utilities import *
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from PIL import Image  # For handling pseudo-RGB images for YOLO
from scipy.spatial.distance import cdist # For calculating distances between points
import random  # <-- ADD THIS LINE

def load_models_data():
    with open('models/configs.pkl', 'rb') as f:
        loaded = pickle.load(f)
        config = loaded['config']
        model_config = loaded['model_config']
        ae_config = loaded['ae_config']
        
    weight_config_params = torch.load('models/cross_weight_predictor_epoch_99_R2_0.757_end_R20.610_runID_i0mijofs.pth', map_location=torch.device('cpu'))
    
    # --- Load Models ---
    # ** REPLACE MOCK MODELS WITH YOUR REAL ONES HERE **
    # yolo_model = YOLO(f'/project_ghent/grapePaper/models/yoloModel_wandb_nstxfge0_map95_0.6564.pt')
    yolo_model = YOLO(f'models/yoloModel_wandb_wy3k2hnb_testRecall_0.7588.pt', task='detect')
    
    
    
    
    # Load weight model
    weight_model = SpectralPredictorWithTransformer(weight_config_params['model_config'])
    weight_model.load_state_dict(weight_config_params['weight_predictor_state_dict'])
    # In reality:
    # weight_model.load_state_dict(torch.load('path/to/weight_model.pth'))
    
    # Load quality model (as per your example)
    predictor = SpectralPredictorWithTransformer(model_config)
    AE = EncoderAE_3D(ae_config)
    predictor.load_state_dict(torch.load('models/ae_predictor_epoch_150_R2_0.5951396226882935_runID_cd7605a0.pth', map_location=torch.device('cpu'))['sugar_predictor_state_dict'])
    predictor.eval()
    AE.load_state_dict(torch.load('models/ae_predictor_epoch_150_R2_0.5951396226882935_runID_cd7605a0.pth', map_location=torch.device('cpu'))['encoder_state_dict'])
    AE.eval()
    
    models = {
        'yolo': yolo_model,
        'weight': weight_model,
        'quality': predictor,
        'encoder': AE,
    }
    
    
    
    print("Loading pre-processed data objects...")
    with open('demo_data/weight_images.pkl', 'rb') as f:
        weight_images = pickle.load(f)
    print("'weight_images' loaded.")
    
    with open('demo_data/images.pkl', 'rb') as f:
        images = pickle.load(f)
    print("'images' loaded.")
    return models, weight_config_params, weight_images, images

# ==============================================================================
# 2. PIPELINE FUNCTIONS
# (These functions implement the logic)
# ==============================================================================
def snv_tensor(x: torch.Tensor) -> torch.Tensor:
        """Applies Standard Normal Variate scaling per spectrum using PyTorch."""
        if x.ndim == 1:  # Single spectrum (C,)
            mean = torch.mean(x)
            std = torch.std(x)
            std = torch.clamp(std, min=1e-6)  # Avoid division by zero
            return (x - mean) / std
        elif x.ndim >= 3:  # Image-like (..., C, H, W) or (..., C, L) - Scale along C dim
            # Assuming C is the dimension to calculate mean/std over (dim=1 if B, C, H, W)
            # For (C, H, W), dim=0
            # For (C, L), dim=0
            dim_to_reduce = 0 if x.ndim == 3 else 1  # Adjust based on common shapes like (C,H,W) or (B,C,H,W)
            if x.shape[dim_to_reduce] <= 1:  # Cannot compute std dev over dimension of size 1
                print(f"Warning: Cannot apply SNV along dimension {dim_to_reduce} with size {x.shape[dim_to_reduce]}")
                return x

            mean = torch.mean(x, dim=dim_to_reduce, keepdim=True)
            std = torch.std(x, dim=dim_to_reduce, keepdim=True)
            std = torch.clamp(std, min=1e-6)  # Avoid division by zero
            return (x - mean) / std
        else:  # e.g. (C, L)
            print(f"Warning: SNV tensor processing not fully implemented for ndim={x.ndim}")
            return x  # Return unchanged


def create_pseudo_rgb(hsi_window, bands=(114, 58, 20), oldWay = True):
    """
    Creates a pseudo-RGB image from a hyperspectral window by selecting 3 bands.
    Args:
        hsi_window (np.ndarray): The HSI data for the current window.
        bands (tuple): The indices of the bands to use for R, G, and B.
    Returns:
        PIL.Image: An image object compatible with YOLO models.
    """
    if oldWay:
        data_point = hsi_window[:, :, [114, 58, 20]]
        histogram, bin_edges = np.histogram(data_point, bins=10)
        scale = bin_edges[-1]
        data_point = data_point / scale
        data_point = np.clip(data_point, 0, 1)
        image_RGB = np.sqrt(data_point)
    else:
        data_point = hsi_window[:, :, [114, 58, 20]]
        p_low, p_high = np.percentile(data_point, (2, 98))
        data_point = np.clip((data_point - p_low) / (p_high - p_low), 0, 1)
        image_RGB = np.sqrt(data_point)
    return image_RGB


    # rgb_data = hsi_window[:, :, bands]
    #
    # # Normalize each channel to 0-255 for image creation
    # normalized_data = np.zeros_like(rgb_data, dtype=np.uint8)
    # for i in range(3):
    #     band = rgb_data[:, :, i]
    #     min_val, max_val = band.min(), band.max()
    #     if max_val > min_val:
    #         normalized_data[:, :, i] = 255 * (band - min_val) / (max_val - min_val)
    #
    # return normalized_data


def extract_patches_from_bbox(hsi_window, bbox, patch_size=(8, 8), stride=3, center_bias_factor=0.5, max_patches=50):
    """
    Extracts patches from the HSI data within a given bounding box, with a maximum limit.
    Samples more densely from the center of the bounding box and ensures not to exceed max_patches.

    Args:
        hsi_window (np.ndarray): The HSI data for the entire window.
        bbox (list): Bounding box [x1, y1, x2, y2].
        patch_size (tuple): The (height, width) of patches to extract.
        stride (int): How many pixels to move between patches.
        center_bias_factor (float): The fraction of the bbox to consider the 'center'.
        max_patches (int): The maximum number of patches to return.

    Returns:
        list[np.ndarray]: A list of extracted HSI patches.
    """
    x1, y1, x2, y2 = map(int, bbox)
    bunch_hsi_data = hsi_window[y1:y2, x1:x2, :]

    h, w, _ = bunch_hsi_data.shape
    patch_h, patch_w = patch_size

    # Define the 'center' region of the bounding box
    center_x = w / 2
    center_y = h / 2
    center_w = w * center_bias_factor
    center_h = h * center_bias_factor
    center_x1, center_y1 = center_x - center_w / 2, center_y - center_h / 2
    center_x2, center_y2 = center_x + center_w / 2, center_y + center_h / 2

    potential_patches = []
    # First, gather all possible patches based on the sampling strategy
    for y in range(0, h - patch_h + 1, stride):
        for x in range(0, w - patch_w + 1, stride):
            patch_center_x, patch_center_y = x + patch_w / 2, y + patch_h / 2

            # Check if patch center is in the 'center region'
            is_in_center = (center_x1 <= patch_center_x <= center_x2) and \
                           (center_y1 <= patch_center_y <= center_y2)

            # Keep all patches from the center, but only some from the edges
            if is_in_center or np.random.rand() < 0.33:
                patch = bunch_hsi_data[y:y + patch_h, x:x + patch_w, :]
                potential_patches.append(patch)

    # If the number of collected patches exceeds the maximum, randomly sample from them
    if len(potential_patches) > max_patches:
        print(f"Found {len(potential_patches)} patches, randomly sampling down to {max_patches}.")
        patches = random.sample(potential_patches, max_patches)
    else:
        patches = potential_patches

    print(f"Extracted {len(patches)} patches from bounding box.")
    return patches


def preprocess_and_predict_quality(patches, quality_model, models):
    """
    Preprocesses patches with Savitzky-Golay and predicts quality attributes.
    Averages the results for patches classified as 'grape'.

    Args:
        patches (list[np.ndarray]): List of HSI patches.
        quality_model (nn.Module): The trained Brix/Acid/Grape model.

    Returns:
        tuple or None: (avg_brix, avg_acid) if grapes were detected, else None.
    """
    if not patches:
        return None

    # Instantiate filters only if a SavGol technique might be selected
    savgol_gpu_deriv1 = SavGolFilterGPU(
        window_length=5, deriv=1)
    preprocessed_patches = [
        savgol_gpu_deriv1(torch.tensor(p.copy())) for p in patches
    ]

    # Stack into a batch and convert to a tensor for the model
    batch_tensor = torch.from_numpy(np.array(preprocessed_patches)).float()
    batch_tensor = batch_tensor.permute(0,3,1,2).contiguous()
    # Get predictions from the multi-head model
    with torch.no_grad():
        if models.get('encoder',False):
            batch_tensor = models['encoder'](batch_tensor)[0]
            print(batch_tensor.shape)
        brix_preds, acid_preds, is_grape_logits = quality_model(batch_tensor)

    # Filter results: only keep predictions where the model classified the patch as 'grape'
    grape_mask = is_grape_logits > 0
    grape_mask = grape_mask.squeeze()
    num_grape_patches = torch.sum(grape_mask).item()
    total_patches = grape_mask.numel()
    grape_percentage = (num_grape_patches / total_patches) * 100
    print(num_grape_patches)
    print(total_patches)
    # Check if the percentage of grapes is below our defined threshold
    if grape_percentage < 30:
        print(
            f"Quality Prediction: Bunch rejected. "
            f"Only {grape_percentage:.1f}% of patches ({int(num_grape_patches)}/{total_patches}) "
            f"were identified as grapes, which is below the {20}% threshold."
        )
        return None
    if torch.any(grape_mask):
        valid_brix = brix_preds[grape_mask]
        valid_acid = acid_preds[grape_mask]

        avg_brix = valid_brix.mean().item()
        avg_acid = valid_acid.mean().item()

        print(
            f"Quality Prediction: Found {grape_mask.sum()} grape patches. Avg Brix: {avg_brix:.2f}, Avg Acid: {avg_acid:.2f}")
        return avg_brix, avg_acid, grape_percentage
    else:
        print("Quality Prediction: No grape patches were identified by the model.")
        return None


def preprocess_and_predict_quality_onnx(patches, quality_session, models):  # model is now an onnx session
    """
    ## MODIFIED for ONNX ##
    Preprocesses patches and predicts quality attributes using a high-performance ONNX session.

    Args:
        patches (list[np.ndarray]): List of HSI patches.
        quality_session (ort.InferenceSession): The ONNX runtime session for the quality model.

    Returns:
        tuple or None: (avg_brix, avg_acid, grape_percentage) if grapes were detected, else None.
    """
    if not patches:
        return None

    # Preprocessing remains the same (SavGol filter)
    # Note: If SavGolFilterGPU requires torch, this part stays.
    savgol_gpu_deriv1 = SavGolFilterGPU(window_length=5, deriv=1)
    # The list comprehension can be slow; batching this step can also help if it's a bottleneck.
    preprocessed_patches_torch = [savgol_gpu_deriv1(torch.tensor(p.copy())) for p in patches]

    # Stack into a batch and permute. This is still easier with PyTorch.
    batch_tensor = torch.stack(preprocessed_patches_torch).permute(0, 3, 1, 2).contiguous()

    # --- CORE ONNX CHANGE ---
    # 1. Convert the final PyTorch tensor to a NumPy array for ONNX Runtime
    batch_numpy = batch_tensor.numpy()

    # 2. Prepare the input dictionary for the ONNX session
    input_name = quality_session.get_inputs()[0].name
    ort_inputs = {input_name: batch_numpy}

    # 3. Run inference. The session outputs a list of NumPy arrays.
    ort_outs = quality_session.run(None, ort_inputs)
    brix_preds, acid_preds, is_grape_logits = ort_outs[0], ort_outs[1], ort_outs[2]
    # --- END OF ONNX CHANGE ---

    # The rest of the logic can use NumPy or be converted back to Tensors
    grape_mask = is_grape_logits > 0
    grape_mask = grape_mask.squeeze()

    num_grape_patches = np.sum(grape_mask)
    total_patches = grape_mask.size
    grape_percentage = (num_grape_patches / total_patches) * 100 if total_patches > 0 else 0

    # if grape_percentage < 30:
    #     print(f"Quality Prediction: Bunch rejected. Only {grape_percentage:.1f}% of patches were grapes.")
    #     return None

    if np.any(grape_mask):
        valid_brix = brix_preds[grape_mask]
        valid_acid = acid_preds[grape_mask]

        avg_brix = valid_brix.mean()
        avg_acid = valid_acid.mean()

        print(
            f"Quality Prediction: Found {int(num_grape_patches)} grape patches. Avg Brix: {avg_brix:.2f}, Avg Acid: {avg_acid:.2f}")
        return avg_brix, avg_acid, grape_percentage
    else:
        print("Quality Prediction: No grape patches were identified by the model.")
        return None

def process_single_window(hsi_window, hsi_window_RGB, models, tracker, gps_coords, window_index, dataset, start_x, weight_config, weight_dataset, rgbTechnique = True, confidence = 0.1):
    """
    ## MODIFIED ##: Main processing function now includes the tracker.
    """
    window_results = []
    window_width = hsi_window.shape[1]

    # 1. Convert HSI to Pseudo-RGB
    pseudo_rgb_img = create_pseudo_rgb(hsi_window_RGB, oldWay=rgbTechnique)
    # # pseudo_rgb_img = cv2.resize(pseudo_rgb_img, (640,640))
    # pseudo_rgb_img = pseudo_rgb_img.transpose(2, 0, 1)
    # pseudo_rgb_img = np.expand_dims(pseudo_rgb_img, axis=0)
    if pseudo_rgb_img.dtype != np.uint8:
        if pseudo_rgb_img.max() <= 1.0 and pseudo_rgb_img.min() >= 0.0:  # Assuming normalized float
            pseudo_rgb_img = (pseudo_rgb_img * 255).astype(np.uint8)
        else:  # Or some other scaling if needed
            pseudo_rgb_img = pseudo_rgb_img.astype(np.uint8)  # Or handle appropriate scaling

    print('image is created and processed for saving', flush=True)

    # --- Using Pillow ---
    pseudo_rgb_img = Image.fromarray(pseudo_rgb_img, 'RGB')


    # 2. Run YOLO to get raw bounding boxes
    # results = models['yolo'](torch.tensor(pseudo_rgb_img),conf=confidence)
    results = models['yolo'](pseudo_rgb_img,conf=confidence)

    # return results
    # ## NEW ##: 3. Filter out bunches touching the window's side edges
    bboxes_tensor = results[0].boxes.xyxy # This is a Tensor on the GPU
    bboxes_for_tracker = bboxes_tensor.cpu().numpy()
    edge_margin = 5  # pixels
    full_bboxes = [
        bbox for bbox in bboxes_for_tracker
        # if bbox[0] > edge_margin and bbox[2] < (window_width - edge_margin)
    ]
    print(f"Detected {len(results[0].boxes.cls)} raw bunches, {len(full_bboxes)} are fully in frame.")

    # ## NEW ##: 4. Update the tracker with the fully-visible bounding boxes
    tracked_objects = tracker.update(full_bboxes)

    # 5. Process each tracked bunch
    for bunch_id, bbox in tracked_objects.items():
        # ## NEW ##: Check if this bunch has already been processed.
        if bunch_id in tracker.processed_ids:
            print(f"Bunch ID {bunch_id} is already processed. Skipping.")
            continue

        # ## NEW ##: Only process if the bunch is near the center of the frame.
        # This is the "prime" opportunity to analyze it.
        bbox_center_x = (bbox[0] + bbox[2]) / 2
        center_zone_start = window_width * 0.05
        center_zone_end = window_width * 0.95
        x1_local, y1_local, x2_local, y2_local = bbox
        global_bbox = [
            x1_local + start_x,
            y1_local,
            x2_local + start_x,
            y2_local
        ]
        #TODO
        # if not (center_zone_start < bbox_center_x < center_zone_end):
        #     print(f"Bunch ID {bunch_id} is not centered. Will re-evaluate in next frame.")
        #     continue

        print(f"--- Processing NEW centered Bunch ID: {bunch_id} ---")

        # 5a. Predict Weight
        x1, y1, x2, y2 = map(int, bbox)
        bunch_hsi_data = hsi_window_RGB[y1:y2, x1:x2, :]
        if models['weight'] is not None:
            with torch.no_grad():
                # Instantiate filters only if a SavGol technique might be selected
                if weight_config['dataLoader.normalize']:
                    bunch_hsi_data = (bunch_hsi_data - weight_dataset.scaler_images[0][np.newaxis, np.newaxis,:]) / weight_dataset.scaler_images[1][np.newaxis, np.newaxis,:]
                weight_input = cv2.resize(np.array(bunch_hsi_data), dsize=(weight_config['model.patch_size'],
                                                                                      weight_config[
                                                                                          'model.patch_size']),
                                          interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
                # savgol_gpu_deriv = SavGolFilterGPU(
                #     window_length=weight_config['dataLoader.Savitzky_Golay_window'], deriv=1)
                if weight_config['dataLoader.processTechnique'] in [4,5]:
                    bunch_hsi_data = snv_tensor(torch.tensor(weight_input.copy()))
                savgol_gpu_deriv = SavGolFilterGPU(
                    window_length=weight_config['dataLoader.Savitzky_Golay_window'], deriv=2 if weight_config['dataLoader.processTechnique'] in [3,5] else 1
                )
                preprocessed_bunch_weight = savgol_gpu_deriv(bunch_hsi_data)
                predicted_weight = models['weight'].forward(preprocessed_bunch_weight.unsqueeze(0).float()).item()
        else:
            predicted_weight = 0
        # 5b. Extract patches
        patches = extract_patches_from_bbox(hsi_window, bbox)

        # 5c. Predict quality
        quality_results = preprocess_and_predict_quality(patches, models['quality'], models)

        # 6. Aggregate and store results
        if quality_results:
            avg_brix, avg_acid, grape_percentage = quality_results
            if dataset.normalize:
                avg_brix = dataset.scaler_labels_brix.inverse_transform(np.array([[avg_brix]]))[0,0]
                avg_acid = dataset.scaler_labels_acid.inverse_transform(np.array([[avg_acid]]))[0,0]
                if weight_config['dataLoader.normalize']:
                    predicted_weight = weight_dataset.scaler_labels_weights.inverse_transform(np.array([[predicted_weight]]))[0,0]
            bunch_result = {
                "bunch_id": bunch_id,
                "coordinates": gps_coords,
                "predicted_weight_g": predicted_weight,
                "predicted_brix": avg_brix,
                "predicted_acidity": avg_acid,
                "global_bounding_box": global_bbox,  # ## NEW ##: Add this to the final result
                "grape_percentage": grape_percentage,
            }
            window_results.append(bunch_result)

            # ## NEW ##: Mark this ID as processed so we don't do it again.
            tracker.processed_ids.add(bunch_id)

    return window_results

# from ultralytics.utils.ops import non_max_suppression
#
# def process_single_window_onnx(hsi_window, hsi_window_RGB, models, tracker, gps_coords, window_index, dataset, start_x,
#                           weight_config, weight_dataset):
#     """
#     ## CORRECTED FOR ONNX INFERENCE ##
#     """
#     # =========================================================================
#     #  STEP 2: THE DEBUGGING PRINT STATEMENT
#     # =========================================================================
#     print("--- CONFIRMATION: RUNNING THE NEW, ONNX-OPTIMIZED process_single_window ---")
#
#     window_results = []
#     window_width = hsi_window.shape[1]
#
#     # 1. Convert HSI to Pseudo-RGB
#     pseudo_rgb_img_np = create_pseudo_rgb(hsi_window_RGB)
#
#     # 2. Prepare image for YOLO ONNX model
#     input_img = pseudo_rgb_img_np.transpose(2, 0, 1)
#     input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
#
#     # 3. Run YOLO inference using the ONNX session's .run() method
#     yolo_session = models['yolo']
#     input_name = yolo_session.get_inputs()[0].name
#     ort_inputs = {input_name: input_img}
#     ort_outs = yolo_session.run(None, ort_inputs)
#
#     # 4. Post-process the raw numeric output to get clean bounding boxes
#     # The `conf_thres=0.4` now lives inside this function call.
#     output_tensor = torch.from_numpy(ort_outs[0])
#     results_nms = non_max_suppression(output_tensor, conf_thres=0.4, iou_thres=0.5)[0]
#
#     if results_nms is None:
#         bboxes_for_tracker = []
#     else:
#         bboxes_for_tracker = results_nms[:, :4].cpu().numpy()
#
#     # THE REST OF THE FUNCTION CONTINUES UNCHANGED...
#     edge_margin = 5
#     full_bboxes = [
#         bbox for bbox in bboxes_for_tracker
#         if bbox[0] > edge_margin and bbox[2] < (window_width - edge_margin)
#     ]
#     print(f"Detected {len(bboxes_for_tracker)} raw bunches, {len(full_bboxes)} are fully in frame.")
#
#     tracked_objects = tracker.update(full_bboxes)
#
#     # Looping logic remains the same
#     for bunch_id, bbox in tracked_objects.items():
#         if bunch_id in tracker.processed_ids:
#             continue
#
#         bbox_center_x = (bbox[0] + bbox[2]) / 2
#         center_zone_start = window_width * 0.05
#         center_zone_end = window_width * 0.95
#         if not (center_zone_start < bbox_center_x < center_zone_end):
#             continue
#
#         print(f"--- Processing NEW centered Bunch ID: {bunch_id} ---")
#
#         # ## NEW ##: Only process if the bunch is near the center of the frame.
#         # This is the "prime" opportunity to analyze it.
#         bbox_center_x = (bbox[0] + bbox[2]) / 2
#         center_zone_start = window_width * 0.05
#         center_zone_end = window_width * 0.95
#         x1_local, y1_local, x2_local, y2_local = bbox
#         global_bbox = [
#             x1_local + start_x,
#             y1_local,
#             x2_local + start_x,
#             y2_local
#         ]
#
#         if not (center_zone_start < bbox_center_x < center_zone_end):
#             print(f"Bunch ID {bunch_id} is not centered. Will re-evaluate in next frame.")
#             continue
#
#         print(f"--- Processing NEW centered Bunch ID: {bunch_id} ---")
#
#         # 5a. Predict Weight
#         x1, y1, x2, y2 = map(int, bbox)
#         bunch_hsi_data = hsi_window_RGB[y1:y2, x1:x2, :]
#         if models['weight'] is not None:
#             with torch.no_grad():
#                 # Instantiate filters only if a SavGol technique might be selected
#                 if weight_config['dataLoader.normalize']:
#                     bunch_hsi_data = (bunch_hsi_data - weight_dataset.scaler_images[0][np.newaxis, np.newaxis, :]) / \
#                                      weight_dataset.scaler_images[1][np.newaxis, np.newaxis, :]
#                 weight_input = cv2.resize(np.array(bunch_hsi_data), dsize=(weight_config['model.patch_size'],
#                                                                            weight_config[
#                                                                                'model.patch_size']),
#                                           interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
#                 # savgol_gpu_deriv = SavGolFilterGPU(
#                 #     window_length=weight_config['dataLoader.Savitzky_Golay_window'], deriv=1)
#                 if weight_config['dataLoader.processTechnique'] in [4, 5]:
#                     bunch_hsi_data = snv_tensor(torch.tensor(weight_input.copy()))
#                 savgol_gpu_deriv = SavGolFilterGPU(
#                     window_length=weight_config['dataLoader.Savitzky_Golay_window'],
#                     deriv=2 if weight_config['dataLoader.processTechnique'] in [3, 5] else 1
#                 )
#                 preprocessed_bunch_weight = savgol_gpu_deriv(bunch_hsi_data)
#                 predicted_weight = models['weight'].forward(preprocessed_bunch_weight.unsqueeze(0).float()).item()
#         else:
#             predicted_weight = 0
#         # 5b. Extract patches
#         patches = extract_patches_from_bbox(hsi_window, bbox)
#
#         # 5c. Predict quality
#         quality_results = preprocess_and_predict_quality_onnx(patches, models['quality'], models)
#
#         # 6. Aggregate and store results
#         if quality_results:
#             avg_brix, avg_acid, grape_percentage = quality_results
#             if dataset.normalize:
#                 avg_brix = dataset.scaler_labels_brix.inverse_transform(np.array([[avg_brix]]))[0, 0]
#                 avg_acid = dataset.scaler_labels_acid.inverse_transform(np.array([[avg_acid]]))[0, 0]
#                 if weight_config['dataLoader.normalize']:
#                     predicted_weight = \
#                     weight_dataset.scaler_labels_weights.inverse_transform(np.array([[predicted_weight]]))[0, 0]
#             bunch_result = {
#                 "bunch_id": bunch_id,
#                 "coordinates": gps_coords,
#                 "predicted_weight_g": predicted_weight,
#                 "predicted_brix": avg_brix,
#                 "predicted_acidity": avg_acid,
#                 "global_bounding_box": global_bbox,  # ## NEW ##: Add this to the final result
#                 "grape_percentage": grape_percentage,
#             }
#             window_results.append(bunch_result)
#
#             # ## NEW ##: Mark this ID as processed so we don't do it again.
#             tracker.processed_ids.add(bunch_id)
#
#     return window_results
#

# ==============================================================================
# ## NEW ##: 2. OBJECT TRACKER CLASS
# ==============================================================================
class bunchTracker:
    def __init__(self, max_disappeared=3, match_threshold=50, slide_step= 180):
        """
        A simple tracker to assign unique IDs to grape bunches across frames.

        Args:
            max_disappeared (int): Number of consecutive frames a bunch can be
                                 'lost' before it's deregistered.
            match_threshold (int): Max pixel distance between centroids to
                                   consider it the same object.
        """
        self.next_object_id = 0
        self.objects = {}  # object_id -> centroid in the *last seen frame's coordinates*
        self.bboxes = {}  # object_id -> bbox in the *last seen frame's coordinates*
        self.disappeared = {}
        self.processed_ids = set()
        self.max_disappeared = max_disappeared
        self.match_threshold = match_threshold
        self.slide_step = slide_step
    def _get_centroids(self, bboxes):
        """Calculate the center point of each bounding box."""
        return np.array([
            [(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in bboxes
        ])

    def update(self, bboxes, slide_step=0):
        """
        ## MODIFIED ##: Update now takes 'slide_step' as an argument.
        This is the distance the camera moved since the last frame.
        """
        if len(bboxes) == 0:
            # No detections, mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return {}

        input_centroids = self._get_centroids(bboxes)

        if len(self.objects) == 0:
            # First frame or no objects being tracked, register all new detections
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], bboxes[i])
            return self.bboxes

        # Get existing object IDs and their last known centroids
        object_ids = list(self.objects.keys())
        last_centroids = np.array(list(self.objects.values()))

        # ## CORE LOGIC CHANGE ##
        # Predict the new position of old objects by applying the motion model.
        # motion = [slide_step_x, slide_step_y]. We assume y-motion is 0.
        predicted_centroids = last_centroids - np.array([self.slide_step, 0])

        # Now, calculate the distance between the NEW input centroids and the
        # PREDICTED centroids of the existing objects.
        D = cdist(predicted_centroids, input_centroids)

        # (The rest of the matching logic is the same as before, but it now
        # operates on the motion-compensated distance matrix D)

        rows = D.min(axis=1).argsort()
        # Find the smallest distance for each new detection (col) and sort based on rows
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        current_matches = {}

        # Match objects based on minimum distance
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.match_threshold:
                continue

            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.bboxes[object_id] = bboxes[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle objects that were not matched (either disappeared or unmatched)
        unmatched_rows = set(range(len(predicted_centroids))) - used_rows
        for row in unmatched_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        # Register any new detections that were not matched
        unmatched_cols = set(range(len(input_centroids))) - used_cols
        for col in unmatched_cols:
            self._register(input_centroids[col], bboxes[col])

        # Return only the objects that are currently visible
        visible_objects = {oid: bbox for oid, bbox in self.bboxes.items() if self.disappeared.get(oid, 0) == 0}
        return visible_objects

    def _register(self, centroid, bbox):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        # We don't remove from processed_ids, because it's a permanent record
        # if object_id in self.processed_ids:
        #    self.processed_ids.remove(object_id)


# ==============================================================================
# 3. MAIN EXECUTION SCRIPT
# ==============================================================================

def main():
    """Main function to run the entire simulated pipeline."""
    print("=== Starting Grape Quality Mapping Pipeline (Simulation) ===")
    config, model_config, ae_config, optimalConfig = AE_modelTrainer('sweep_grapePaper_baseline_AE_MLP', 'cd7605a0',
                                                                     get_configs=True)
    # --- Configuration ---
    # These would be your actual system parameters
    HSI_HEIGHT = 400
    HSI_WIDTH = 2000  # A long strip simulating a row scan
    NUM_BANDS = 224
    WINDOW_WIDTH = 640  # The width of the camera's view at one time
    SLIDE_STEP = 320  # How far the robot moves before taking the next shot

    # --- Load Models ---
    # ** REPLACE MOCK MODELS WITH YOUR REAL ONES HERE **
    yolo_model = YOLO(f'/project_ghent/grapePaper/models/yoloModel_wandb_nstxfge0_map95_0.6564')

    # Load weight model
    weight_model = None
    # In reality:
    # weight_model.load_state_dict(torch.load('path/to/weight_model.pth'))
    weight_model.eval()

    # Load quality model (as per your example)
    quality_model = SpectralPredictorWithTransformer(model_config)
    AE = EncoderAE_3D(ae_config)
    quality_model.load_state_dict(torch.load('models/ae_predictor_epoch_25_R2_0.3163496255874634_runID_3nh4xvru.pth')[
                                  'sugar_predictor_state_dict'])
    quality_model.eval()
    AE.load_state_dict(
        torch.load('models/ae_predictor_epoch_25_R2_0.3163496255874634_runID_3nh4xvru.pth')['encoder_state_dict'])
    AE.eval()

    models = {
        'yolo': yolo_model,
        'weight': weight_model,
        'quality': quality_model,
        'encoder': AE,
    }

    # ## NEW ##: Initialize the tracker
    bunch_tracker = bunchTracker(match_threshold=75, slide_step=SLIDE_STEP)

    # --- Simulate Data Acquisition ---
    print("\n--- Simulating data acquisition and processing ---")
    fieldData = grape_dataLoader(camera='10', front_or_back='Front', normalize=True, indivdualGrapes=True,
                                 white_ref=False,
                                 black_ref=False, patches=True, patchSize=10, labo=False, preLoadData=False,
                                 longSweep=True)
    hsi_data_cube = fieldData.whiteAndBlackRef(3).transpose(1, 0, 2)[::-1]
    print("\n--- Data loaded in, starting pipeline ---")
    print(f"HSI data cube of shape ({HSI_HEIGHT}, {HSI_WIDTH}, {NUM_BANDS}).")

    all_results = []
    # Simulate the robot moving along the vineyard row
    for i, start_x in enumerate(range(0, HSI_WIDTH - WINDOW_WIDTH + 1, SLIDE_STEP)):
        end_x = start_x + WINDOW_WIDTH
        print(f"\n--- Processing Window {i + 1}: pixels {start_x}-{end_x} ---")

        # Get the current window from the full HSI scan
        current_window = hsi_data_cube[:, start_x:end_x, :]

        # Simulate getting georeferenced coordinates for this window
        # (e.g., from GPS and wheel odometry)
        current_gps = (40.7128 + i * 0.0001, -74.0060 + i * 0.0001)

        # Process this window and get results for any bunches found
        window_results = process_single_window(current_window, models, bunch_tracker, current_gps, i)

        # Add the results to our main database/list
        if window_results:
            all_results.extend(window_results)

    # --- Final Output ---
    print("\n\n=== PIPELINE EXECUTION COMPLETE ===")
    print(f"A total of {len(all_results)} grape bunches were detected and analyzed.")
    print("Final aggregated data:")
    for res in all_results:
        print(
            f"  - Coords: ({res['coordinates'][0]:.4f}, {res['coordinates'][1]:.4f}), "
            f"Weight: {res['predicted_weight_g']:.1f}g, "
            f"Brix: {res['predicted_brix']:.2f}, "
            f"Acidity: {res['predicted_acidity']:.2f}"
        )
    print("\nThis data can now be used to generate a spatially-resolved map.")


def interpolate_gps(start_coords, end_coords, i, total_steps):
    """
    Calculates an intermediate GPS coordinate by linearly interpolating
    between a start and end point.

    Args:
        start_coords (tuple): The starting (latitude, longitude).
        end_coords (tuple): The ending (latitude, longitude).
        i (int): The current step index (0-based).
        total_steps (int): The total number of steps in the sequence.

    Returns:
        tuple: The interpolated (latitude, longitude) for the current step.
    """
    # Handle edge case of a single point to avoid division by zero
    if total_steps <= 1:
        return start_coords

    # Calculate the progress fraction (from 0.0 to 1.0)
    progress_fraction = i / (total_steps - 1)

    # Calculate the total change for latitude and longitude
    total_lat_change = end_coords[0] - start_coords[0]
    total_lon_change = end_coords[1] - start_coords[1]

    # Calculate the current coordinates
    current_lat = start_coords[0] + (progress_fraction * total_lat_change)
    current_lon = start_coords[1] + (progress_fraction * total_lon_change)

    return (current_lat, current_lon)


from pathlib import Path

def get_most_recent_folder(parent_folder: str) -> str | None:
    """
    Finds the most recently modified subfolder in a given folder.

    Args:
        parent_folder: The path to the folder to search within.

    Returns:
        The full, absolute path of the most recent subfolder as a string,
        or None if the path is invalid or no subfolders are found.
    """
    # 1. Create a Path object and check if it's a valid directory
    parent_path = Path(parent_folder)
    if not parent_path.is_dir():
        print(f"Error: Provided path '{parent_folder}' is not a valid directory.")
        return None

    # 2. Get all subdirectories, filtering out files
    try:
        subfolders = [f for f in parent_path.iterdir() if f.is_dir()]
    except PermissionError:
        print(f"Error: Permission denied to access '{parent_folder}'.")
        return None

    # 3. If no subfolders exist, return None
    if not subfolders:
        return None

    # 4. Find the folder with the latest modification time
    latest_folder = max(subfolders, key=lambda f: f.stat().st_mtime)

    # 5. Return its full, resolved path as a string
    return str(latest_folder.resolve())


from typing import List

def get_folders_sorted_by_date(parent_folder: str) -> List[str]:
    """
    Finds all subfolders in a given folder and returns them sorted by
    modification time, from oldest to newest.

    Args:
        parent_folder: The path to the folder to search within.

    Returns:
        A list of full, absolute folder paths as strings, sorted from
        oldest to newest. Returns an empty list if the path is invalid
        or no subfolders are found.
    """
    # 1. Create a Path object and check if it's a valid directory
    parent_path = Path(parent_folder)
    if not parent_path.is_dir():
        print(f"Error: Provided path '{parent_folder}' is not a valid directory.")
        return []

    # 2. Get all subdirectories, filtering out files
    try:
        subfolders = [f for f in parent_path.iterdir() if f.is_dir()]
    except PermissionError:
        print(f"Error: Permission denied to access '{parent_folder}'.")
        return []

    # 3. Sort the folders by their modification time (st_mtime)
    # The 'key' tells sorted() to use the modification time for comparison.
    # The default sort order is ascending (oldest to newest).
    sorted_folders = sorted(subfolders, key=lambda f: f.stat().st_mtime)

    # 4. Return the list of resolved, absolute paths as strings
    return [str(f.resolve()) for f in sorted_folders]

def data_finder(dir):
    data_point_dict = {}
    capture_path = os.path.join(dir, 'capture')
    for file in os.listdir(capture_path):
        file_type = file.split('_')[0]
        if file.endswith('.hdr') and  file_type != 'DARKREF':
            data_point_dict[f'hdr'] = os.path.join(capture_path, file)
        if file.endswith('.raw') and  file_type != 'DARKREF':
            data_point_dict[f'img'] = os.path.join(capture_path, file)
    return data_point_dict



if __name__ == '__main__':
    main()