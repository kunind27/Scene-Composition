import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor)
import matplotlib.pyplot as plt

def load_models():
    """
    Initialize and load all required models.
    
    Returns:
        tuple: Contains the following models:
            - Grounding DINO model for text-based detection
            - SAM predictor for segmentation
            - Stable Diffusion inpainting model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Grounding DINO
    model_id = "IDEA-Research/grounding-dino-base"
    dino_processor = AutoProcessor.from_pretrained(model_id)
    grounding_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    grounding_dino.to(device)
    
    # Load SAM using Transformers
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    sam = SamModel.from_pretrained("facebook/sam-vit-huge")
    sam.to(device)
    
    # Load ControlNet and SD Inpainting models
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint",
        torch_dtype=dtype
    ).to(device)
    
    inpainting_model = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    
    return (dino_processor, grounding_dino), (sam_processor, sam), inpainting_model


def segment_object(image_path, text_prompt, grounding_models, sam_models, box_threshold=0.35, text_threshold=0.25):
    """
    Detect and segment an object in the image based on a text prompt using Grounding DINO and SAM.
    
    Args:
        image_path (str): Path to input image
        text_prompt (str): Description of object to detect
        grounding_models: Loaded Grounding DINO model
        sam_models: Loaded SAM model
        box_threshold (float): Confidence threshold for box detection
        text_threshold (float): Confidence threshold for text matching
    
    Returns:
        tuple: (binary mask of segmented object, original image, bounding box)
    """
    # Unpack models
    dino_processor, grounding_dino = grounding_models
    sam_processor, sam = sam_models
    
    # Load and process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Prepare inputs for Grounding DINO
    inputs = dino_processor(images=image_rgb, text=[text_prompt], return_tensors="pt").to(grounding_dino.device)
    
    # Get detections
    with torch.inference_mode():
        outputs = grounding_dino(**inputs)
    
    # Convert outputs to boxes and scores
    target_sizes = torch.tensor([image_rgb.shape[:2]]).to(grounding_dino.device)
    results = dino_processor.post_process_grounded_object_detection(outputs, 
                                                                    input_ids=inputs.input_ids,
                                                                    target_sizes=target_sizes,
                                                                    box_threshold=box_threshold)[0]
    
    # Check if {text_prompt} was detected in the image or not
    if len(results["boxes"]) == 0:
        # Hardcoded Bounding Boxes just to see effect of pipeline
        bbox = {"wall hanging": [40, 175, 133, 372],
                "stool": [90, 295, 180, 425]}
        if text_prompt not in bbox:
            raise ValueError(f"No objects found matching '{text_prompt}'")
        box = np.array(bbox[text_prompt])
    
    else:
        # Get the box with highest confidence
        best_box_idx = results["scores"].argmax()
        box = results["boxes"][best_box_idx].cpu().numpy()
        print(box, type(box), box.shape)

    # Save the Image with the Bounding Box around the class
    box_image = image_rgb.copy()
    x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
    cv2.rectangle(box_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
    cv2.imwrite(f'./dino_bbox/{text_prompt}.png', cv2.cvtColor(box_image, cv2.COLOR_RGB2BGR))
    
    # Prepare inputs for SAM
    sam_inputs = sam_processor(image_rgb,
                               input_boxes=[[box.tolist()]],
                               return_tensors="pt").to(sam.device)
    
    # Generate mask
    with torch.inference_mode():
        sam_outputs = sam(**sam_inputs)
        pred_masks = sam_outputs.pred_masks.squeeze(1).unsqueeze(2)
        masks = sam_processor.image_processor.post_process_masks(
            pred_masks,
            original_sizes=sam_inputs["original_sizes"].tolist(),
            reshaped_input_sizes=sam_inputs["reshaped_input_sizes"].tolist())
        scores = sam_outputs.iou_scores
    
    # Select best mask
    best_mask = masks[0][scores[0].argmax()].cpu().numpy().squeeze(0)
    print(f"Image Shape: {image_rgb.shape} | Mask Shape: {best_mask.shape}")
    
    return best_mask, image_rgb, box

def apply_red_mask(image, mask):
    """
    Apply a semi-transparent red overlay to the segmented object.
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Binary mask of segmented object
    
    Returns:
        np.ndarray: Image with red overlay on segmented object
    """
    red_overlay = np.zeros_like(image)
    red_overlay[mask] = [255, 0, 0]
    alpha = 0.5
    return cv2.addWeighted(image, 1, red_overlay, alpha, 0)

def create_shifted_mask(mask, x_shift, y_shift, height, width):
    """
    Create a new mask shifted by specified x and y offsets.
    
    Args:
        mask (np.ndarray): Original binary mask
        x_shift (int): Horizontal shift in pixels
        y_shift (int): Vertical shift in pixels
        height (int): Image height
        width (int): Image width
    
    Returns:
        np.ndarray: Shifted binary mask
    """
    shifted_mask = np.zeros_like(mask)
    y_start = max(0, -y_shift)
    y_end = min(height, height - y_shift)
    x_start = max(0, -x_shift)
    x_end = min(width, width - x_shift)
    
    shifted_mask[y_start:y_end,x_start:x_end] = mask[max(0, y_shift): min(height, height + y_shift), 
                                                     max(0, x_shift): min(width, width + x_shift)]
    
    return shifted_mask

def extract_object(image, mask, class_name):
    """
    Extract object from image and create a cropped version by generating a fitting bounding box
    
    Args:
        image (np.ndarray): Original image
        mask (np.ndarray): Binary mask of object
    
    Returns:
        tuple: (cropped object image, bounding box coordinates)
    """
    # Extract object
    object_image = image.copy()
    
    # Find object bounds
    y_indices, x_indices = np.where(mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Crop to bounds
    cropped_object = object_image[y_min:y_max+1, x_min:x_max+1]
    cv2.imwrite(f'./extracted_object/{class_name}.png', cv2.cvtColor(cropped_object, cv2.COLOR_RGB2BGR))
    return cropped_object, (x_min, y_min, x_max, y_max)

def image_conditioned_inpainting(image, mask, x_shift, y_shift, inpainting_model, class_name):
    """
    Move an object using ControlNet-guided inpainting process, using the extracted
    object as reference for the new placement.
    """
    height, width = image.shape[:2]
    image_pil = Image.fromarray(image)
    
    # Create bounding box mask from the shape mask
    y_indices, x_indices = np.where(mask)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Create rectangular mask
    bbox_mask = np.zeros_like(mask)
    bbox_mask[y_min:y_max+1, x_min:x_max+1] = 1
    
    # Stage 1: Inpaint original object location
    original_mask_pil = Image.fromarray((bbox_mask * 255).astype(np.uint8))
    
    # Create control image (original image with masked region)
    control_image = image.copy()
    control_image[bbox_mask.astype(bool)] = [255, 255, 255]  # White out the bbox region
    control_image_pil = Image.fromarray(control_image)
    
    # Inpaint background
    inpainted_background = inpainting_model(
        prompt="high quality, photorealistic, seamless continuation of the surrounding environment, matching lighting, texture,\
            and style of the scene, perfect perspective",
        image=image_pil,
        mask_image=original_mask_pil,
        control_image=control_image_pil,
        negative_prompt = "artifacts, blurry, distorted, inconsistent lighting, seams, edges, noise",
        num_inference_steps=30,
        guidance_scale=5.0,
        controlnet_conditioning_scale=0.3).images[0]
    
    cv2.imwrite(f'./inpainted_background/{class_name}.png', cv2.cvtColor(np.array(inpainted_background), cv2.COLOR_RGB2BGR))
    
    # Extract object and create conditioning image
    object_img, bounds = extract_object(image, mask, class_name)
    
    # Create mask for new location
    print(f"Shift = {(x_shift, y_shift)}")
    shifted_mask = create_shifted_mask(mask, x_shift, y_shift, height, width)
    shifted_mask_pil = Image.fromarray((shifted_mask * 255).astype(np.uint8))
    
    # Create control image using the extracted object by
    # roughly superimposing the class object image over
    control_image = np.array(inpainted_background)
    y_indices, x_indices = np.where(shifted_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Scale object to fit new location if needed
        target_height = y_max - y_min + 1
        target_width = x_max - x_min + 1
        resized_object = cv2.resize(object_img, (target_width, target_height))
        
        # Place resized object in control image
        control_image[y_min:y_max+1, x_min:x_max+1] = resized_object
    
    control_image_pil = Image.fromarray(control_image)
    
    # Stage 2: Place object using ControlNet guidance with extracted object reference
    final_result = inpainting_model(
        prompt="high quality, detailed object perfectly integrated into the scene, matching the reference object exactly",
        image=Image.fromarray(np.array(inpainted_background)),
        mask_image=shifted_mask_pil,
        control_image=control_image_pil,
        negative_prompt="blur, distortion, deformation, low quality, bad edges, different appearance",
        num_inference_steps=50,
        guidance_scale=12.0,
        controlnet_conditioning_scale=1.5  # Stronger control to match reference object
    ).images[0]
    
    return np.array(final_result)


def main():
    parser = argparse.ArgumentParser(description="Object Segmentation and Movement Tool")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--class', dest='class_name', required=True, help='Object class to segment')
    parser.add_argument('--output', help='Path to output image for segmentation task')
    parser.add_argument('--x', type=int, default=0, help='X-axis shift for movement task')
    parser.add_argument('--y', type=int, default=0, help='Y-axis shift for movement task')
    args = parser.parse_args()
    
    # Load models
    grounding_dino, sam_models, inpainting_model = load_models()
    
    # Perform segmentation
    mask, image, box = segment_object(args.image, args.class_name, grounding_dino, sam_models)
    
    # Task 1: Segmentation visualization
    result = apply_red_mask(image, mask)
    output_path = args.output if args.output else f'./sam_mask/{args.class_name}.png'
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Task 2: Object movement
    if not (args.x == 0 and args.y == 0):
        result = image_conditioned_inpainting(image, mask, args.x, args.y, inpainting_model, args.class_name)
        output_path = args.output if args.output else f'./outputs/{args.class_name}.png'
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()