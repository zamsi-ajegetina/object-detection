"""
Prompt templates for Sprint 3 VLM-based Failure Mining.

Each function returns a carefully engineered prompt string
to extract specific diagnostic information from the VLM.
"""


def failure_diagnosis_prompt(class_names, num_detections, confidence_stats):
    """
    Ask the VLM to explain WHY the detector failed on a frame.

    Args:
        class_names: List of class names the detector is trained on.
        num_detections: How many detections the model produced.
        confidence_stats: Dict with 'max_conf', 'mean_conf', etc.
    """
    classes_str = ', '.join(class_names)
    return f"""You are evaluating an object detection model that was trained to detect these classes on roads: [{classes_str}].

On this image, the detector produced {num_detections} detections with a maximum confidence of {confidence_stats.get('max_conf', 0):.2f}.

Analyze this image and answer in JSON format:
{{
    "objects_visible": ["list of objects you can clearly see in this road scene"],
    "objects_missed": ["objects the detector likely missed"],
    "failure_reasons": ["specific visual conditions causing failures, e.g. 'heavy dust haze obscuring vehicles', 'deep shadow hiding pedestrian', 'unusual vehicle type not in training data'"],
    "difficulty_score": <1-10 rating of scene difficulty>,
    "dominant_failure_mode": "<one of: dust_haze, shadow, occlusion, unusual_object, low_contrast, motion_blur, crowded_scene, other>"
}}

Be specific about the visual conditions. Focus on what makes this frame challenging for a computer vision model."""


def scene_description_prompt():
    """
    Ask the VLM to describe all objects visible in a frame.
    Used for pseudo-annotation of unlabelled data.
    """
    return """You are a road safety expert analyzing a dashcam frame from Ghana.

List every object relevant to driving safety that you can see. For each object, estimate its approximate position in the image.

Respond in JSON format:
{
    "objects": [
        {
            "class": "<object type, e.g. car, motorcycle, pedestrian, pothole, speed_bump, animal, traffic_cone, open_drain>",
            "location": "<approximate position: left/center/right, near/mid/far>",
            "confidence": "<how certain you are: high/medium/low>",
            "notes": "<any special conditions, e.g. 'partially occluded by dust'>"
        }
    ],
    "road_condition": "<overall assessment: good, fair, poor, hazardous>",
    "visibility": "<clear, hazy, dusty, shadowed, rain>"
}"""


def hard_negative_ranking_prompt():
    """
    Ask the VLM to assess how difficult a frame is for detection.
    Used to prioritize which frames to include in retraining.
    """
    return """Rate the difficulty of this road scene for an object detection model.

Consider:
1. Are there objects that are hard to distinguish from the background?
2. Is there dust, haze, shadow, or glare that obscures objects?
3. Are objects crowded or heavily occluded?
4. Are there unusual objects not commonly found in training datasets?

Respond in JSON format:
{
    "difficulty_score": <1-10, where 10 is extremely challenging>,
    "challenge_factors": ["list of specific factors making detection hard"],
    "retraining_value": "<high/medium/low - how valuable is this frame for retraining?>",
    "recommended_augmentations": ["augmentation strategies that would help, e.g. 'add dust haze', 'increase shadow intensity'"]
}"""


def comparative_prompt(sprint2_detections):
    """
    Ask the VLM to compare its own assessment against the detector's output.
    """
    det_str = ', '.join([f"{d['class']}({d['confidence']:.2f})" for d in sprint2_detections[:10]])
    return f"""A detection model analyzed this road scene and produced these detections: [{det_str}].

Compare the model's detections against what you can actually see:
1. Which detections are CORRECT (true positives)?
2. Which detections are WRONG (false positives)?
3. What objects did the model MISS (false negatives)?

Respond in JSON format:
{{
    "true_positives": ["list of correct detections"],
    "false_positives": ["list of incorrect detections with reason"],
    "false_negatives": ["list of missed objects with likely reason"],
    "overall_accuracy": "<good/fair/poor>",
    "key_improvement_area": "<single most important thing to fix>"
}}"""
