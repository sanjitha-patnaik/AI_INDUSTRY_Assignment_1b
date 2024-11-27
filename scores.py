import os
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from fid_metrics.inception import InceptionV3
from fid_metrics.inception3d import InceptionI3d
from fid_metrics.fid import calculate_act_statistics, calculate_frechet_distance


def load_video_frames(video_path, max_frames=120):
    """Load video frames up to a maximum number."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def preprocess_frames_2d(frames, size=(299, 299)):
    """Preprocess frames for 2D models (InceptionV3)."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [transform(frame) for frame in frames]
    return torch.stack(tensors)  # Stack along the batch dimension


def preprocess_frames_3d(frames, size=(224, 224)):
    """Preprocess frames for 3D models (InceptionI3d)."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [transform(frame) for frame in frames]
    tensor = torch.stack(tensors).permute(1, 0, 2, 3)  # (C, T, H, W)
    return tensor.unsqueeze(0)  # Add batch dimension: (B, C, T, H, W)


def get_inception_model(model_type='2d'):
    """Load the appropriate Inception model."""
    if model_type == '2d':
        model = InceptionV3(output_blocks=[InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    elif model_type == '3d':
        model = InceptionI3d(num_classes=400, in_channels=3)
        model.replace_logits(400)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def calculate_fid_score(frames1, frames2, model):
    """Calculate FIDVID score using InceptionV3."""
    tensor1 = preprocess_frames_2d(frames1)
    tensor2 = preprocess_frames_2d(frames2)

    if torch.cuda.is_available():
        tensor1, tensor2 = tensor1.cuda(), tensor2.cuda()

    with torch.no_grad():
        features1 = model(tensor1)[0]
        features2 = model(tensor2)[0]

    # Ensure the features are flattened into 2D
    features1 = features1.view(features1.size(0), -1).cpu().numpy()
    features2 = features2.view(features2.size(0), -1).cpu().numpy()

    m1, s1 = calculate_act_statistics(features1)
    m2, s2 = calculate_act_statistics(features2)

    return calculate_frechet_distance(m1, s1, m2, s2)


def calculate_fvd_score(frames1, frames2, model):
    """Calculate FVD score using InceptionI3D."""
    tensor1 = preprocess_frames_3d(frames1)
    tensor2 = preprocess_frames_3d(frames2)

    if torch.cuda.is_available():
        tensor1, tensor2 = tensor1.cuda(), tensor2.cuda()

    with torch.no_grad():
        features1 = model.extract_features(tensor1)
        features2 = model.extract_features(tensor2)

    # Flatten features for covariance calculation
    features1 = features1.view(features1.size(0), -1).cpu().numpy()
    features2 = features2.view(features2.size(0), -1).cpu().numpy()

    m1, s1 = calculate_act_statistics(features1)
    m2, s2 = calculate_act_statistics(features2)

    return calculate_frechet_distance(m1, s1, m2, s2)


def process_video_folders(real_videos_folder, synthetic_videos_folder):
    """Process all video pairs in the given folders and return a result table."""
    real_videos = sorted(os.listdir(real_videos_folder))
    synthetic_videos = sorted(os.listdir(synthetic_videos_folder))

    # Ensure both folders have the same number of videos
    if len(real_videos) != len(synthetic_videos):
        raise ValueError("The number of videos in both folders must match.")

    results = []

    # Load models
    inception_2d = get_inception_model('2d')
    inception_3d = get_inception_model('3d')

    # Process each pair of videos
    for real_video, synthetic_video in zip(real_videos, synthetic_videos):
        real_video_path = os.path.join(real_videos_folder, real_video)
        synthetic_video_path = os.path.join(synthetic_videos_folder, synthetic_video)

        # Load frames
        frames1 = load_video_frames(real_video_path)
        frames2 = load_video_frames(synthetic_video_path)

        # Calculate FIDVID and FVD
        fid_score = calculate_fid_score(frames1, frames2, inception_2d)
        fvd_score = calculate_fvd_score(frames1, frames2, inception_3d)

        # Append results
        results.append({
            "Real Video": real_video,
            "Synthetic Video": synthetic_video,
            "FIDVID Score": fid_score,
            "FVD Score": fvd_score
        })

    # Convert results to a Pandas DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def setup_gradio():
    """Set up the Gradio interface."""
    def process_and_display(real_videos_folder, synthetic_videos_folder):
        results_df = process_video_folders(real_videos_folder, synthetic_videos_folder)
        return results_df

    iface = gr.Interface(
        fn=process_and_display,
        inputs=[
            gr.Textbox(label="Path to Real Videos Folder"),
            gr.Textbox(label="Path to Synthetic Videos Folder")
        ],
        outputs="dataframe",
        title="Video Evaluation (FIDVID and FVD Metrics)",
        description="Provide folder paths for real and synthetic videos to calculate FIDVID and FVD scores."
    )
    return iface


def main():
    """Launch the Gradio app."""
    iface = setup_gradio()
    iface.launch()


if _name_ == "_main_":
    main()