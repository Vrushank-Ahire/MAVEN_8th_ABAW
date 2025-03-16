import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

# Setup logging
def setup_logging(log_file="predictions.txt"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(message)s",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# MultiVideoDataLoader
class MultiVideoDataLoader:
    def __init__(self, embeddings_root, fps=30):
        self.embeddings_root = embeddings_root
        self.fps = fps
        self.segments = self._build_segments_list()

    def _get_video_ids(self):
        return sorted([d for d in os.listdir(self.embeddings_root) 
                      if os.path.isdir(os.path.join(self.embeddings_root, d))])

    def _parse_timestep(self, timestep_path):
        try:
            with open(timestep_path, 'r') as f:
                lines = f.readlines()
            timestep_dict = {}
            for line in lines:
                key, value = line.strip().split(': ')
                timestep_dict[key] = float(value)
            start = timestep_dict.get('start', 0.0)
            end = timestep_dict.get('end', 0.0)
            prev_end = timestep_dict.get('prev_end', 0.0)
            return start, end, prev_end
        except Exception as e:
            # logging.warning(f"Error parsing timestep file {timestep_path}: {e}")
            return 0.0, 0.0, 0.0

    def _build_segments_list(self):
        segments_list = []
        for vid_id in self._get_video_ids():
            video_path = os.path.join(self.embeddings_root, vid_id)
            segment_ids = sorted([d for d in os.listdir(video_path) 
                                if os.path.isdir(os.path.join(video_path, d))])
            for seg_id in segment_ids:
                seg_path = os.path.join(video_path, seg_id)
                required_files = {
                    "frames": os.path.join(seg_path, "frames_embedding.pt"),
                    "audio": os.path.join(seg_path, "audio_embedding.pt"),
                    "transcript": os.path.join(seg_path, "transcript_embedding.pt"),
                    "timestep": os.path.join(seg_path, "timestep.txt")
                }
                if all(os.path.isfile(p) for p in required_files.values()):
                    start, end, prev_end = self._parse_timestep(required_files["timestep"])
                    segments_list.append((vid_id, seg_id, start, end, prev_end))
                else:
                    logging.warning(f"Missing required files in {seg_path}: {required_files}")
        return segments_list

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        vid_id, seg_id, start_time, end_time, prev_end_time = self.segments[idx]
        seg_path = os.path.join(self.embeddings_root, vid_id, seg_id)
        
        try:
            frames = torch.load(os.path.join(seg_path, "frames_embedding.pt"), weights_only=True)
            audio = torch.load(os.path.join(seg_path, "audio_embedding.pt"), weights_only=True)
            transcript = torch.load(os.path.join(seg_path, "transcript_embedding.pt"), weights_only=True)
            
            frames += torch.randn_like(frames) * 0.01
            audio += torch.randn_like(audio) * 0.01
            transcript += torch.randn_like(transcript) * 0.01
            
            frames = (frames - frames.mean()) / (frames.std() + 1e-8)
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            transcript = (transcript - transcript.mean()) / (transcript.std() + 1e-8)
            
            num_frames = int((end_time - (prev_end_time if prev_end_time != 0 else 0)) * self.fps)
            if num_frames <= 0:
                logging.warning(f"Invalid num_frames ({num_frames}) for {vid_id}/{seg_id}")
                return None
        except Exception as e:
            logging.error(f"Error loading data for {vid_id}/{seg_id}: {e}")
            return None
        
        return frames, audio, transcript, num_frames, vid_id, seg_id

# MultiModalCollator
class MultiModalCollator:
    def __call__(self, batch):
        frames_batch, audio_batch, transcript_batch = [], [], []
        num_frames_batch, vid_id_batch, seg_id_batch = [], [], []
        
        batch = [b for b in batch if b is not None]
        if not batch:
            logging.warning("Collator received no valid items")
            return None
        
        max_frames_in_batch = max(b[3] for b in batch)
        
        for frames, audio, transcript, num_frames, vid_id, seg_id in batch:
            frames_batch.append(frames)
            audio_batch.append(audio.squeeze())
            transcript_batch.append(transcript.squeeze())
            num_frames_batch.append(num_frames)
            vid_id_batch.append(vid_id)
            seg_id_batch.append(seg_id)
            
            if frames.shape[0] < max_frames_in_batch:
                padding = torch.zeros(max_frames_in_batch - frames.shape[0], frames.shape[1])
                frames_batch[-1] = torch.cat([frames, padding], dim=0)
            elif frames.shape[0] > max_frames_in_batch:
                frames_batch[-1] = frames[:max_frames_in_batch]
        
        return {
            'frames': torch.stack(frames_batch),
            'audio': torch.stack(audio_batch),
            'transcript': torch.stack(transcript_batch),
            'num_frames': num_frames_batch,
            'vid_id': vid_id_batch,
            'seg_id': seg_id_batch
        }

# MultiHeadCrossModalAttention
class MultiHeadCrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.video_dim, self.audio_dim, self.text_dim = 1024, 768, 768
        self.video_proj = nn.Linear(self.video_dim, embed_dim)
        self.audio_proj = nn.Linear(self.audio_dim, embed_dim)
        self.text_proj = nn.Linear(self.text_dim, embed_dim)
        
        self.attn_v2a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_v2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_a2v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_a2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t2v = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_t2a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, video_emb, audio_emb, text_emb):
        video_proj = self.video_proj(video_emb)
        audio_proj = self.audio_proj(audio_emb).unsqueeze(1)
        text_proj = self.text_proj(text_emb).unsqueeze(1)
        
        v2a, _ = self.attn_v2a(query=video_proj, key=audio_proj, value=audio_proj)
        v2t, _ = self.attn_v2t(query=video_proj, key=text_proj, value=text_proj)
        a2v, _ = self.attn_a2v(query=audio_proj, key=video_proj, value=video_proj)
        a2t, _ = self.attn_a2t(query=audio_proj, key=text_proj, value=text_proj)
        t2v, _ = self.attn_t2v(query=text_proj, key=video_proj, value=video_proj)
        t2a, _ = self.attn_t2a(query=text_proj, key=audio_proj, value=audio_proj)
        
        video_enhanced = self.layer_norm(self.dropout((video_proj + v2a + v2t) / 3))
        audio_enhanced = self.layer_norm(self.dropout((audio_proj + a2v + a2t) / 3)).squeeze(1)
        text_enhanced = self.layer_norm(self.dropout((text_proj + t2v + t2a) / 3)).squeeze(1)
        
        return video_enhanced, audio_enhanced, text_enhanced

# ThetaIntensityRegressor
class ThetaIntensityRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.fc(x)

# MultiModalEmotionModel
class MultiModalEmotionModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=2, dropout=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        self.concat_dim = embed_dim * 3
        self.multihead_attention = MultiHeadCrossModalAttention(embed_dim, num_heads, dropout)
        self.regressor = ThetaIntensityRegressor(self.concat_dim)
        self.to(self.device)

    def forward(self, batch_data):
        frames = batch_data['frames'].to(self.device)
        audio = batch_data['audio'].to(self.device)
        text = batch_data['transcript'].to(self.device)
        
        max_frames_in_batch = frames.size(1)
        video_enhanced, audio_enhanced, text_enhanced = self.multihead_attention(frames, audio, text)
        
        audio_enhanced = audio_enhanced.unsqueeze(1).expand(-1, max_frames_in_batch, -1)
        text_enhanced = text_enhanced.unsqueeze(1).expand(-1, max_frames_in_batch, -1)
        
        concat_features = torch.cat([video_enhanced, audio_enhanced, text_enhanced], dim=-1)
        theta_intensity = self.regressor(concat_features)
        return theta_intensity

# Test function with updated image_location format
def test(model, test_loader, device, model_path="best_model.pth", output_file="predictions.txt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    predictions = []
    frame_counter = {}  # To track frame numbers across segments for each vid_id
    progress_bar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                logging.warning(f"Batch {batch_idx} is None, skipping")
                continue
            
            # logging.info(f"Processing batch {batch_idx} with keys: {list(batch.keys())}")
            # logging.info(f"num_frames: {batch['num_frames']}, vid_ids: {batch['vid_id']}")
            
            batch['frames'] = batch['frames'].to(device)
            batch['audio'] = batch['audio'].to(device)
            batch['transcript'] = batch['transcript'].to(device)
            num_frames = batch['num_frames']
            vid_ids = batch['vid_id']
            
            outputs = model(batch)
            logging.info(f"image_location,valence,arousal")
            
            for b_idx in range(len(vid_ids)):
                vid_id = vid_ids[b_idx]
                num_f = num_frames[b_idx]
                
                if vid_id not in frame_counter:
                    frame_counter[vid_id] = 0
                
                # logging.info(f"Sample {b_idx}: vid_id={vid_id}, num_frames={num_f}")
                
                theta = outputs[b_idx, :num_f, 0]
                intensity = outputs[b_idx, :num_f, 1]
                valence = intensity * torch.cos(theta)
                arousal = intensity * torch.sin(theta)
                
                for frame_idx in range(num_f):
                    frame_counter[vid_id] += 1
                    frame_num = f"{frame_counter[vid_id]:05d}"  # e.g., 00001
                    image_location = f"{vid_id}/{frame_num}.jpg"  # e.g., 2-30-640x360/00001.jpg
                    pred_valence = valence[frame_idx].item()
                    pred_arousal = arousal[frame_idx].item()
                    predictions.append((image_location, pred_valence, pred_arousal))
                    logging.info(f"{image_location},{pred_valence},{pred_arousal}")
    
    progress_bar.close()
    if not predictions:
        # logging.error("No predictions generated. Check test dataset or collator.")
        raise ValueError("No valid predictions were generated during testing")
    
    # Save predictions
    with open(output_file, 'w') as f:
        f.write("image_location,valence,arousal\n")
        for image_loc, val, aro in predictions:
            f.write(f"{image_loc},{val:.1f},{aro:.1f}\n")
    
    total_frames = len(predictions)
    print(f"Computed valence and arousal for {total_frames} frames, saved to {output_file}")
    logging.info(f"Test Results - Computed for {total_frames} frames, saved to {output_file}")
    return predictions

# Main testing function
def main():
    setup_logging("test_log.txt")
    
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    test_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/test"
    
    if not os.path.exists(test_root):
        logging.error(f"Test directory {test_root} not found")
        raise FileNotFoundError(f"Test directory {test_root} not found")
    
    # Load test dataset
    test_dataset = MultiVideoDataLoader(test_root)
    if len(test_dataset) == 0:
        logging.error(f"Test dataset is empty at {test_root}")
        raise ValueError(f"Test dataset is empty at {test_root}")
    # logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # DataLoader
    collator = MultiModalCollator()
    batch_size = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collator, num_workers=4)
    
    # Initialize model
    model = MultiModalEmotionModel(device=device)
    model = model.to(device)
    
    # Run test
    model_path = "/home/lownish/abaw2/home/kunal/transfer/model_codes/multimodal/best_model.pth"
    predictions = test(model, test_loader, device, model_path=model_path, output_file="predictions.txt")
    
    # Optional: Save as tensor dictionary
    valence = torch.tensor([p[1] for p in predictions])
    arousal = torch.tensor([p[2] for p in predictions])
    torch.save({'valence': valence, 'arousal': arousal}, "test_predictions.pth")
    print("Saved test predictions to test_predictions.pth")

if __name__ == "__main__":
    main()