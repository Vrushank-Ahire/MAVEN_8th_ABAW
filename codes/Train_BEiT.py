import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BeitModel
import copy
import logging
from tqdm import tqdm
import sys


def compute_ccc(preds, targets):
    preds_mean = preds.mean()
    targets_mean = targets.mean()
    preds_var = preds.var() + 1e-8
    targets_var = targets.var() + 1e-8
    if preds_var < 1e-8 or targets_var < 1e-8:
        return 0.0
    covariance = torch.mean((preds - preds_mean) * (targets - targets_mean))
    correlation = covariance / (torch.sqrt(preds_var) * torch.sqrt(targets_var) + 1e-8)
    ccc = (2 * covariance) / (preds_var + targets_var + (preds_mean - targets_mean) ** 2 + 1e-8)
    return ccc.item()


# MultiVideoDataLoader remains unchanged
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
            prev_end = timestep_dict.get('prev_end', 0.0)  # Default to 0.0 if not present
            return start, end, prev_end
        except Exception as e:
            print(f"Error parsing timestep file {timestep_path}: {e}")
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
                    "annotations": os.path.join(seg_path, "annotations.txt"),
                    "timestep": os.path.join(seg_path, "timestep.txt")
                }
                if all(os.path.isfile(p) for p in required_files.values()):
                    start, end, prev_end = self._parse_timestep(required_files["timestep"])
                    segments_list.append((vid_id, seg_id, start, end, prev_end))
        return segments_list

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        vid_id, seg_id, start_time, end_time, prev_end_time = self.segments[idx]
        seg_path = os.path.join(self.embeddings_root, vid_id, seg_id)
        
        try:
            # Modified torch.load calls with weights_only=True
            frames = torch.load(os.path.join(seg_path, "frames_embedding.pt"), weights_only=True)
            audio = torch.load(os.path.join(seg_path, "audio_embedding.pt"), weights_only=True)
            transcript = torch.load(os.path.join(seg_path, "transcript_embedding.pt"), weights_only=True)
            
            frames += torch.randn_like(frames) * 0.01
            audio += torch.randn_like(audio) * 0.01
            transcript += torch.randn_like(transcript) * 0.01
            
            frames = (frames - frames.mean()) / (frames.std() + 1e-8)
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            transcript = (transcript - transcript.mean()) / (transcript.std() + 1e-8)
            
            with open(os.path.join(seg_path, "annotations.txt"), "r") as f:
                annotations = f.read().strip().splitlines()[1:]
            num_frames = int((end_time - (prev_end_time if prev_end_time != 0 else 0)) * self.fps)
            invalid_count = 0
            for ann in annotations:
                try:
                    parts = ann.strip().split(',')
                    float(parts[0]), float(parts[1])  # Check valence and arousal
                except:
                    invalid_count += 1
            if invalid_count > 0:
                print(f"Segment {vid_id}/{seg_id}: {invalid_count}/{len(annotations)} invalid annotations")
        except Exception as e:
            print(f"Error loading data for {vid_id}/{seg_id}: {e}")
            return None
        
        return frames, audio, transcript, annotations, num_frames, vid_id, seg_id


# MultiModalCollator remains unchanged
class MultiModalCollator:
    def __call__(self, batch):
        frames_batch, audio_batch, transcript_batch = [], [], []
        annotation_batch, num_frames_batch = [], []
        vid_id_batch, seg_id_batch = [], []
        
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        max_frames_in_batch = max(b[4] for b in batch)
        
        for frames, audio, transcript, annotations, num_frames, vid_id, seg_id in batch:
            frames_batch.append(frames)
            audio_batch.append(audio.squeeze())
            transcript_batch.append(transcript.squeeze())
            annotation_batch.append(annotations)
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
            'annotations': annotation_batch,
            'num_frames': num_frames_batch,
            'vid_id': vid_id_batch[0],
            'seg_id': seg_id_batch[0]
        }


# MultiHeadCrossModalAttention remains unchanged
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


# Renamed and modified to use valence and arousal
class ValenceArousalRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Outputs valence and arousal
        )

    def forward(self, x):
        return self.fc(x)


# CustomBeitEncoder remains unchanged
class CustomBeitEncoder(nn.Module):
    def __init__(self, beit_encoder):
        super().__init__()
        self.layer = beit_encoder.layer

    def forward(self, hidden_states, output_attentions=False):
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            layer_outputs = layer_module(
                hidden_states,
                head_mask=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        return hidden_states if not output_attentions else (hidden_states, all_attentions)


class MultiModalEmotionModel(nn.Module):
    def __init__(self, embed_dim=512, num_heads=2, dropout=0.1, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Default to "cuda" for DataParallel
        
        self.concat_dim = embed_dim * 3
        self.multihead_attention = MultiHeadCrossModalAttention(embed_dim, num_heads, dropout)
        
        try:
            base_beit3 = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            print("BEiT-3 base model loaded successfully")
        except Exception as e:
            print(f"Error loading BEiT-3: {e}")
            base_beit3 = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        
        self.beit_hidden_size = getattr(base_beit3.config, 'hidden_size', 768)
        self.projection_layer = nn.Linear(self.concat_dim, self.beit_hidden_size)
        self.beit3_encoders = nn.ModuleList([
            CustomBeitEncoder(copy.deepcopy(base_beit3.encoder)) for _ in range(2)
        ])
        self.regressor = ValenceArousalRegressor(self.beit_hidden_size * 2)
        self.mse_loss = nn.MSELoss()
        self.criterion = self.combined_loss
        
        self.to(self.device)

    def combined_loss(self, preds, targets):
        mse = self.mse_loss(preds, targets)
        ccc_valence = compute_ccc(preds[:, 0], targets[:, 0])
        ccc_arousal = compute_ccc(preds[:, 1], targets[:, 1])
        ccc_loss = 1 - (ccc_valence + ccc_arousal) / 2
        return mse + 0.1 * ccc_loss

    def forward(self, batch_data):
        frames = batch_data['frames'].to(self.device)
        audio = batch_data['audio'].to(self.device)
        text = batch_data['transcript'].to(self.device)
        num_frames = batch_data['num_frames']
        
        max_frames_in_batch = frames.size(1)
        video_enhanced, audio_enhanced, text_enhanced = self.multihead_attention(frames, audio, text)
        audio_enhanced = audio_enhanced.unsqueeze(1).expand(-1, max_frames_in_batch, -1)
        text_enhanced = text_enhanced.unsqueeze(1).expand(-1, max_frames_in_batch, -1)
        
        concat_features = torch.cat([video_enhanced, audio_enhanced, text_enhanced], dim=-1)
        projected = self.projection_layer(concat_features)
        
        beit_outputs = [encoder(projected) for encoder in self.beit3_encoders]
        combined = torch.cat(beit_outputs, dim=-1)
        
        return self.regressor(combined)
    
    def configure_optimizers(self, lr=1e-3):
        params = [
            {'params': self.multihead_attention.parameters()},
            {'params': self.projection_layer.parameters()},
            {'params': self.regressor.parameters()}
        ] + [{'params': encoder.parameters()} for encoder in self.beit3_encoders]
        return optim.Adam(params, lr=lr, weight_decay=1e-4)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss, total_samples = 0.0, 0
    
    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch in progress_bar:
        if batch is None:
            print("Skipping empty batch")
            continue
        # Move data to device (DataParallel will split it across GPUs)
        batch['frames'] = batch['frames'].to(device)
        batch['audio'] = batch['audio'].to(device)
        batch['transcript'] = batch['transcript'].to(device)
        num_frames = batch['num_frames']
        batch_size = len(num_frames)
        
        max_frames_in_batch = batch['frames'].size(1)
        batch_targets = torch.zeros(batch_size, max_frames_in_batch, 2, device=device)
        mask = torch.zeros(batch_size, max_frames_in_batch, dtype=torch.bool, device=device)
        
        for batch_idx, (ann_list, num_f) in enumerate(zip(batch['annotations'], num_frames)):
            num_frames_valid = min(num_f, len(ann_list))
            mask[batch_idx, :num_frames_valid] = True
            for i in range(num_frames_valid):
                try:
                    parts = ann_list[i].strip().split(',')
                    if len(parts) < 2:
                        raise ValueError("Insufficient fields")
                    valence, arousal = float(parts[0]), float(parts[1])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Invalid annotation at index {i} for batch {batch_idx}: {e}")
                    valence, arousal = 0.0, 0.0
                batch_targets[batch_idx, i] = torch.tensor([valence, arousal], device=device)
        
        optimizer.zero_grad()
        outputs = model(batch)  # DataParallel handles splitting across GPUs
        masked_outputs, masked_targets = outputs[mask], batch_targets[mask]
        loss = model.criterion(masked_outputs, masked_targets)
        
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        num_valid_frames = mask.sum().item()
        running_loss += loss.item() * num_valid_frames
        total_samples += num_valid_frames
        
        progress_bar.set_postfix({'batch_loss': loss.item(), 'grad_norm': total_norm})
    
    progress_bar.close()
    return running_loss / total_samples if total_samples > 0 else running_loss


def evaluate(model, dataloader, device):
    model.eval()
    running_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                continue
            batch['frames'] = batch['frames'].to(device)
            batch['audio'] = batch['audio'].to(device)
            batch['transcript'] = batch['transcript'].to(device)
            num_frames = batch['num_frames']
            batch_size = len(num_frames)
            
            max_frames_in_batch = batch['frames'].size(1)
            batch_targets = torch.zeros(batch_size, max_frames_in_batch, 2, device=device)
            mask = torch.zeros(batch_size, max_frames_in_batch, dtype=torch.bool, device=device)
            
            for batch_idx, (ann_list, num_f) in enumerate(zip(batch['annotations'], num_frames)):
                num_frames_valid = min(num_f, len(ann_list))
                mask[batch_idx, :num_frames_valid] = True
                for i in range(num_frames_valid):
                    try:
                        parts = ann_list[i].strip().split(',')
                        if len(parts) < 4:
                            raise ValueError("Insufficient fields")
                        t, i_val = float(parts[2]), float(parts[3])
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Invalid annotation at index {i} for batch {batch_idx}: {e}")
                        t, i_val = 2.5, 0.5
                    batch_targets[batch_idx, i] = torch.tensor([t, i_val], device=device)
            
            outputs = model(batch)
            masked_outputs, masked_targets = outputs[mask], batch_targets[mask]
            
            if torch.isnan(masked_outputs).any() or torch.isinf(masked_outputs).any():
                print(f"Warning: NaN or inf in masked_outputs: {masked_outputs}")
            if torch.isnan(masked_targets).any() or torch.isinf(masked_targets).any():
                print(f"Warning: NaN or inf in masked_targets: {masked_targets}")
            
            loss = model.criterion(masked_outputs, masked_targets)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN or inf: {loss.item()}")
                continue
            
            num_valid_frames = mask.sum().item()
            running_loss += loss.item() * num_valid_frames
            total_samples += num_valid_frames
            all_preds.append(masked_outputs.cpu())
            all_targets.append(masked_targets.cpu())
            
            progress_bar.set_postfix({'batch_loss': loss.item()})
    
    progress_bar.close()
    if total_samples == 0:
        print("Warning: No valid samples in evaluation")
        return {'loss': float('nan'), 'mse': float('nan'), 'ccc_valence': 0.0, 'ccc_arousal': 0.0, 'mean_ccc': 0.0}
    
    all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
    
    # Convert theta and intensity to valence and arousal for both predictions and targets
    pred_theta, pred_intensity = all_preds[:, 0], all_preds[:, 1]
    target_theta, target_intensity = all_targets[:, 0], all_targets[:, 1]
    
    pred_valence = pred_intensity * torch.cos(pred_theta)
    pred_arousal = pred_intensity * torch.sin(pred_theta)
    target_valence = target_intensity * torch.cos(target_theta)
    target_arousal = target_intensity * torch.sin(target_theta)
    
    # Compute CCC for valence and arousal
    ccc_valence = compute_ccc(pred_valence, target_valence)
    ccc_arousal = compute_ccc(pred_arousal, target_arousal)
    
    return {
        'loss': running_loss / total_samples,
        'mse': torch.nn.functional.mse_loss(all_preds, all_targets).item(),
        'ccc_valence': ccc_valence,
        'ccc_arousal': ccc_arousal,
        'mean_ccc': (ccc_valence + ccc_arousal) / 2
    }



def test(model, test_loader, device, model_path="best_model.pth", output_file="predictions.txt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    predictions = []
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue
            batch['frames'] = batch['frames'].to(device)
            batch['audio'] = batch['audio'].to(device)
            batch['transcript'] = batch['transcript'].to(device)
            num_frames = batch['num_frames'][0]
            
            vid_id = batch['vid_id']
            frame_folder = f"{vid_id}"
            
            outputs = model(batch)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or inf in model outputs: {outputs}")
                continue
            valence, arousal = outputs[0, :, 0], outputs[0, :, 1]
            
            for frame_idx in range(num_frames):
                frame_num = f"{frame_idx + 1:05d}"
                image_location = f"{frame_folder}/{frame_num}.jpg"
                pred_valence = valence[frame_idx].item()
                pred_arousal = arousal[frame_idx].item()
                predictions.append((image_location, pred_valence, pred_arousal))
    
    progress_bar.close()
    if not predictions:
        raise ValueError("No valid predictions were generated during testing")
    
    with open(output_file, 'w') as f:
        f.write("image_location,valence,arousal\n")
        for image_loc, val, aro in predictions:
            f.write(f"{image_loc},{val:.6f},{aro:.6f}\n")
    
    total_frames = len(predictions)
    print(f"Computed valence and arousal for {total_frames} frames, saved to {output_file}")
    logging.info(f"Test Results - Computed for {total_frames} frames, saved to {output_file}")
    return predictions


def train(devices):
    train_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/train"
    val_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/val"
    
    train_dataset = MultiVideoDataLoader(train_root)
    val_dataset = MultiVideoDataLoader(val_root)
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    all_targets = []
    for batch in DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=MultiModalCollator()):
        if batch is None:
            continue
        for ann_list, num_f in zip(batch['annotations'], batch['num_frames']):
            for i in range(min(num_f, len(ann_list))):
                try:
                    parts = ann_list[i].strip().split(',')
                    valence, arousal = float(parts[0]), float(parts[1])
                    all_targets.append([valence, arousal])
                except:
                    continue
    all_targets = torch.tensor(all_targets)
    print(f"Target valence mean: {all_targets[:, 0].mean():.4f}, std: {all_targets[:, 0].std():.4f}")
    print(f"Target arousal mean: {all_targets[:, 1].mean():.4f}, std: {all_targets[:, 1].std():.4f}")
    
    collator = MultiModalCollator()
    train_loader = DataLoader(train_dataset, batch_size=len(devices) * 2, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=len(devices) * 2, shuffle=False, collate_fn=collator, num_workers=0)
    
    # Initialize model
    model = MultiModalEmotionModel()
    # Configure optimizers BEFORE wrapping with DataParallel
    optimizer = model.configure_optimizers(lr=1e-3)
    # Now wrap with DataParallel if using multiple GPUs
    if len(devices) > 1:
        model = nn.DataParallel(model, device_ids=devices)
        print(f"Using DataParallel with devices: {devices}")
    model = model.to(devices[0])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    num_epochs, best_val_loss = 50, float('inf')
    try:
        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            train_loss = train_epoch(model, train_loader, optimizer, devices[0])
            val_metrics = evaluate(model, val_loader, devices[0])
            
            print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_metrics['loss']:.4f}, Mean CCC: {val_metrics['mean_ccc']:.4f}")
            logging.info(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, {val_metrics}")
            
            scheduler.step(val_metrics['loss'])
            if val_metrics['loss'] < best_val_loss and not torch.isnan(torch.tensor(val_metrics['loss'])):
                best_val_loss = val_metrics['loss']
                torch.save(model.module.state_dict(), "best_model.pth")
                print(f"Saved best model with val loss: {best_val_loss:.4f}")
            
            if epoch > 5 and val_metrics['loss'] >= best_val_loss:
                print("Loss plateaued, reinitializing model")
                model = MultiModalEmotionModel()
                optimizer = model.configure_optimizers(lr=1e-3)  # Reconfigure optimizer
                if len(devices) > 1:
                    model = nn.DataParallel(model, device_ids=devices)
                model = model.to(devices[0])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        if best_val_loss != float('inf'):
            torch.save(model.module.state_dict(), "best_model.pth")
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        sys.exit(0)


def test_main(devices):
    test_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/val"
    test_dataset = MultiVideoDataLoader(test_root)
    collator = MultiModalCollator()
    test_loader = DataLoader(test_dataset, batch_size=len(devices) * 4, shuffle=False, 
                             collate_fn=collator, num_workers=4)
    
    model = MultiModalEmotionModel()
    if len(devices) > 1:
        model = nn.DataParallel(model, device_ids=devices)
    model = model.to(devices[0])
    predictions = test(model, test_loader, devices[0], output_file="predictions.txt")
    valence = torch.tensor([p[1] for p in predictions])
    arousal = torch.tensor([p[2] for p in predictions])
    torch.save({'valence': valence, 'arousal': arousal}, "test_predictions.pth")
    print("Saved test predictions to test_predictions.pth")

def main():
    # Specify available GPU devices (e.g., [0, 1, 2, 3] or [4, 5])
    devices = [3,4,6] if torch.cuda.device_count() > 1 else [4]  # Example: use GPUs 4 and 5
    if not torch.cuda.is_available():
        devices = [torch.device("cpu")]
    print(f"Using devices: {devices}")
    logging.basicConfig(filename="training_log.log", level=logging.INFO, 
                       format="%(asctime)s - %(message)s")
    
    train_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/train"
    val_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers/val"
    if not (os.path.exists(train_root) and os.path.exists(val_root)):
        print(f"Error: Data directories {train_root} or {val_root} not found")
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_main(devices)
    else:
        train(devices)

if __name__ == "__main__":
    main()