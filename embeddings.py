import os
import re
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import librosa
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
from transformers import HubertModel, Wav2Vec2FeatureExtractor, RobertaModel, RobertaTokenizer
from tqdm import tqdm

class MultiVideoDataLoader:
    def __init__(self, frames_root, audio_root, transcript_root, annotation_root, fps=30):
        self.frames_root = frames_root
        self.audio_root = audio_root
        self.transcript_root = transcript_root
        self.annotation_root = annotation_root
        self.fps = fps
        self.video_ids = self._get_video_ids()
        self.segments = self._build_segments_list()

    def _get_video_ids(self):
        """Get list of video IDs, skipping those with 'left' or 'right' in the name."""
        video_ids = []
        for file in sorted(os.listdir(self.transcript_root)):
            if file.endswith('.txt'):
                vid_id = os.path.splitext(file)[0]
                # Skip if 'left' or 'right' is in the vid_id
                if 'left' in vid_id.lower() or 'right' in vid_id.lower():
                    print(f"Skipping {vid_id} due to 'left' or 'right' in name.")
                    continue
                frames_path = os.path.join(self.frames_root, vid_id)
                audio_path = os.path.join(self.audio_root, f"{vid_id}.wav")
                annotation_path = os.path.join(self.annotation_root, f"{vid_id}.txt")
                if os.path.isdir(frames_path) and os.path.isfile(audio_path) and os.path.isfile(annotation_path):
                    video_ids.append(vid_id)
        return video_ids

    def _parse_transcript(self, lines):
        """Parse transcript file to extract segments with start and end times."""
        segments = []
        pattern = re.compile(r"\[(\d+\.\d+)s\s*-\s*(\d+\.\d+)s\]\s*(.*)")
        for line in lines:
            match = pattern.match(line.strip())
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                text = match.group(3)
                if end_time > start_time:
                    segments.append({'start': start_time, 'end': end_time, 'text': text})
        return segments

    def _build_segments_list(self):
        """Build list of segments with start and end times."""
        segments_list = []
        for vid_id in self.video_ids:
            transcript_path = os.path.join(self.transcript_root, f"{vid_id}.txt")
            with open(transcript_path, "r") as f:
                lines = f.readlines()
            transcript_segs = self._parse_transcript(lines)

            if not transcript_segs:
                # Empty transcript: treat entire video as one segment
                frames_dir = os.path.join(self.frames_root, vid_id)
                total_frames = len(sorted(glob.glob(os.path.join(frames_dir, '*'))))
                end_time = total_frames / self.fps
                segments_list.append((vid_id, None, 0.0, end_time, 1))
            else:
                prev_end_time = 0.0
                for idx, seg in enumerate(transcript_segs, 1):
                    start_time = prev_end_time
                    end_time = seg['end']
                    segments_list.append((vid_id, seg, start_time, end_time, idx))
                    prev_end_time = end_time
        return segments_list

    def __len__(self):
        """Return total number of segments."""
        return len(self.segments)

    def __getitem__(self, idx):
        """Get data for a specific segment."""
        vid_id, seg, start_time, end_time, seg_num = self.segments[idx]

        # Load frames
        frames_dir = os.path.join(self.frames_root, vid_id)
        frames_files = sorted(glob.glob(os.path.join(frames_dir, '*')))
        total_frames = len(frames_files)
        video_duration = total_frames / self.fps

        # Load annotations (skip header)
        annotation_path = os.path.join(self.annotation_root, f"{vid_id}.txt")
        with open(annotation_path, "r") as f:
            all_annotations = f.read().strip().splitlines()[1:]  # Skip header

        # Load audio
        audio_path = os.path.join(self.audio_root, f"{vid_id}.wav")
        audio, audio_sr = librosa.load(audio_path, sr=16000)

        if seg is None:
            # Empty transcript: entire video as one segment
            num_frames = int(end_time * self.fps)  # (end_time - 0) * 30
            num_frames = min(num_frames, total_frames)
            frames = [Image.open(frames_files[i]) for i in range(num_frames)]
            annotations = all_annotations[:num_frames]
            audio_segment = audio[:int(end_time * audio_sr)]
            transcript_text = ""
            print(f"{vid_id} (entire video): Frames={num_frames}, Annotations={len(annotations)}")
            assert len(frames) == len(annotations), "Mismatch in frames and annotations"
            return frames, audio_segment, transcript_text, annotations, [start_time, end_time], seg_num

        # Segment-specific logic
        transcript_text = seg['text']
        num_frames = int((end_time - start_time) * self.fps)  # (end_time[n] - start_time) * 30
        start_frame_idx = int(start_time * self.fps)
        end_frame_idx = min(start_frame_idx + num_frames, total_frames)

        if start_frame_idx >= total_frames:
            print(f"Warning: Segment {seg_num} of {vid_id} starts beyond video duration.")
            frames = []
            annotations = []
            audio_segment = np.array([])
        else:
            frames = [Image.open(frames_files[i]) for i in range(start_frame_idx, end_frame_idx)]
            annotations = all_annotations[start_frame_idx:end_frame_idx]
            start_sample = int(start_time * audio_sr)
            end_sample = min(int(end_time * audio_sr), len(audio))
            audio_segment = audio[start_sample:end_sample] if start_sample < len(audio) else np.array([])

        print(f"{vid_id}, Segment {seg_num}: Start={start_time}, End={end_time}, Frames={len(frames)}, Annotations={len(annotations)}")
        if frames:
            assert len(frames) == len(annotations), "Mismatch in frames and annotations"

        return frames, audio_segment, transcript_text, annotations, [start_time, end_time], seg_num


class MultiModalCollator:
    def __init__(self, frame_size=224, audio_sample_rate=16000, max_audio_len=16000):
        self.frame_size = frame_size
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_len = max_audio_len
        
        self.transform = transforms.Compose([
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, batch):
        frames_batch = []
        audio_batch = []
        transcript_batch = []
        annotation_batch = []
        timestep_batch = []
        frames_count = []
        segment_nums = []
        
        for frames, audio, transcript, annotations, timestep, seg_num in batch:
            if not frames:  # Handle empty frame list
                print(f"Warning: No frames for segment {seg_num}, using dummy frame.")
                processed_frames = [torch.zeros(3, self.frame_size, self.frame_size)]  # Dummy frame
            else:
                processed_frames = [self.transform(frame) for frame in frames]
            
            actual_frame_count = len(processed_frames)
            frames_count.append(actual_frame_count)
            segment_nums.append(seg_num)
            
            if len(annotations) != actual_frame_count:
                print(f"Warning: Annotations ({len(annotations)}) != Frames ({actual_frame_count}), adjusting...")
                annotations = annotations[:actual_frame_count] if annotations else ["dummy_annotation"]
            
            if len(audio) > self.max_audio_len:
                audio = audio[:self.max_audio_len]
            elif len(audio) == 0:
                audio = np.zeros(self.max_audio_len)  # Dummy audio if empty
            else:
                audio = np.pad(audio, (0, max(0, self.max_audio_len - len(audio))))
            
            frames_batch.append(processed_frames)
            audio_batch.append(torch.tensor(audio, dtype=torch.float32))
            transcript_batch.append(transcript)
            annotation_batch.append(annotations)
            timestep_batch.append(timestep)
        
        max_frames_in_batch = max(frames_count)
        padded_frames_batch = []
        
        for frames in frames_batch:
            if len(frames) < max_frames_in_batch:
                padding = [torch.zeros(3, self.frame_size, self.frame_size)
                          for _ in range(max_frames_in_batch - len(frames))]
                frames.extend(padding)
            padded_frames_batch.append(torch.stack(frames))
        
        return {
            'frames': torch.stack(padded_frames_batch),
            'audio': torch.stack(audio_batch),
            'transcript': transcript_batch,
            'annotations': annotation_batch,
            'timestep': timestep_batch,
            'frames_count': frames_count,
            'segment_nums': segment_nums
        }


class EmbeddingExtractor(nn.Module):
    def __init__(self):
        super(EmbeddingExtractor, self).__init__()
        self.frame_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.audio_device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        self.text_device = torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cpu")
        
        print("Loading feature extractors...")
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0).to(self.frame_device)
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.audio_device)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.roberta = RobertaModel.from_pretrained("roberta-base").to(self.text_device)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        for model in [self.swin, self.hubert, self.roberta]:
            for param in model.parameters():
                param.requires_grad = False

    def extract_video_embeddings(self, frames):
        frames = frames.to(self.frame_device)
        batch_size, num_frames = frames.shape[:2]
        frames_flat = frames.view(-1, 3, 224, 224)
        
        video_embeddings_list = []
        with torch.no_grad():
            for i in range(0, batch_size * num_frames, 64):
                end_idx = min(i + 64, batch_size * num_frames)
                batch_frames = frames_flat[i:end_idx]
                if batch_frames.shape[0] > 0:
                    emb = self.swin(batch_frames)
                    video_embeddings_list.append(emb)
        
        if video_embeddings_list:
            video_embeddings = torch.cat(video_embeddings_list, dim=0)
            video_embeddings = video_embeddings.view(batch_size, num_frames, -1)
            return video_embeddings
        raise ValueError("No video embeddings processed")

    def extract_audio_embeddings(self, audio_waveforms):
        audio_waveforms = audio_waveforms.to(self.audio_device)
        batch_size = audio_waveforms.shape[0]
        audio_embeddings_list = []
        
        for i in range(batch_size):
            try:
                audio_features = self.feature_extractor(
                    audio_waveforms[i].cpu().numpy().reshape(1, -1),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                audio_input_values = audio_features.input_values.to(self.audio_device)
                with torch.no_grad():
                    audio_outputs = self.hubert(audio_input_values)
                    audio_emb = audio_outputs.last_hidden_state.mean(dim=1)
                    audio_embeddings_list.append(audio_emb)
            except Exception as e:
                print(f"Error processing audio sample {i}: {e}")
                audio_embeddings_list.append(torch.zeros(1, 768, device=self.audio_device))
        
        audio_embeddings = torch.cat(audio_embeddings_list, dim=0)
        return audio_embeddings

    def extract_text_embeddings(self, transcripts):
        try:
            inputs = self.tokenizer(transcripts, return_tensors="pt", padding=True, truncation=True).to(self.text_device)
            with torch.no_grad():
                text_outputs = self.roberta(**inputs)
                text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
            return text_embeddings
        except Exception as e:
            print(f"Error processing text: {e}")
            return torch.zeros((len(transcripts), 768), device=self.text_device)


def save_embeddings(extractor, dataset, dataloader, output_dir, split_name):
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    
    print(f"Processing {split_name} split...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {split_name} embeddings")):
        with torch.no_grad():
            video_embs = extractor.extract_video_embeddings(batch['frames'])
            audio_embs = extractor.extract_audio_embeddings(batch['audio'])
            text_embs = extractor.extract_text_embeddings(batch['transcript'])
        
        batch_size = len(batch['transcript'])
        for i in range(batch_size):
            # Ensure index is within bounds
            seg_idx = batch_idx * dataloader.batch_size + i
            if seg_idx >= len(dataset.segments):
                break  # Skip if index exceeds dataset size
            vid_id, seg, start_time, end_time, seg_num = dataset.segments[seg_idx]
            
            video_dir = os.path.join(split_output_dir, vid_id, str(seg_num))
            os.makedirs(video_dir, exist_ok=True)
            
            torch.save(video_embs[i].cpu(), os.path.join(video_dir, "frames_embedding.pt"))
            torch.save(audio_embs[i].cpu(), os.path.join(video_dir, "audio_embedding.pt"))
            torch.save(text_embs[i].cpu(), os.path.join(video_dir, "transcript_embedding.pt"))
            
            with open(os.path.join(video_dir, "annotations.txt"), "w") as f:
                f.write("\n".join(batch['annotations'][i]))
            
            timestep = batch['timestep'][i]
            with open(os.path.join(video_dir, "timestep.txt"), "w") as f:
                f.write(f"start: {timestep[0]}\nend: {timestep[1]}")


def main():
    train_root = "/home/lownish/ABAW_FINAL/train_val/Train"
    val_root = "/home/lownish/ABAW_FINAL/train_val/Validation"
    output_root = "/home/lownish/ABAW_FINAL/Embeddings/Transformers"
    
    for root in [train_root, val_root]:
        for subdir in ["frames", "audio", "transcript", "annotations"]:
            assert os.path.exists(os.path.join(root, subdir)), f"Directory {os.path.join(root, subdir)} does not exist"
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if num_gpus < 3:
        print("Warning: Need at least 3 GPUs for this configuration. Falling back to CPU if fewer are available.")
    
    extractor = EmbeddingExtractor()
    
    train_dataset = MultiVideoDataLoader(
        frames_root=os.path.join(train_root, "frames"),
        audio_root=os.path.join(train_root, "audio"),
        transcript_root=os.path.join(train_root, "transcript"),
        annotation_root=os.path.join(train_root, "annotations"),
        fps=30
    )
    train_collator = MultiModalCollator()
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, 
                              collate_fn=train_collator, num_workers=4, pin_memory=True)
    save_embeddings(extractor, train_dataset, train_loader, output_root, "train")
    
    val_dataset = MultiVideoDataLoader(
        frames_root=os.path.join(val_root, "frames"),
        audio_root=os.path.join(val_root, "audio"),
        transcript_root=os.path.join(val_root, "transcript"),
        annotation_root=os.path.join(val_root, "annotations"),
        fps=30
    )
    val_collator = MultiModalCollator()
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                            collate_fn=val_collator, num_workers=4, pin_memory=True)
    save_embeddings(extractor, val_dataset, val_loader, output_root, "val")
    
    print("Embedding extraction completed for both train and validation splits")


if __name__ == "__main__":
    main()