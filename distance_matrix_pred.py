import argparse
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List
import time
import os
from Bio import SeqIO
from tqdm import tqdm
import sys

# Global device setting
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RNA Distance Map Prediction')
    parser.add_argument('--model', choices=['rinalmo', 'birnabert'], default='birnabert',
                        help='Model to use for inference (default: birnabert)')
    parser.add_argument('--fasta_path', required=True,
                        help='Path to input FASTA file containing RNA sequences')
    parser.add_argument('--output_path', required=True,
                        help='Path for output pickle file')
    parser.add_argument('--embeddings_dir', default=None,
                        help='Directory to save individual embedding files (optional)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')

    return parser.parse_args()

def validate_inputs(args):
    """Validate input files and arguments"""
    
    global model_path
    if args.model == 'rinalmo':
        model_path = "trained_models/rinalmo.pth"
    elif args.model == 'birnabert':
        model_path = "trained_models/birnabert.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(args.fasta_path):
        print(f"Error: FASTA file not found: {args.fasta_path}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def read_fasta(fasta_path):
    """Read sequences from FASTA file"""
    sequences = []
    seq_ids = []

    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            # Convert to RNA (T -> U) for RiNALMo, keep as-is for BiRNABert
            sequences.append(str(record.seq).upper())
            seq_ids.append(record.id)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)

    return sequences, seq_ids

def load_rinalmo_model():
    """Load RiNALMo tokenizer and base model"""
    try:
        from multimolecule import RnaTokenizer, RiNALMoModel
        import multimolecule  # Required for model registration

        tokenizer = RnaTokenizer.from_pretrained('multimolecule/rinalmo')
        base_model = RiNALMoModel.from_pretrained('multimolecule/rinalmo')

        return tokenizer, base_model
    except ImportError as e:
        print(f"Error importing RiNALMo dependencies: {e}")
        print("Please install: pip install multimolecule transformers torch biopython")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading RiNALMo model: {e}")
        sys.exit(1)

def load_birnabert_model():
    """Load BiRNABert tokenizer and base model"""
    try:
        import multimolecule
        import transformers
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")
        config = transformers.BertConfig.from_pretrained("buetnlpbio/birna-bert")
        base_model = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",
                                                          config=config,
                                                          trust_remote_code=True)
        base_model.cls = torch.nn.Identity()

        return tokenizer, base_model
    except ImportError as e:
        print(f"Error importing BiRNABert dependencies: {e}")
        print("Please install: pip install multimolecule transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading BiRNABert model: {e}")
        sys.exit(1)

def load_model_components(model_type):
    """Load tokenizer and base model based on model type"""
    if model_type == 'rinalmo':
        return load_rinalmo_model()
    elif model_type == 'birnabert':
        return load_birnabert_model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut  # Residual connection
        return self.relu(x)

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale  # Recalibrate features

class DistanceMapPredictor(nn.Module):
    def __init__(self, base_model, model_type):
        super(DistanceMapPredictor, self).__init__()
        self.bert = base_model
        self.model_type = model_type
        self.hidden_size = self.bert.config.hidden_size  # Dynamically fetch hidden size

        # Adjust bottleneck projection to dynamically match input size
        self.projection = nn.Conv2d(2 * self.hidden_size, 512, kernel_size=1)

        # Enhanced convolutional layers with batch normalization, residual connections, and attention
        self.conv_layers = nn.Sequential(
            ResidualBlock(512, 512, dilation=1),
            ResidualBlock(512, 256, dilation=2),  # Multi-scale context with dilation
            SqueezeExcitationBlock(256),         # Channel attention
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        # Initialize weights
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids)

        # Handle different model outputs
        if self.model_type == 'rinalmo':
            embeddings = outputs.last_hidden_state  # RiNALMo uses last_hidden_state
        elif self.model_type == 'birnabert':
            embeddings = outputs.logits  # BiRNABert uses logits

        max_len = embeddings.size(1)

        # Pairwise concatenation
        concat_embeddings = torch.cat(
            [
                embeddings.unsqueeze(2).expand(-1, -1, max_len, -1),
                embeddings.unsqueeze(1).expand(-1, max_len, -1, -1),
            ],
            dim=-1,
        )  # Shape: (batch, max_len, max_len, 2 * hidden_size)

        # Permute to match convolution input expectations
        concat_embeddings = concat_embeddings.permute(0, 3, 1, 2)  # Shape: (batch, 2 * hidden_size, max_len, max_len)

        # Reduce dimensionality with bottleneck
        concat_embeddings = self.projection(concat_embeddings)

        # Apply convolutional layers
        output_distance_map = self.conv_layers(concat_embeddings).squeeze(1)  # Shape: (batch, max_len, max_len)

        # Mask upper triangle for valid lengths
        upper_tri_mask = torch.triu(torch.ones(max_len, max_len, device=DEVICE), diagonal=1)  # Upper triangle mask
        distance_map_mask_list = []
        for l in lengths:
            distance_map_mask = torch.zeros(max_len, max_len, device=DEVICE)
            distance_map_mask[1 : l + 1, 1 : l + 1] = 1
            distance_map_mask_list.append(distance_map_mask)

        distance_map_masks = torch.stack(distance_map_mask_list)  # Shape: (batch, max_len, max_len)
        valid_upper_tri_mask = distance_map_masks * upper_tri_mask  # Combine masks

        # Extract upper triangle and enforce symmetry
        upper_triangle = output_distance_map * valid_upper_tri_mask
        symmetric_map = upper_triangle + upper_triangle.transpose(-1, -2)

        diag_indices = torch.arange(max_len, device=DEVICE)
        symmetric_map[:, diag_indices, diag_indices] = 0.0

        # Return symmetric map (loss will propagate naturally)
        return symmetric_map, embeddings

class RNADatasetInference(Dataset):
    """Dataset for inference - contains only sequences, no distance maps"""
    def __init__(self, sequences: List[str], tokenizer, model_type):
        self.model_type = model_type
        self.tokenizer = tokenizer

        if model_type == 'rinalmo':
            # RiNALMo expects RNA sequences (T -> U conversion)
            self.sequences = [seq.replace('T', 'U') for seq in sequences]
        elif model_type == 'birnabert':
            # BiRNABert expects space-separated sequences
            self.sequences = [' '.join(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        input_ids = self.tokenizer(sequence, return_tensors="pt").input_ids
        return input_ids



def run_inference(args):
    """Main inference function"""

    # Validate inputs
    validate_inputs(args)

    # Load model components
    tokenizer, base_model = load_model_components(args.model)
    base_model.to(DEVICE)

    # Create distance map predictor
    model = DistanceMapPredictor(base_model, args.model).to(DEVICE)

    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    except Exception as e:
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        except Exception as e2:
            print(f"Failed to load model weights: {e2}")
            sys.exit(1)

    model.eval()

    # Read sequence from FASTA (modified to handle single sequence)
    sequences, seq_ids = read_fasta(args.fasta_path)
    sequence = sequences[0]  # Get the single sequence
    seq_id = seq_ids[0]

    # Create dataset and dataloader (simplified for single sequence)
    dataset = RNADatasetInference([sequence], tokenizer, args.model)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run inference
    for data in dataloader:  # No need for tqdm with single sequence
        input_ids = data[0].to(DEVICE)
        lengths = [len(input_ids[0]) - 2]  # Subtract CLS and SEP tokens
        
        with torch.no_grad():
            output_distance_map, embeddings = model(input_ids, lengths)
            
            # Extract and trim the distance matrix
            distance_map = output_distance_map.cpu().numpy()[0]  # Get first (and only) batch item
            trimmed_distance_map = distance_map[1:-1, 1:-1]  # Trim first and last rows/columns
            
            # Save only the distance matrix
            with open(args.output_path, 'wb') as f:
                pickle.dump(trimmed_distance_map, f)
            
            # Save embeddings if requested
            if args.embeddings_dir:
                os.makedirs(args.embeddings_dir, exist_ok=True)
                embed_path = os.path.join(args.embeddings_dir, f'{seq_id}.pkl')
                with open(embed_path, 'wb') as file:
                    pickle.dump(embeddings.squeeze(0).cpu().numpy(), file)


def main():
    """Main function"""
    args = parse_arguments()
    run_inference(args)

if __name__ == "__main__":
    main()
