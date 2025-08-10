#!/usr/bin/env python3
"""
HuggingFace Transformers GPT-2 Demo 🚀

This script demonstrates:
1. Loading a pre-trained GPT-2 model
2. Generating text from custom prompts
3. Visualizing attention weights

Requirements:
pip install transformers torch matplotlib seaborn numpy
"""

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def setup_model():
    """Load and setup the GPT-2 model."""
    print("=== Loading Pre-trained GPT-2 Model ===")
    
    # You can choose different model sizes:
    # - 'gpt2' (124M parameters) - fastest
    # - 'gpt2-medium' (355M parameters)
    # - 'gpt2-large' (774M parameters)
    # - 'gpt2-xl' (1.5B parameters) - slowest
    
    model_name = 'gpt2'  # Start with the smallest for faster loading
    
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ Model loaded: {model_name}")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✓ Model moved to: {device}")
    print()
    
    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt: str, max_length: int = 100, 
                  temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9, 
                  do_sample: bool = True):
    """
    Generate text from a given prompt using GPT-2.
    
    Args:
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Controls randomness (lower = more deterministic)
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling or greedy decoding
    
    Returns:
        Generated text and attention weights
    """
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=True
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return generated_text, outputs.attentions

def demo_text_generation(model, tokenizer, device):
    """Demonstrate text generation with different prompts and parameters."""
    print("=== Text Generation Examples ===")
    
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a magical forest",
        "The best way to learn programming is",
        "In the year 2050, humanity will"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: {prompt}")
        
        # Generate with different temperatures
        for temp in [0.7, 1.0, 1.3]:
            try:
                generated_text, attentions = generate_text(
                    model, tokenizer, device,
                    prompt, 
                    max_length=len(prompt.split()) + 20,  # Add ~20 words
                    temperature=temp,
                    do_sample=True
                )
                
                print(f"  Temperature {temp}: {generated_text}")
            except Exception as e:
                print(f"  Temperature {temp}: Error - {e}")
        
        print()
    
    return prompts

def visualize_attention_weights(attentions: Tuple[torch.Tensor], 
                               tokens: List[str], 
                               layer_idx: int = 0, 
                               head_idx: int = 0):
    """
    Visualize attention weights for a specific layer and head.
    
    Args:
        attentions: Tuple of attention tensors from model output
        tokens: List of token strings
        layer_idx: Index of the transformer layer to visualize
        head_idx: Index of the attention head to visualize
    """
    if not attentions:
        print("No attention weights available")
        return
    
    try:
        # Get attention weights for the specified layer and head
        # attentions[layer_idx] has shape (batch_size, num_heads, seq_len, seq_len)
        attention_tensor = attentions[layer_idx]
        
        # Debug: print the shape of the attention tensor
        print(f"Attention tensor shape for layer {layer_idx}: {attention_tensor.shape}")
        
        # Ensure we have the right dimensions
        if len(attention_tensor.shape) == 4:
            # Standard format: (batch_size, num_heads, seq_len, seq_len)
            attention = attention_tensor[0, head_idx].cpu().numpy()
        elif len(attention_tensor.shape) == 3:
            # Alternative format: (num_heads, seq_len, seq_len)
            attention = attention_tensor[head_idx].cpu().numpy()
        else:
            print(f"Unexpected attention tensor shape: {attention_tensor.shape}")
            return
        
        # Ensure tokens list matches the attention matrix dimensions
        if len(tokens) != attention.shape[0]:
            print(f"Warning: Number of tokens ({len(tokens)}) doesn't match attention matrix size ({attention.shape[0]})")
            # Truncate or pad tokens to match
            if len(tokens) > attention.shape[0]:
                tokens = tokens[:attention.shape[0]]
            else:
                # Pad with empty strings if needed
                tokens.extend([''] * (attention.shape[0] - len(tokens)))
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention, 
            xticklabels=tokens, 
            yticklabels=tokens,
            cmap='Blues',
            annot=False,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Attention weight statistics for Layer {layer_idx}, Head {head_idx}:")
        print(f"  Mean: {attention.mean():.4f}")
        print(f"  Std: {attention.std():.4f}")
        print(f"  Min: {attention.min():.4f}")
        print(f"  Max: {attention.max():.4f}")
        
    except Exception as e:
        print(f"Error in visualize_attention_weights: {e}")
        print(f"Attention tensor type: {type(attentions[layer_idx])}")
        if hasattr(attentions[layer_idx], 'shape'):
            print(f"Attention tensor shape: {attentions[layer_idx].shape}")
        else:
            print(f"Attention tensor has no shape attribute")

def demo_attention_visualization(model, tokenizer, device):
    """Demonstrate attention weight visualization."""
    print("=== Attention Weight Visualization ===")
    
    # Generate text with attention tracking
    prompt = "The artificial intelligence revolution began"
    print(f"Generating text from prompt: '{prompt}'")
    
    try:
        generated_text, attentions = generate_text(
            model, tokenizer, device,
            prompt, 
            max_length=len(prompt.split()) + 15,
            temperature=0.8,
            do_sample=True
        )
        
        print(f"Generated text: {generated_text}")
        print(f"Number of layers: {len(attentions)}")
        
        # Debug attention tensor structure
        if attentions:
            print(f"Attention type: {type(attentions)}")
            print(f"First layer type: {type(attentions[0])}")
            if hasattr(attentions[0], 'shape'):
                print(f"First layer shape: {attentions[0].shape}")
            else:
                print(f"First layer has no shape attribute")
        else:
            print("No attention weights available")
        
        # Get tokens for visualization
        tokens = tokenizer.tokenize(generated_text)
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print()
        
        if attentions:
            # Visualize first layer, first head
            print("1. Single attention head visualization:")
            visualize_attention_weights(attentions, tokens, layer_idx=0, head_idx=0)
            
            # Visualize multiple heads from the same layer
            print("\n2. Multiple attention heads visualization:")
            
            # Create subplot grid for multiple heads
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.ravel()
            
            for i in range(4):
                try:
                    attention_tensor = attentions[0]
                    if len(attention_tensor.shape) == 4:
                        attention = attention_tensor[0, i].cpu().numpy()
                    elif len(attention_tensor.shape) == 3:
                        attention = attention_tensor[i].cpu().numpy()
                    else:
                        print(f"Unexpected attention tensor shape: {attention_tensor.shape}")
                        continue
                    
                    sns.heatmap(
                        attention,
                        xticklabels=tokens,
                        yticklabels=tokens,
                        cmap='Blues',
                        annot=False,
                        ax=axes[i],
                        cbar_kws={'label': 'Attention Weight'}
                    )
                    
                    axes[i].set_title(f'Head {i}')
                    axes[i].set_xlabel('Key Tokens')
                    axes[i].set_ylabel('Query Tokens')
                    axes[i].tick_params(axis='x', rotation=45, ha='right')
                except Exception as e:
                    print(f"Error visualizing head {i}: {e}")
                    # Create empty plot for failed head
                    axes[i].text(0.5, 0.5, f'Head {i}\nError', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Head {i} (Error)')
            
            plt.suptitle('Attention Weights for Multiple Heads - Layer 0', fontsize=16)
            plt.tight_layout()
            plt.show()
            
            # Visualize attention from different layers
            print("\n3. Attention from different layers:")
            for layer_idx in [0, len(attentions)//2, len(attentions)-1]:
                print(f"\nLayer {layer_idx}:")
                visualize_attention_weights(attentions, tokens, layer_idx=layer_idx, head_idx=0)
        else:
            print("No attention weights available for visualization")
            
    except Exception as e:
        print(f"Error during attention visualization: {e}")

def compare_generation_strategies(model, tokenizer, device, prompt: str, max_length: int = 50):
    """Compare different text generation strategies."""
    print(f"=== Generation Strategy Comparison ===")
    print(f"Prompt: {prompt}\n")
    
    strategies = [
        {"name": "Greedy", "do_sample": False, "temperature": 1.0},
        {"name": "Temperature 0.5", "do_sample": True, "temperature": 0.5},
        {"name": "Temperature 1.0", "do_sample": True, "temperature": 1.0},
        {"name": "Temperature 1.5", "do_sample": True, "temperature": 1.5},
        {"name": "Top-k (k=10)", "do_sample": True, "temperature": 1.0, "top_k": 10},
        {"name": "Top-p (p=0.9)", "do_sample": True, "temperature": 1.0, "top_p": 0.9}
    ]
    
    for strategy in strategies:
        try:
            generated_text, _ = generate_text(
                model, tokenizer, device,
                prompt,
                max_length=len(prompt.split()) + max_length,
                **{k: v for k, v in strategy.items() if k != 'name'}
            )
            
            print(f"{strategy['name']}:")
            print(f"  {generated_text}\n")
            
        except Exception as e:
            print(f"{strategy['name']}: Error - {e}\n")

def main():
    """Main demo function."""
    print("🚀 HuggingFace Transformers GPT-2 Demo")
    print("=" * 50)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # Setup model
        model, tokenizer, device = setup_model()
        
        # Demo text generation
        demo_text_generation(model, tokenizer, device)
        
        # Demo attention visualization
        demo_attention_visualization(model, tokenizer, device)
        
        # Compare generation strategies
        test_prompt = "The future of technology lies in"
        compare_generation_strategies(model, tokenizer, device, test_prompt)
        
        print("=== Demo Completed Successfully! ===")
        print("\nNext steps:")
        print("1. Try different model sizes (gpt2-medium, gpt2-large, gpt2-xl)")
        print("2. Experiment with different prompts and parameters")
        print("3. Explore attention patterns across different layers")
        print("4. Build your own text generation applications")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
