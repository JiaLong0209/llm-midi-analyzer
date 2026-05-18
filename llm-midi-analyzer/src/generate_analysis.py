"""
generate_analysis.py — End-to-End MIDI Analysis Inference
=========================================================
Pipeline:
  Raw MIDI (.mid) → OctupleMIDI (8D) → Adapter/VQ-VAE → Llama 3.2 1B → Analysis Text

Supports both Direct and VQ-VAE modes.
Requires a trained checkpoint from train_adapter.py.
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from config import AdapterConfig
from models.adapters import AdapterFactory
from models.octuple import get_extractor


def load_inference_model(checkpoint_path, mode_override=None, device="cuda", vqvae_path=None, d_vq=None, musicbert_path=None, dtype_override="float16", lang="en"):
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    
    # Load config from checkpoint if available
    config_dict = state.get("config", {})
    acfg = AdapterConfig.from_dict(config_dict) if config_dict else AdapterConfig()
    
    # Override mode if provided by user or inferred from filename
    if mode_override:
        acfg.projection_mode = mode_override
    elif not config_dict:
        # Smart inference from filename if config is missing
        fname = os.path.basename(checkpoint_path).lower()
        if "vqvae" in fname:
            acfg.projection_mode = "vqvae"
        elif "musicbert" in fname:
            acfg.projection_mode = "musicbert"
        elif "direct" in fname:
            acfg.projection_mode = "direct"
        print(f"🕵️  Inferred projection_mode: {acfg.projection_mode}")
    
    # Apply VQ-VAE overrides if provided
    if vqvae_path:
        acfg.vqvae_checkpoint = vqvae_path
    if d_vq:
        acfg.d_vq = d_vq
    if musicbert_path:
        acfg.musicbert_model_path = musicbert_path
    if dtype_override:
        acfg.torch_dtype = dtype_override
    
    # If in vqvae mode but no vqvae_checkpoint is set in acfg, try a guess
    if acfg.projection_mode == "vqvae" and not acfg.vqvae_checkpoint:
        print("⚠️  Warning: vqvae mode detected but no vqvae_checkpoint provided. Ensure it is set in the saved config or pass --vqvae.")

    # 1. Build and load Adapter
    import gc
    adapter = AdapterFactory.build(acfg).to(device)
    adapter.load_state_dict(state["adapter"], strict=False)
    adapter.eval()
    
    # 2. Load LLM and local LoRA
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🤖 Loading LLM (Low Memory Mode): {acfg.llm_model_path}")
    
    # Resolve compute_dtype
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map.get(acfg.torch_dtype, torch.float16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(acfg.llm_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        acfg.llm_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    
    # Wrap with LoRA and load saved LoRA weights
    lora_config = LoraConfig(
        r=acfg.qlora_r,
        lora_alpha=acfg.qlora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora_config)
    
    if "lora" in state:
        llm.load_state_dict(state["lora"], strict=False)
        print("✅ LoRA weights loaded successfully.")
    else:
        print("⚠️  No LoRA weights found in checkpoint. Using base model + random LoRA (re-training recommended).")
    
    llm.eval()
    return tokenizer, llm, adapter, acfg


def run_inference(midi_path, tokenizer, llm, adapter, acfg, device="cuda", custom_seq_len=None, lang="en"):
    print(f"🎼 Processing MIDI: {midi_path}")
    
    # Use custom seq_len if provided (to see more of the song)
    effective_seq_len = custom_seq_len or acfg.seq_len
    
    # 1. Extract OctupleMIDI
    extractor = get_extractor("octuple_8d")
    tokens = extractor.extract(midi_path)
    if tokens is None or len(tokens) == 0:
        raise ValueError("MIDI file yielded no tokens.")
    
    # Slicing: Ensure we don't go out of bounds
    tokens_to_use = tokens[:effective_seq_len]
    adapter_device = next(adapter.parameters()).device
    x = torch.from_numpy(tokens_to_use).unsqueeze(0).to(adapter_device)
    
    # 2. Project through Adapter
    device_str = str(device).lower()
    is_cpu = "cpu" in device_str
    
    if is_cpu:
        import contextlib
        autocast_ctx = contextlib.nullcontext()
    else:
        autocast_ctx = torch.autocast(device_type="cuda")
    
    with torch.no_grad(), autocast_ctx:
        music_prefix = adapter(x)  # (1, L/4, 2048)
        
        # Build a more structured prompt with language constraints
        lang_instruction = ""
        if lang == "zh":
            lang_instruction = "Respond in Traditional Chinese (繁體中文). However, ALWAYS use formal English terms for chords (including Roman Numeral Analysis like 'i', 'V7', 'ii°'), scales (e.g. 'A Minor'), and musical techniques. "
        
        prompt = f"### Instruction: {lang_instruction}Analyze the following MIDI sequence. "
        prompt += "Identify the musical style, possible instruments, key, and overall atmosphere.\n"
        if hasattr(acfg, 'theory_context') and acfg.theory_context:
            prompt += f"### Context: {acfg.theory_context}\n"
        prompt += "### Analysis: "
        
        print(f"✍️  Generating analysis (Prefix tokens: {music_prefix.size(1)})...")
        
        tok = tokenizer(prompt, return_tensors="pt").to(device)
        text_embeds = llm.get_input_embeddings()(tok.input_ids)
        
        # Cast prefix to LLM dtype
        # Quantized models might return float32 for next(llm.parameters()).dtype, but expect float16 or bfloat16 for input embeddings.
        if "cpu" in device_str:
            llm_dtype = torch.float32
        else:
            llm_dtype = torch.float32
            for param in llm.parameters():
                if param.dtype in [torch.float16, torch.bfloat16]:
                    llm_dtype = param.dtype
                    break
            else:
                if getattr(llm, "is_loaded_in_4bit", False) or getattr(llm, "is_loaded_in_8bit", False):
                    llm_dtype = torch.float16
                else:
                    llm_dtype = next(llm.parameters()).dtype
                
        music_prefix = music_prefix.to(llm_dtype)
        text_embeds = text_embeds.to(llm_dtype)
        
        full_embeds = torch.cat([music_prefix, text_embeds], dim=1)
        
        out_ids = llm.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=acfg.gen_max_new_tokens,
            do_sample=acfg.gen_do_sample,
            temperature=acfg.gen_temperature,
            top_p=acfg.gen_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=acfg.gen_repetition_penalty,
        )
        
        # Decode and remove the prompt part
        full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return full_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI Analysis")
    parser.add_argument("--midi", type=str, required=True, help="Path to input .mid file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--mode", choices=["direct", "vqvae", "musicbert"], help="Override projection mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seq_len", type=int, default=None, help="Override sequence length (e.g. 512 or 1024)")
    parser.add_argument("--vqvae", type=str, help="Path to VQ-VAE checkpoint (required for vqvae mode if not in saved config)")
    parser.add_argument("--musicbert", type=str, help="Override MusicBERT model path")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16", help="Precision for models")
    parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Analysis language (en or zh)")
    parser.add_argument("--d_vq", type=int, help="VQ-VAE hidden dim (default: 256)")
    args = parser.parse_args()
    
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    try:
        tokenizer, llm, adapter, config = load_inference_model(
            args.checkpoint, 
            args.mode, 
            args.device, 
            vqvae_path=args.vqvae, 
            d_vq=args.d_vq,
            musicbert_path=args.musicbert,
            dtype_override=args.dtype,
            lang=args.lang
        )
        analysis = run_inference(args.midi, tokenizer, llm, adapter, config, args.device, custom_seq_len=args.seq_len, lang=args.lang)
        
        print("\n" + "="*50)
        print("📜 GENERATED ANALYSIS:")
        print("="*50)
        print(analysis)
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
