from unsloth import FastLanguageModel

class LLMEngine:
    def __init__(self, model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit"):
        # 配置 Llama-3.2-1B-Instruct
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=16384, # 應付長篇 MIDI 序列
            load_in_4bit=True,
        )

        # 加入自定義 Token：<m1_0>...<m1_511>, <m4_0>...<m4_511>
        special_tokens = [f"<m1_{i}>" for i in range(512)] + [f"<m4_{i}>" for i in range(512)]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 凍結 Backbone，僅訓練 Adapter 與 LoRA Layers (使用 Unsloth 效率工具)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        
    def generate(self, input_ids):
        return self.model.generate(input_ids)
