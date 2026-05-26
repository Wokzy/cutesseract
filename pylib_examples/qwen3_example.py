
import torch
from qwen3_patch import Qwen3ForCausalLM
from transformers import AutoTokenizer


def main():
    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2? Provide detailed explanation"
        }
    ]

    model_path = 'Qwen3-0.6B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3ForCausalLM.from_pretrained(model_path).cuda().eval()

    print(model.device)

    inp = tokenizer.apply_chat_template(messages)
    res = model.generate(input_ids=torch.tensor([inp['input_ids']], device='cuda'), max_new_tokens=64).cpu().tolist()

    print(tokenizer.decode(res[0]))


if __name__ == "__main__":
    main()

