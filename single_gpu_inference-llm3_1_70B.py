import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def inference(input_texts):
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B")

    ### NOTE: THIS MODEL REQUIRES AT LEAST 80GiB OF GPU MEMORY ###
    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    for i, text in enumerate(generated_texts):
        print(f"Question: {input_texts[i]}")
        print(f"Answer: {text}")
        print()

def main():
    input_texts = [
        "Explain the theory of relativity.",
        "What is the capital of France?",
        "How does quantum computing work?",
        "What are the benefits of machine learning?",
    ]

    inference(input_texts)

if __name__ == "__main__":
    main()
