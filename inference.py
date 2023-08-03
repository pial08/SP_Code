from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import argparse


def main():
    parser = argparse.ArgumentParser(description="Interact with trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the trained model files")

    args = parser.parse_args()
    model_dir = args.model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)

    generator = TextGenerationPipeline(
        model=model, tokenizer=tokenizer)

    while True:
        # Take instruction and input from the user
        instruction = input("Enter the instruction (or 'exit' to quit): ")
        if instruction.lower() == 'exit':
            break

        user_input = input("Enter the input context (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        if user_input.strip() != "":
            prompt = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n" + instruction +
                "\n\n### Input:\n" + user_input + "\n\n### Response:"
            )
        else:
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n" + instruction + "\n\n### Response:"
            )

        responses = generator(prompt, max_length=512, do_sample=False)

        # Display the response
        print("\nResponse:")
        print(responses[0]['generated_text'])

if __name__ == "__main__":
    main()