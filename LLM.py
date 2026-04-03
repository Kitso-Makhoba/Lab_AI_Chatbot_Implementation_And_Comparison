import os
os.environ["HF_TOKEN"] = "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"

from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium"
)

def get_llm_response(user_input: str) -> str:
    response = chatbot(
        user_input,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=50256
    )
    output = response[0]["generated_text"]
    # Remove the input prompt from the output
    if user_input in output:
        output = output.replace(user_input, "").strip()
    return output.strip()

if __name__ == "__main__":
    print("Modern AI Chatbot")
    print("Type 'quit' to stop.\n")

    while True:
        user = input("You: ")

        if user.lower() == "quit":
            print("Bot: Goodbye!")
            break

        print("Bot:", get_llm_response(user))
