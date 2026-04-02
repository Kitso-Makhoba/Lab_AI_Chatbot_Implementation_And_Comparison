from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct"
)

def get_llm_response(user_input: str) -> str:
    prompt = f"User: {user_input}\nAssistant:"

    response = chatbot(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    output = response[0]["generated_text"].split("Assistant:")[-1]
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