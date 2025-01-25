from openai import OpenAI
import os
import json

# 更改当前文件的运行路径
os.chdir('/home/xtz')
print(os.getcwd())

# Set your OpenAI API key
api_key = "sk-3mLfSEhWcv00gR9tG5VBkvSUhiyaIYnPuWWhC6yKWEOadv2p"

# Define function to map max tokens to text length
def length_to_str(max_tokens):
    if max_tokens <= 10: return "short"
    else: return "middle"

# Define the function to create a prompt based on the requirements
def generate_dialogue(scenario, emotion, max_tokens):
    length = length_to_str(max_tokens)
    message = []
    message.append({"role": "system", "content": "You are a helpful assistant with human emotions and talking styles."})
    if emotion == "depressed":
        message.append({"role": "user", "content":f"In the context of {scenario}, say something in first-person or second-person "
            f"that expresses feelings of sadness, hopelessness, or emotional heaviness, "
            f"using the speaking style of depression. Imagine you are talking to someone close to you, "
            f"but feel overwhelmed or unable to fully express your emotions. "
            f"Do not provide any explanations, and just respond to the situation. "
            f"What you say should be in {length} length, with no more than {max_tokens} tokens."})
    else:
        message.append({"role": "user", "content":f"In the context of {scenario}, say something in first-person or second-person "
            f"that expresses your feeling, or using the speaking style of neutral, "
            f"as if you are talking to somebody. Do not write any explanations and just respond to the situation. "
            f"What you say should be in {length} length with no more than {max_tokens} tokens."})
    return message

# Function to sample generated text from GPT-4
def generate_emotional_text(scenarios, emotions, max_tokens_list):
    generated_texts = []

    client = OpenAI(
        base_url="https://api.wlai.vip/v1", 
        api_key=api_key
    )

    for emotion in emotions:
        print(f"Generating text for emotion: {emotion}")
        for scenario in scenarios:
            for max_tokens in max_tokens_list:
                # Generate prompt
                prompt = generate_dialogue(scenario, emotion, max_tokens)
                
                # Call GPT-4 to generate the response (assumes GPT-4 endpoint)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=prompt,
                        max_tokens=max_tokens,
                        n=1,
                        stop=None,
                        temperature=0.6
                    )

                    generated_text = response.choices[0].message.content.strip()

                    # wrap the generated text in a json object with other info
                    print(f"Generated text for {scenario}, {emotion}, {max_tokens}: {generated_text}")
                    generated_text = {
                        "scenario": scenario,
                        "emotion": emotion,
                        "max_tokens": max_tokens,
                        "text": generated_text.replace("...", ", ")
                    }
                    generated_texts.append(generated_text)

                except Exception as e:
                    print(f"Error generating text for {scenario}, {emotion}, {max_tokens}: {e}")

    return generated_texts

def text_generation():
    scenarios = [
        "arts", 
        "autos and vehicles", 
        "business", 
        "comedy", 
        "crime", 
        "education", 
        "entertainment", 
        "film and animation", 
        "gaming", 
        "health and fitness", 
        "history", 
        "howto and style", 
        "kids and family", 
        "leisure", 
        "music", 
        "news and politics", 
        "nonprofits and activism", 
        "people and blogs", 
        "pets and animals", 
        "religion and spirituality", 
        "science and technology", 
        "society and culture", 
        "sports", 
        "travel and events"
    ]

    emotions = ["depressed", "neutral"]
    max_tokens_list = [15, 30, 50]

    # Generate emotional text
    generated_texts = generate_emotional_text(scenarios, emotions, max_tokens_list)

    # Perform data engineering (optional, for cleaning or filtering generated texts)
    def data_engineering(generated_texts):
        return [obj for obj in generated_texts if len(obj['text'].split()) > 0]

    cleaned_texts = data_engineering(generated_texts)

    # Save the generated texts to a JSON file
    target_file_path = "/home/xtz/datasets/augment_data/generated_texts.json"
    with open(target_file_path, "w") as f:
        json.dump(cleaned_texts, f, indent=4)
    
    return cleaned_texts

if __name__ == "__main__":
    text_generation()