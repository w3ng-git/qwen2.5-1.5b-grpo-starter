from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import readline
import os
tok = AutoTokenizer.from_pretrained("qwen2.5-1.5b")
# replace with your model
model = AutoModelForCausalLM.from_pretrained("trainOutput", device_map='auto') # 
model.eval()
model.half()
inputs = ''


system_message = "You are a helpful assistant."

chat_template = [
    {'role': 'system', 'content': system_message},
]

terminators = [
    tok.eos_token_id,
    tok.convert_tokens_to_ids("<|im_end|>"),
]
while True:
    print('<user>: (Enter your message, type "end" on a new line to finish)')
    lines = []
    while True:
        line = input()
        if line.lower() == 'end':
            break
        lines.append(line)
    
    if not lines:  # If no input was provided before 'end'
        break
        
    in_user = '\n'.join(lines)
    os.system('clear')
    chat_template.append({'role':'user', 'content': in_user})
    inputs = tok(tok.apply_chat_template(chat_template, add_generation_prompt=True,tokenize=False), return_tensors="pt")
    inputs = inputs.to('cuda')
    streamer = TextStreamer(tok)
    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    model_out = model.generate(**inputs,eos_token_id=terminators,
            do_sample=True,
            #use_cache=False,
            temperature=0.6,
            top_p=0.9,
            streamer=streamer, max_new_tokens=128)
    # 
    print('\nRaw token IDs:', model_out.tolist()[0][len(inputs[0]):-1])
    model_out = model_out.tolist()[0][len(inputs[0]):-1]
    model_out = tok.decode(model_out)
    print('modelout'+80*'-')
    print(model_out)
    print(80*'-')
    chat_template.append({'role': 'assistant', 'content': model_out})
    # 
    torch.cuda.empty_cache()
