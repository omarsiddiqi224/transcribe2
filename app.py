import subprocess
#import streamlit as st
import gradio as gr
from transformers import pipeline
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
#import whisper
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device="cuda")
#pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
 

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
 
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
 
    #transcribed_text += transcriber({"sampling_rate": sr, "raw": stream})["text"] + " "
 
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()
        

def summarize(transcribed_text):
    print("before prompt")
    prompt_template = """Based on the provided conversation, your task is to summarize the key findings and derive insights. Please create a thorough summary note under the heading 'SUMMARY KEY NOTES' and include bullet points about the key items discussed.
    Ensure that your summary is clear and informative, conveying all necessary information (include how the caller was feeling, meaning sentiment analysis). Focus on the main points mentioned in the conversation, such as Claims, Benefits, Providers, and other relevant topics. Additionally, create an action items/to-do list based on the insights and findings from the conversation.
    The main points to look for in a conversation are: Claims, Correspondence and Documents, Eligibility & Benefits, Financials, Grievance & Appeal, Letters, Manage Language, Accumulators, CGHP & Spending Account Buy Up, Group Search, Member Enrollment & Billing, Manage ID Cards, Member Limited Liability, Member Maintenance, Other Health insurance (COB), Provider Lookup, Search/ Update UM Authorization, Prefix and Inter Plan Search, Promised Action Search Inventory.
    Please note that while you can look for other points, it is important to prioritize the main points mentioned above.

    The following is a summary example from a call which discussed Claims, Benefits, and the Provider. Use this structure to build the summary and action items:
    SUMMARY KEY NOTE:
        Highlighting Key Findings:

        1. CLAIM:
            -Caller's claim for 'Member' was denied because it was for an uncovered service.
            -Caller expressed frustration regarding the claim denial.
        2. BENEFITS:
            -Caller inquired about the coverage of 'ultrasound' and was informed that it is covered, but the service was provided at an out-of-network facility, leading to the denial of benefits.
            -Caller expressed confusion about the network status of 'Dr. David'
        3. PROVIDER:
            -'Dr. David had switched networks in the last month, making him out of the coverage for the caller.
            -Caller expressed concern about 'Dr. David being in-network for the past two years.

        ACTION ITEMS:
        Based on the summarized coversation, the following immediate actions are suggeested:
        
        1. Address the caller's claim denial for the uncovered service by providing a clear explanation or guiding them on how to address the issue.
        2. Clarify the network status of 'Dr. David'

        """

    #clean_text1 = prompt_template.strip()
    #clean_text1 = prompt_template

    print("before message")
    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {"role": "user", "content": transcribed_text},
    ]

    print("before tokenizer")
    
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("before pipe output")
    
    outputs = pipe(prompt, max_new_tokens=2000, do_sample=True, temperature=0.5, top_k=50, top_p=0.95)
    
    print("before generated text (all)")
    
    my_string = outputs[0]["generated_text"]
    
    print("before answer (split)")
    
    summarizing = my_string.split("<|assistant|>",1)[1]
    
    print("before return")
    
    return summarizing                                           

transcripts = ""
def transcribe2(audio_file):
    if audio_file:
        head, tail = os.path.split(audio_file)
        path = head
 
        if tail[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
            tail = "audio.wav"
  
        subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
        tail = "audio.wav"
        print("before diarize")
        running = subprocess.run([f"python diarize.py -a {tail}"], shell=True, capture_output=True, text=True)
        print("after diarize")
        text = read_file_content('audio.txt')
        print("after reading")
        fixed = text.strip()
        summarized = summarize(fixed)
        #print("after summary")
        global transcripts
        transcripts = text
        return(text, summarized)
 

    
with gr.Blocks() as demo:
 
    gr.Interface(
        transcribe,
        ["state", gr.Audio(sources=["microphone"], streaming=True)],
        ["state", "text"],
        live=True,
        title="Real-Time Transcription",
    )
 
    gr.Interface(transcribe2,
        inputs=[
            gr.Audio(sources ='upload', type='filepath', label='Audio File'),
           
            ],
        outputs=["text", "text"],
        title="Transcribe and Summarize Files" 
    )

    gr.Interface(summarize,
            inputs="text", 
            outputs="text", 
            title="Summarize Transcription")
 
 

#demo.launch(share=True)
demo.queue().launch(debug=True, share=True, inline=False)
