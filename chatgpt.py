
import os
import cv2
import time
import glob
import base64
import numpy as np

import openai
from config import candidate_keys

global_index = 0
openai.api_key = candidate_keys[global_index]

# request for one time
def func_get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print ('Error:', e) # change key to avoid RPD
        global global_index 
        global_index = (global_index + 1) % len(candidate_keys)
        print (f'========== key index: {global_index} ==========')
        openai.api_key = candidate_keys[global_index]
        return ''

# request for three times
def get_completion(prompt, model, maxtry=5):
    response = ''
    try_number = 0
    while len(response) == 0:
        try_number += 1
        if try_number == maxtry: 
            print (f'fail for {maxtry} times')
            break
        response = func_get_completion(prompt, model)
    return response

# polish chatgpt's outputs
def func_postprocess_chatgpt(response):
    response = response.strip()
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    response = response.strip()
    response = response.replace('\n', '')
    response = response.strip()
    return response


# ---------------------------------------------------------------------
## convert image/video into GPT4 support version
def func_image_to_base64(image_path, grey_flag=False): # support more types
    image = cv2.imread(image_path)
    if grey_flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return func_opencv_to_base64(image)

def func_opencv_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# deal with text
def func_nyp_to_text(npy_path):
    text = np.load(npy_path).tolist()
    text = text.strip()
    text = text.replace('\n', '') # remove \n
    text = text.replace('\t', '') # remove \t
    text = text.strip()
    return text

# support two types: (video) or (frames in dir)
def sample_frames_from_video(video_path, samplenum=3):
    if os.path.isdir(video_path): # already sampled video, frame store in video_path
        select_frames = sorted(glob.glob(video_path + '/*'))
        select_frames = select_frames[:samplenum]
        select_frames = [cv2.imread(item) for item in select_frames]
    else: # original video
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret == False: break
            frames.append(frame)
        cap.release()
        
        # return frames
        while len(frames) < samplenum:
            frames.append(frames[-1])
        
        tgt_length = int(len(frames)/samplenum)*samplenum
        frames = frames[:tgt_length]
        indices = np.arange(0, len(frames), int(len(frames) / samplenum)).astype(int).tolist()
        print ('sample indexes: ', indices)
        assert len(indices) == samplenum
        select_frames = [frames[index] for index in indices]
    assert len(select_frames) == samplenum, 'actual sampled frames is ont equal to tgt samplenum'
    return select_frames


# ---------------------------------------------------------------------
## Emotion
# ---------------------------------------------------------------------
# 20 images per time
def get_image_emotion_batch(image_paths, candidate_list, sleeptime=0, grey_flag=False, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a facial expression classification expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                              For each image, please sort the provided categories from high to low according to the top 5 similarity with the input image. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path, grey_flag),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_evoke_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a emotion recognition expert. We provide {len(image_paths)} images. \
                              Please recognize sentiments evoked by these images (i.e., guess how viewer might emotionally feel after seeing these images.) \
                              If there is a person in the image, ignore that person's identity. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              Here are the optional categories: {candidate_list}. If there is a person in the image, ignore that person's identity. \
                              The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_micro_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a micro-expression recognition expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              The expression may not be obvious, please pay attention to the details of the face. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

# # 20 images per time
# def get_audio_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
#     prompt = [
#                 {
#                     "type":  "text", 
#                     "text": f"Please play the role of a audio expression classification expert. We provide {len(image_paths)} audios, each with an image of Mel spectrogram. \
#                               Please ignore the speaker's identity and recognize the speaker's expression from the provided Mel spectrogram. \
#                               For each sample, please sort the provided categories from high to low according to the top 5 similarity with the input. \
#                               Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each audio."
#                 }
#             ]
#     for ii, image_path in enumerate(image_paths):
#         prompt.append(
#             {
#                 "type":  f"audio-{ii+1}",
#                 "image": func_image_to_base64(image_path),
#             }
#         )
#     print (prompt[0]['text']) # debug
#     for item in prompt: print (item['type']) # debug
#     time.sleep(sleeptime)
#     response = get_completion(prompt, model)
#     response = func_postprocess_chatgpt(response)
#     print (response)
#     return response


def get_text_emotion_batch(npy_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a textual emotion classification expert. We provide {len(npy_paths)} texts. \
                              Please recognize the speaker's expression from the provided text. \
                              For each text, please sort the provided categories from high to low according to the top 5 similarity with the input. \
                              Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each text."
                }
            ]
    for ii, npy_path in enumerate(npy_paths):
        prompt.append(
            {
                "type":  f"text",
                "text": f"{func_nyp_to_text(npy_path)}",
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# 20 images per time
def get_video_emotion_batch(video_paths, candidate_list, sleeptime=0, samplenum=3, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with {samplenum} temporally uniformly sampled frames. Please ignore the speaker's identity and focus on their facial expression. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        video_frames = sample_frames_from_video(video_path, samplenum)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_multi_emotion_batch(video_paths, candidate_list, sleeptime=0, samplenum=3, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with the speaker's content and {samplenum} temporally uniformly sampled frames.\
                              Please ignore the speaker's identity and focus on their emotions. Please ignore the speaker's identity and focus on their emotions. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on their emotions. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        # convert video_path to text path
        split_paths = video_path.split('/')
        split_paths[-2] = 'text'
        split_paths[-1] = split_paths[-1].rsplit('.', 1)[0] + '.npy'
        text_path = "/".join(split_paths)
        assert os.path.exists(text_path)
        prompt.append(
                {
                    "type": "text",
                    "text": f"{func_nyp_to_text(text_path)}",
                },
        )

        # read frames
        video_frames = sample_frames_from_video(video_path, samplenum=3)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
       
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response
