# SEEDTEST
audio_offsets = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
fps = 1
TESTS=16
p="""Santa in sleight flying with christmas reindeers, detailed painting by greg rutkowski, (((trending on ArtStation))), trending on CGSociety, volumetric lighting, dramatic lighting"""
height= 896
width=512
name="santa-seedtest"
audio_filepath='SnapSpot26.mp3'

prompts=[]
for i in range(TESTS+1):
  prompts.append(p)

import random
seeds=[]
for i in range(TESTS+1):
  n=random.randint(0,100000)
  seeds.append(int(n))


# Convert seconds to frames
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]
 
 
video_path = pipeline.walk(
    prompts,
    seeds,
    num_interpolation_steps=num_interpolation_steps,
    height = height,                            # use multiples of 64
    width = width,                             # use multiples of 64
    negative_prompt='amateur, poorly drawn, ugly, flat, lowres, cropped, ugly, bad anatomy, deformed body, missing fingers, extra fingers, blurry, deformed face, cropped, cropped face, conjoined twins, siamese twins, chibi, weird eyes, worst quality, low quality',
    audio_filepath = audio_filepath,    # Use your own file
    audio_start_sec=audio_offsets[0],       # Start second of the provided audio
    fps=fps,                               # important to set yourself based on the num_interpolation_steps you defined
    batch_size=1,                          # increase until you go out of memory.
    output_dir='/content/gdrive/MyDrive/stable_diffusion_videos',                 # Where images will be saved
    name=name,                             # Subdir of output dir. will be timestamp by default
)
visualize_video_colab(video_path)
