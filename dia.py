from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter
import os


# from huggingface_hub import login
# login(token="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")

pipeline = SpeakerDiarization()
print("-----pipeline build -----",pipeline)
mic = MicrophoneAudioSource()
print("mic source ------>",mic)
print("before inference")
inference = StreamingInference(pipeline, mic, do_plot=True)
print("inference build complete")

# Create the output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
inference.attach_observers(RTTMWriter(mic.uri, os.path.join(output_dir, "file.rttm")))

#inference.attach_observers(RTTMWriter(mic.uri, "/output/file.rttm"))
prediction = inference()
print("------prediction-------",prediction)



