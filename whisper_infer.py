import whisper

import soundfile
import pandas as pd
import jiwer
import json

model = whisper.load_model("turbo")

# test_df = pd.read_csv("/cpfs04/user/cuiziyun/rzd/final_project_data/30h_data/test.tsv", sep='\t')
with open("/cpfs04/user/cuiziyun/rzd/final_project_data/30h_data/test.tsv") as f:
    test_datas = f.readlines()
with open("/cpfs04/user/cuiziyun/rzd/final_project_data/30h_data/test.wrd") as f:
    refs_file = f.readlines()
# refs = [jiwer.wer_standardize(ref) for ref in refs]
refs = []
for ref in refs_file:
    ref = " ".join(jiwer.wer_standardize(ref)[0])
    refs.append(ref)
preds = []
# for _, sample in test_df.iterrows():
for sample in test_datas[1:]:
    audio_path = sample.split('\t')[2].replace("/data2", "/cpfs04/user/cuiziyun/rzd")

    result = model.transcribe(audio_path) 
    text = " ".join(jiwer.wer_standardize(result["text"])[0])
    print(text)
    preds.append(text)

with open("pred_whisper.json", 'w') as f:
    json.dump(preds, f, ensure_ascii=False, indent=4)
with open("refs.json", 'w') as f:
    json.dump(refs, f, ensure_ascii=False, indent=4)

wer = jiwer.wer(refs, preds)
print(wer)
