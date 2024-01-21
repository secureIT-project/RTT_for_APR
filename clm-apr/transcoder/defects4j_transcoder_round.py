import codecs
import json
import os
import re
import sys
import subprocess
from codet5_config import CodeT5InputConfig, parse_method
CUR_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(CUR_DIR + '../../')
sys.path.append(CUR_DIR + '../../codegen/')
from codegen_sources.model.translate_returned import returned_translation, Translator

TRANSCODER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
BPE_path= TRANSCODER_DIR +"./codegen/data/bpe/cpp-java-python/codes"
JAVA_DIR = TRANSCODER_DIR + '../../jasper/'
import random
import gc
import torch
import argparse
DEVICE_MAP = "auto"


def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def defects4j_transcoder_output(input_file, output_file, model_dir, model_name, seed, tgt_lang, num_output=5, temp=1.0, num_beams = 10):
    # The implementation of the model ignores temperature if beam_size != 1
    torch.manual_seed(seed)

    translator = Translator(model_dir, BPE_path)

    transcoder_output = json.load(open(input_file, 'r'))
    transcoder_output['model'] = model_name
    transcoder_output['seed'] = seed

    src_lang = 'java'

    for filename in transcoder_output['data']:
        raw_output = []
        output = []
        first_output = []
        try:
            text = transcoder_output['data'][filename]['input']
            inp_scope, inp_ret_type, inp_name = parse_method(text)

            print('generating', filename)

            temp = float(temp)
            num_output = int(num_output)
            num_beams = int(num_beams)
            ######
            first_output_full = translator.translate(
                text,
                lang1=src_lang,
                lang2=tgt_lang,
                beam_size=num_beams,
                sample_temperature=temp,
                max_tokens=512
            )
            first_output = first_output_full[:num_output]
            if first_output_full[0][:5] == 'Error':
                raise ValueError('Input too long')
        except Exception as e:
            for i in range(num_output):
                #first_output.append("Input too long")
                for j in range(num_output):
                    raw_output.append("Too long")
                    output.append("")

            transcoder_output['data'][filename]['raw_output'] = raw_output
            transcoder_output['data'][filename]['mid_translation'] = first_output
            transcoder_output['data'][filename]['output'] = output
            continue


        for translation in first_output:
            try:
                final_output_full = translator.translate(
                    translation,
                    lang1=tgt_lang,
                    lang2=src_lang,
                    beam_size=num_beams,
                    sample_temperature=temp
                )
                for out in final_output_full[:num_output]:
                    raw_output.append(out)

                    out_scope, out_ret_type, out_name = parse_method(out)
                    if out_name == "" or out_ret_type == "":
                        output.append("")
                        continue
                    outputted_code = out[out.index(out_name) + len(out_name):]
                    fixed_output = inp_scope + out_ret_type + inp_name + outputted_code

                    output.append(fixed_output)
            except Exception as e:
                for i in range(num_output):
                    print(e)
                    raw_output.append("Error")
                    output.append("")
                continue

            gc.collect()
            torch.cuda.empty_cache()

        #
        transcoder_output['data'][filename]['raw_output'] = raw_output
        transcoder_output['data'][filename]['mid_translation'] = first_output
        transcoder_output['data'][filename]['output'] = output
        json.dump(transcoder_output, open(output_file, 'w'), indent=2)
    json.dump(transcoder_output, open(output_file, 'w'), indent=2)


if __name__ == '__main__':
    # Input is same as for CodeT5

    # To cpp
    tgt_lang_first = 'cpp'
    model_dir_first = TRANSCODER_DIR+ "../../models/transcoder_cpp/model.pth"
    # To python
    tgt_lang_second = 'python'
    model_dir_second = TRANSCODER_DIR+ "../../models/transcoder_python/model.pth"

    config = 'CODET5_REFINE_CODEFORM_NOCOMMENT'
    input_file = TRANSCODER_DIR + '../defects4j/transcoder_result/transcoder_input_round.json'

    modes = [[tgt_lang_first, model_dir_first],[tgt_lang_second, model_dir_second]]

    n_runs = 10
    for i in range(n_runs):
        folder_run = TRANSCODER_DIR + f'../defects4j/transcoder_result/run_{i}/'
        print(f"Creating run {i}, folder: {folder_run}")
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)

        for mode in modes:
            lang = mode[0]
            model_dir = mode[1]
            model_name = f'transcoder-java-{lang}-java'
            output_file = folder_run + '_'.join(model_name.split('-')) + f'_output_round_{lang}_batch.json'

            print("==========Generating output of Defects4j benchmark by " + model_name + ", Config: " + config + "==========")
            defects4j_transcoder_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed, tgt_lang=lang, num_output=5, temp=1.0, num_beams=10)
            print("==========Output written to " + output_file)

