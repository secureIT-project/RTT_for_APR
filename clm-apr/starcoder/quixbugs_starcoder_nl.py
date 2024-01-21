import os
import json
import sys
import codecs
import subprocess
import re
from accelerate import infer_auto_device_map
import random
import gc
import torch
from starcoder_config import InCoderInputConfig, extract_fim_part, incoder_output_to_patch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import argparse
DEVICE_MAP = "auto"


STARCODER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = STARCODER_DIR + '../../jasper/'
FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
FIM_PAD = "<fim_pad>"
EOD = "<|endoftext|>"


def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err


def get_starcoder_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.incoder.InCoderInputParser',
        filename, start, end, config, tmp_file
    ])


def quixbugs_starcoder_input(config, output_file):
    loc_fp = codecs.open(STARCODER_DIR + '../quixbugs/quixbugs_loc.txt', 'r', 'utf-8')
    starcoder_input = {'config': config, 'data': {}}
    head_insert_comment = f"{FIM_PREFIX}\n\n/**\n@description {FIM_SUFFIX}\n*/\n"
    for line in loc_fp.readlines():
        filename, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = STARCODER_DIR + '../quixbugs/tmp.json'
        get_starcoder_input(STARCODER_DIR + '../quixbugs/java_programs/' + filename + '.java', start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        result['input'] = re.sub('// buggy line:', '', result['input'])
        result['input'] = head_insert_comment+re.sub('<\|mask:0\|>\n','', result['input'])
        result['input'] = re.sub('\n<\|mask:0\|>','', result['input'])+f"\n{FIM_MIDDLE}"

        #
        get_starcoder_input(STARCODER_DIR + '../quixbugs/correct_java_programs/' + filename + '.java', start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result_target = json.load(open(tmp_file, 'r'))
        golden = re.sub('// buggy line:', '', result_target['input'])
        golden = golden.replace('<|mask:0|>\n', '').replace('<|mask:0|>', '')


        #


        starcoder_input['data'][filename] = {
            'loc': rem_loc,
            'input': result['input'],
            'target': golden,
            'function range': result['function range']
        }
        command(['rm', '-rf', tmp_file])
    json.dump(starcoder_input, open(output_file, 'w'), indent=2)

def quixbugs_starcoder_output(input_file, output_file, model_dir, model_name, seed, num_outputs=5, temp_summ = 0.3, temp_gen=0.4):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir + model_name, local_files_only =True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name, local_files_only=True, device_map=DEVICE_MAP, trust_remote_code=True)#.to(device_id)
    tokenizer.add_special_tokens({
        "pad_token": EOD
    })
    starcoder_output = json.load(open(input_file, 'r'))
    #starcoder_output = json.load(open(output_file, 'r'))
    starcoder_output['model'] = model_name
    starcoder_output['seed'] = seed
    total = 0
    for filename in starcoder_output['data']:
        if 'output' in starcoder_output['data'][filename]:
            continue
        total +=1
        text = starcoder_output['data'][filename]['input']
        signature = text[:text.index('{')].split('\n')[-1]
        second_new_inp = signature + '{\n     '+FIM_SUFFIX+'\n}\n'+FIM_MIDDLE

        print('generating', filename)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)#.to(device_id)
        length_limit = 128

        generated_ids = model.generate(
        input_ids, num_return_sequences=num_outputs,
            do_sample=True, top_p= 0.95, pad_token_id=tokenizer.pad_token_id,
            temperature=temp_summ, max_new_tokens=length_limit,
            repetition_penalty=1,
            bad_words_ids=[[10766],[4296]] # ban word 'TODO'
        )

        first_raw_output = []
        first_output = []
        for generated_id in generated_ids:
            raw = tokenizer.decode(generated_id, skip_special_tokens=False)
            first_raw_output.append(raw)
            extracted_description = '\n\n/**\n@description ' + extract_fim_part(raw).strip() + '\n*/\n'
            first_output.append(extracted_description)

        raw_output = []
        output = []
        input_generate_batch = []
        for nl_description in first_output:
            if len(nl_description)< 15:
                for i in range(num_outputs):
                    raw_output.append("Description too short")
                    output.append("")
                continue


            first_new_inp = f"{FIM_PREFIX}"+ nl_description
            new_input = first_new_inp + second_new_inp
            input_generate_batch.append(new_input)


        inp_generation = tokenizer(input_generate_batch, return_tensors="pt", padding=True).to(device)#.to(device_id)
        length_limit = 512
        generated_ids_generation = model.generate(
            **inp_generation, num_return_sequences=num_outputs,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
            top_p = 0.95,
            temperature=temp_gen, max_new_tokens=length_limit,
        repetition_penalty=1,
        bad_words_ids=[[10766],[4296]] # ban word 'TODO'
        )

        outputs_generation = (tokenizer.batch_decode(generated_ids_generation, clean_up_tokenization_spaces=False,
                                                     skip_special_tokens=False))
        for out in outputs_generation:
            raw_output.append(out)
            patch = signature + '{\n     ' + extract_fim_part(out)
            output.append(incoder_output_to_patch(patch + '}'))

        while len(output)< num_outputs*num_outputs:
            raw_output.append("Description too short")
            output.append("")

        del input_ids
        del inp_generation
        gc.collect()
        torch.cuda.empty_cache()

        starcoder_output['data'][filename]['raw_mid_translation'] = first_raw_output
        starcoder_output['data'][filename]['mid_translation'] = first_output
        starcoder_output['data'][filename]['raw_output'] = raw_output
        starcoder_output['data'][filename]['output'] = output
        json.dump(starcoder_output, open(output_file, 'w'), indent=2)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device_map",
        type=str,
        help="set up a custom device_map to restrict the model to run on CPU (cpu)",
        default="auto",
    )

    args = parser.parse_args()

    DEVICE_MAP = args.device_map

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = "./../../models/"

    input_file = STARCODER_DIR + '../quixbugs/starcoder_result/starcoder_input_round_nl.json'
    print("==========Preparing input of QuixBugs benchmark to STARCODER model"+ "==========")
    config = "INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT"
    # Uncomment this to generate input file
    # quixbugs_starcoder_input(config, input_file)
    print("==========Input written to " + input_file)

    n_runs = 10
    for i in range(n_runs):
        folder_run = STARCODER_DIR + f'../quixbugs/starcoder_result/run_{i}/'
        print(f"Creating run {i}, folder: {folder_run}")
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)
        for model_name in ['starcoderbase']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_nl_batch.json'

            print("==========Generating output of QuixBugs benchmark by " + model_name + "==========")
            quixbugs_starcoder_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed)
            print("==========Output written to " + output_file)
