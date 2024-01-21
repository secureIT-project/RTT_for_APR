import os
import sys
import json
import codecs
import subprocess
import re
from starcoder_config import InCoderInputConfig, extract_fim_part, incoder_output_to_patch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import gc
import torch
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


def defects4j_starcoder_input(config, output_file, tmp_dir):
    loc_fp = codecs.open(STARCODER_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')
    starcoder_input = {'config': config, 'data': {}}
    head_insert_comment = f"{FIM_PREFIX}\n\n/**\n@description {FIM_SUFFIX}\n*/\n"

    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = STARCODER_DIR + '../defects4j/tmp.json'

        subprocess.run(
            ['defects4j', 'checkout', '-p', proj, '-v',
             bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_starcoder_input(tmp_dir + path, start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        if result['input'].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue

        result = json.load(open(tmp_file, 'r'))
        result['input'] = re.sub('// buggy line:', '', result['input'])
        result['input'] = head_insert_comment + re.sub('<\|mask:0\|>\n', '', result['input'])
        result['input'] = re.sub('\n<\|mask:0\|>', '', result['input']) + f"\n{FIM_MIDDLE}"
        #
        subprocess.run(
            ['defects4j', 'checkout', '-p', proj, '-v',
             bug_id + 'f', '-w', tmp_dir], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        get_starcoder_input(tmp_dir + path, start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result_target = json.load(open(tmp_file, 'r'))
        if result_target['input'].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue

        result_target = json.load(open(tmp_file, 'r'))
        golden = re.sub('// buggy line:', '', result_target['input'])
        golden = golden.replace('<|mask:0|>\n', '').replace('<|mask:0|>', '')
        #

        starcoder_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
            'loc': rem_loc,
            'input': result['input'],
            'target': golden,
            'function range': result['function range']
        }

        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(starcoder_input, open(output_file, 'w'), indent=2)
    json.dump(starcoder_input, open(output_file, 'w'), indent=2)


def defects4j_incoder_output(input_file, output_file, model_dir, model_name, seed, num_outputs=5, temp_summ=0.3,
                             temp_gen=0.4):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir + model_name, local_files_only =True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name, local_files_only=True, device_map=DEVICE_MAP,
                                                 trust_remote_code=True)  # .to(device_id)
    tokenizer.add_special_tokens({
        "pad_token": EOD
    })

    starcoder_output = json.load(open(input_file, 'r'))
    starcoder_output['model'] = model_name
    starcoder_output['seed'] = seed
    total = 0
    for filename in starcoder_output['data']:
        if 'output' in starcoder_output['data'][filename]:
            continue
        total += 1
        text = starcoder_output['data'][filename]['input']
        signature = text[:text.index('{')].split('\n')[-1]
        second_new_inp = signature + '{\n     ' + FIM_SUFFIX + '\n}\n' + FIM_MIDDLE

        print('generating', filename)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  # .to(device_id)

        raw_output = []
        output = []
        first_raw_output = []
        first_output = []
        if input_ids.size(1) >= 8192 or 'Jsoup_15_/src/main/java/org/jsoup/parser/TreeBuilderState.java_283-284' == filename:
            print('too long:', input_ids.size(1))
            for i in range(num_outputs):
                first_output.append("Input too long")
                for j in range(num_outputs):
                    raw_output.append("Too long")
                    output.append("")

            starcoder_output['data'][filename]['raw_output'] = raw_output
            starcoder_output['data'][filename]['mid_translation'] = first_output
            starcoder_output['data'][filename]['output'] = output

            continue
        length_limit = 128
        generated_ids = model.generate(
            input_ids, num_return_sequences=num_outputs,
            do_sample=True, top_p=0.95,  pad_token_id=tokenizer.pad_token_id,
            temperature=temp_summ, max_new_tokens=length_limit,
            repetition_penalty=1,
            bad_words_ids=[[10766], [4296]]  # ban word 'TODO'
        )
        for generated_id in generated_ids:
            raw = tokenizer.decode(generated_id, skip_special_tokens=False)
            first_raw_output.append(raw)
            extracted_description = '\n\n/**\n@description ' + extract_fim_part(raw).strip() + '\n*/\n'
            first_output.append(extracted_description)


        raw_output = []
        output = []
        input_generate_batch = []
        for nl_description in first_output:
            if len(nl_description) < 15:
                for i in range(num_outputs):
                    raw_output.append("Description too short")
                    output.append("")
                continue

            first_new_inp = f"{FIM_PREFIX}" + nl_description
            new_input = first_new_inp + second_new_inp
            input_generate_batch.append(new_input)


        inp_generation = tokenizer(input_generate_batch, return_tensors="pt", padding=True).to(device)  # .to(device_id)
        length_limit = 2048
        generated_ids_generation = model.generate(
            **inp_generation, num_return_sequences=num_outputs,
            do_sample=True,  pad_token_id=tokenizer.pad_token_id,
            top_p=0.95,
            temperature=temp_gen, max_new_tokens=length_limit,
            repetition_penalty=1,
            bad_words_ids=[[10766], [4296]]  # ban word 'TODO'
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

    model_dir = "./../../models/"
    config = "INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT"
    input_file = STARCODER_DIR + '../defects4j/starcoder_result/starcoder_input_round.json'
    print("==========Preparing input of Defects4J benchmark to STARCODER model")
    # Uncomment this to generate input file
    # defects4j_starcoder_input(config, input_file, tmp_dir='/tmp/starcoder/')
    print("==========Input written to " + input_file)

    n_runs = 10
    for i in range(n_runs):
        folder_run = STARCODER_DIR + f'../defects4j/starcoder_result/run_{i}/'
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)
        for model_name in ['starcoderbase']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_nl_batch.json'

            print("==========Generating output of Defects4J benchmark by " + model_name)
            defects4j_incoder_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed)
            print("==========Output written to " + output_file)
