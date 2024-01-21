import os
import sys
import json
import codecs
import subprocess
import re
from incoder_config import InCoderInputConfig, incoder_output_to_patch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import random
import gc
import torch
import argparse
DEVICE_MAP = "auto"

INCODER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = INCODER_DIR + '../../jasper/'


def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err


def get_incoder_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.incoder.InCoderInputParser',
        filename, start, end, config, tmp_file
    ])


def defects4j_incoder_input(config, output_file, tmp_dir):
    loc_fp = codecs.open(INCODER_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')
    incoder_input = {'config': config, 'data': {}}
    head_insert_comment = "\n\n/**\n@description <|mask:0|>\n*/\n"

    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = INCODER_DIR + '../defects4j/tmp.json'

        subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_incoder_input(tmp_dir + path, start, end, config, tmp_file)
        
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
        result['input'] = head_insert_comment+re.sub('<\|mask:0\|>\n','', result['input'])
        #
        subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'f', '-w', tmp_dir], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        get_incoder_input(tmp_dir + path, start, end, config, tmp_file)

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

        incoder_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
            'loc': rem_loc,
            'input': result['input'],
            'target': golden,
            'function range': result['function range']
        }

        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(incoder_input, open(output_file, 'w'), indent=2)
    json.dump(incoder_input, open(output_file, 'w'), indent=2)


def defects4j_incoder_output(input_file, output_file, model_dir, model_name, seed, num_outputs=5, temp_summ = 0.3, temp_gen=0.4):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir + model_name, local_files_only=True,
                                              config=model_dir + model_name + '/config.json')
    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name, local_files_only=True,
                                                 device_map=DEVICE_MAP, load_in_8bit=True)  # .to(device_id)

    tokenizer.pad_token = "<pad>"
    tokenizer.padding_side = "left"

    try:
        incoder_output = json.load(open(output_file, 'r'))
    except:
        incoder_output = json.load(open(input_file, 'r'))
    incoder_output['model'] = model_name
    incoder_output['seed'] = seed
    total = 0
    for filename in incoder_output['data']:
        if 'output' in incoder_output['data'][filename]:
            continue
        total += 1
        text = incoder_output['data'][filename]['input']
        signature = text[:text.index('{')].split('\n')[-1]
        second_new_inp = signature + '{\n     <|mask:0|>\n}\n<|mask:0|>'

        print(device_id)
        print('generating', filename)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)  # .to(device_id)
        # eos_id = tokenizer.convert_tokens_to_ids('</code>')

        current_length = input_ids.flatten().size(0)
        length_limit = 128
        first_raw_output = []
        first_output = []
        raw_output = []
        output = []
        if input_ids.size(1) >= 2048:
            print('too long:', input_ids.size(1))
            for i in range(num_outputs):
                first_output.append("Input too long")
                for j in range(num_outputs):
                    raw_output.append("Too long")
                    output.append("")

            incoder_output['data'][filename]['raw_output'] = raw_output
            incoder_output['data'][filename]['mid_translation'] = first_output
            incoder_output['data'][filename]['output'] = output

            continue

        generated_ids = model.generate(
            input_ids, num_return_sequences=5,
            # early_stopping=True,
            #  pad_token_id=eos_id, eos_token_id=eos_id,
            do_sample=True, top_p=0.95,
            temperature=temp_summ, max_new_tokens=length_limit
        )


        for generated_id in generated_ids:
            raw = tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            first_raw_output.append(raw)
            extracted_description = '/**\n@description ' + raw.split('<|mask:0|>')[2].split('*/')[0].strip() + '\n*/\n'
            first_output.append(extracted_description)


        input_generate_batch = []
        for nl_description in first_output:
            if len(nl_description) < 15:
                for i in range(num_outputs):
                    raw_output.append("Description too short")
                    output.append("")
                continue

            first_new_inp = '<| file ext=.java |>\n' + nl_description
            new_input = first_new_inp + second_new_inp
            input_generate_batch.append(new_input)

        inp_generation = tokenizer(input_generate_batch, return_tensors="pt", padding=True).to(device)  # .to(device_id)
        # current_length = inp_generation.flatten().size(0)
        length_limit = 512
        max_length = min(length_limit + current_length, 2048)
        generated_ids_generation = model.generate(
            input_ids=inp_generation.input_ids, attention_mask=inp_generation.attention_mask, num_return_sequences=5,
            # early_stopping=True,
            #  pad_token_id=eos_id, eos_token_id=eos_id,
            do_sample=True, top_p=0.95,
            temperature=temp_gen, max_length=max_length
        )

        outputs_generation = (tokenizer.batch_decode(generated_ids_generation, clean_up_tokenization_spaces=False,
                                                     skip_special_tokens=False))
        for out in outputs_generation:
            raw_output.append(out)
            patch = incoder_output_to_patch(out)
            end_comment_ind = patch.find('*/')
            end_comment_ind = 0 if end_comment_ind == -1 else end_comment_ind + 2
            output.append(patch[end_comment_ind:])

        while len(output) < num_outputs * num_outputs:
            raw_output.append("Description too short")
            output.append("")

        del input_ids
        del inp_generation
        gc.collect()
        torch.cuda.empty_cache()

        incoder_output['data'][filename]['raw_mid_translation'] = first_raw_output
        incoder_output['data'][filename]['mid_translation'] = first_output
        incoder_output['data'][filename]['raw_output'] = raw_output
        incoder_output['data'][filename]['output'] = output
        json.dump(incoder_output, open(output_file, 'w'), indent=2)
    json.dump(incoder_output, open(output_file, 'w'), indent=2)



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
    device_id = 0
    config = "INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT"
    input_file = INCODER_DIR + '../defects4j/incoder_result/incoder_input_round_nl.json'
    print("==========Preparing input of Defects4J benchmark to INCODER model")
    # Uncomment this to generate input file
    # defects4j_incoder_input(config, input_file, tmp_dir='/tmp/incoder/')
    print("==========Input written to " + input_file)

    n_runs = 9
    for i in range(n_runs):
        folder_run = INCODER_DIR + f'../defects4j/incoder_result/run_{i}/'
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)
        for model_name in ['incoder-1B','incoder-6B']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_nl_batch.json'
            print("==========Generating output of QuixBugs benchmark by " + model_name + "==========")
            defects4j_incoder_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed)
            print("==========Output written to " + output_file)
