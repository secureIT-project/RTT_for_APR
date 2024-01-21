import os
import sys
import re
import json
import codecs
import subprocess
from codet5_config import CodeT5InputConfig, parse_method
from transformers import RobertaTokenizer, T5ForConditionalGeneration, set_seed
import random
import gc
import torch
import argparse
DEVICE_MAP = "auto"

CODET5_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = CODET5_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err


def get_codet5_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.codet5.CodeT5InputParser',
        filename, start, end, config, tmp_file
    ])


def quixbugs_codet5_input(config, output_file):
    loc_fp = codecs.open(CODET5_DIR + '../quixbugs/quixbugs_loc.txt', 'r', 'utf-8')
    codet5_input = {'config': config, 'data': {}}
    # codet5_input = json.load(open(output_file, 'r'))
    for line in loc_fp.readlines():
        filename, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = CODET5_DIR + '../quixbugs/tmp.json'
        get_codet5_input(CODET5_DIR + '../quixbugs/java_programs/' + filename + '.java', start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        result['input'] = re.sub('// buggy line', '', result['input'])

        #
        get_codet5_input(CODET5_DIR + '../quixbugs/correct_java_programs/' + filename + '.java', start, end, config,
                         tmp_file)
        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')
        result_target = json.load(open(tmp_file, 'r'))
        golden = re.sub('// buggy line', '', result_target['input'])
        #

        codet5_input['data'][filename] = {
            'loc': rem_loc,
            'input': result['input'],
            'target': golden,
            'function range': result['function range']
        }
        command(['rm', '-rf', tmp_file])
    json.dump(codet5_input, open(output_file, 'w'), indent=2)



def quixbugs_codet5_output(input_file, output_file, model_dir, model_name, seed, num_output=5, temp=1.0, num_beams = 10):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_name = 'codet5-base'
    first_model = 'codet5-base-codexglue-translate-java-cs'
    second_model = 'codet5-base-codexglue-translate-cs-java'

    tokenizer = RobertaTokenizer.from_pretrained(model_dir + tokenizer_name, local_files_only=True, config=model_dir + tokenizer_name+'/config.json')

    model_first = T5ForConditionalGeneration.from_pretrained(model_dir + first_model, local_files_only=True, device_map=DEVICE_MAP)
    model_second = T5ForConditionalGeneration.from_pretrained(model_dir + second_model, local_files_only=True, device_map=DEVICE_MAP)

    codet5_output = json.load(open(input_file, 'r'))
    codet5_output['model'] = model_name
    codet5_output['seed'] = seed

    for filename in codet5_output['data']:
        text = codet5_output['data'][filename]['input']
        inp_scope, inp_ret_type, inp_name = parse_method(text)

        temp = float(temp)
        num_output = int(num_output)
        num_beams = int(num_beams)
        print('generating', filename)

        first_output = []
        raw_output = []
        output = []

        input_ids = tokenizer(text, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
        generated_ids = model_first.generate(
            input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, max_length=512, num_beams=num_beams, num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp
        )
        for generated_id in generated_ids:
            first_output.append(tokenizer.decode(generated_id, skip_special_tokens=True))


        inp_generation = tokenizer(first_output, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
        generated_ids = model_second.generate(
            input_ids=inp_generation.input_ids, attention_mask=inp_generation.attention_mask, max_length=512, num_beams=num_beams, num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp
        )


        for generated_id in generated_ids:
            out = tokenizer.decode(generated_id, skip_special_tokens=True)
            raw_output.append(out)

            out_scope, out_ret_type, out_name = parse_method(out)
            if out_name == "" or out_ret_type == "":
                output.append("")
                continue
            outputted_code = out[out.index(out_name)+len(out_name):]
            fixed_output = inp_scope + out_ret_type + inp_name + outputted_code
            output.append(fixed_output)

        del input_ids
        del inp_generation
        gc.collect()
        torch.cuda.empty_cache()

        codet5_output['data'][filename]['raw_output'] = raw_output
        codet5_output['data'][filename]['mid_translation'] = first_output
        codet5_output['data'][filename]['output'] = output
        json.dump(codet5_output, open(output_file, 'w'), indent=2)


def codet5_output_to_patch(output, config):
    if config in ['CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT', 'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT']:
        return output.strip()
    elif config == 'CODET5_REFINE_CODEFORM_NOCOMMENT':
        stack = ['{']
        start_index = output.index('{')
        patch = output[: start_index + 1]
        for c in output[start_index + 1: ]:
            patch += c
            if c == '}':
                top = stack.pop()
                if top != '{':
                    return ''
                if len(stack) == 0:
                    return patch.strip()
            elif c == '{':
                stack.append(c)
        return ''


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
    config = 'CODET5_REFINE_CODEFORM_NOCOMMENT'


    input_file = CODET5_DIR + '../quixbugs/codet5_result/codet5_input_round.json'
    print("==========Preparing input of QuixBugs benchmark to CODET5 model, Config: " + config + "==========")
    # Uncomment this to generate input file
    # quixbugs_codet5_input(config, input_file)
    print("==========Input written to " + input_file)

    n_runs = 10
    for i in range(n_runs):
        folder_run = CODET5_DIR + f'../quixbugs/codet5_result/run_{i}/'
        print(f"Creating run {i}, folder: {folder_run}")
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)

        for model_name in ['codet5-java-cs-java']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_csharp_batch.json'

            print("==========Generating output of Quixbugs benchmark by " + model_name + ", Config: " + config + "==========")
            quixbugs_codet5_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed, num_output=5, temp=1.0, num_beams=10)
            print("==========Output written to " + output_file)
