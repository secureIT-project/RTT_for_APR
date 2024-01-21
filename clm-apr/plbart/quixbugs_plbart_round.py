import re
import os
import sys
import json
import codecs
import subprocess
from plbart_config import PLBartInputConfig, parse_method
from transformers import PLBartForConditionalGeneration, PLBartTokenizer, set_seed
import random
import gc
import torch
import argparse
DEVICE_MAP = "auto"

PLBART_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = PLBART_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err


def get_plbart_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.plbart.PLBartInputParser',
        filename, start, end, config, tmp_file
    ])


def quixbugs_plbart_input(config, output_file):
    loc_fp = codecs.open(PLBART_DIR + '../quixbugs/quixbugs_loc.txt', 'r', 'utf-8')
    plbart_input = {'config': config, 'data': {}}
    for line in loc_fp.readlines():
        filename, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = PLBART_DIR + '../quixbugs/tmp.json'
        get_plbart_input(PLBART_DIR + '../quixbugs/java_programs/' + filename + '.java', start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        result['input'] = re.sub('/\\* buggy line:', '/* ', result['input']).replace('/*', '').replace('*/',
                                                                                                       '').replace(
            '<mask>', '')

        #
        get_plbart_input(PLBART_DIR + '../quixbugs/correct_java_programs/' + filename + '.java', start, end, config, tmp_file)
        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')
        result_target = json.load(open(tmp_file, 'r'))
        golden = re.sub('/\\* buggy line:', '/* ', result_target['input']).replace('/*', '').replace('*/',
                                                                                                       '').replace(
            '<mask>', '')
        #

        plbart_input['data'][filename] = {
            'loc': rem_loc,
            'input': re.sub('\\s+', ' ', result['input']).strip(),
            'target': re.sub('\\s+', ' ', golden).strip(),
            'function range': result['function range']
        }
        command(['rm', '-rf', tmp_file])
    json.dump(plbart_input, open(output_file, 'w'), indent=2)


def quixbugs_plbart_output(input_file, output_file, model_dir, model_name, seed, num_output=5, temp=1.0, num_beams = 10):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_model = f"plbart-java-cs"
    second_model = f"plbart-cs-java"

    tokenizer_first = PLBartTokenizer.from_pretrained(model_dir + first_model, local_files_only =True, config=model_dir + model_name+'/config.json')
    tokenizer_second = PLBartTokenizer.from_pretrained(model_dir + second_model, local_files_only =True, config=model_dir + model_name+'/config.json')

    model_first = PLBartForConditionalGeneration.from_pretrained(model_dir + first_model, local_files_only=True, device_map=DEVICE_MAP)
    model_second = PLBartForConditionalGeneration.from_pretrained(model_dir + second_model, local_files_only=True, device_map=DEVICE_MAP)

    plbart_output = json.load(open(input_file, 'r'))
    plbart_output['model'] = model_name
    plbart_output['seed'] = seed


    for filename in plbart_output['data']:            
        text = plbart_output['data'][filename]['input']
        inp_scope, inp_ret_type, inp_name = parse_method(text)

        temp = float(temp)
        num_output = int(num_output)
        num_beams = int(num_beams)

        print('generating', filename)
        first_output = []

        raw_output = []
        output = []
        input_ids = tokenizer_first(text, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
        generated_ids = model_first.generate(
            input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, max_length=512, num_beams=num_beams, num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp
        )
        for generated_id in generated_ids:
            first_output.append(tokenizer_first.decode(generated_id, skip_special_tokens=True))


        inp_generation = tokenizer_second(first_output, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
        generated_ids = model_second.generate(
            input_ids=inp_generation.input_ids, attention_mask=inp_generation.attention_mask, max_length=512, num_beams=num_beams, num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp
        )

        #

        for generated_id in generated_ids:
            out = tokenizer_second.decode(generated_id, skip_special_tokens=True)
            raw_output.append(out)

            out_scope, out_ret_type, out_name = parse_method(out)
            if out_name == "" or out_ret_type == "":
                output.append("")
                continue
            outputted_code = out[out.index(out_name) + len(out_name):]
            fixed_output = inp_scope + out_ret_type + inp_name + outputted_code
            output.append(fixed_output)

        del input_ids
        del inp_generation
        gc.collect()
        torch.cuda.empty_cache()

        plbart_output['data'][filename]['raw_output'] = raw_output
        plbart_output['data'][filename]['mid_translation'] = first_output
        plbart_output['data'][filename]['output'] = output
        json.dump(plbart_output, open(output_file, 'w'), indent=2)


def plbart_output_to_patch(output, config):
    output = re.sub('/\\*.*\\*/', '', output)
    if config in ['PLBART_SEQFORM_MASKFORM_NOCOMMENT', 'PLBART_SEQFORM_COMMENTFORM_NOCOMMENT']:
        stack = ['{']
        if '{' not in output:
            return ''
        start_index = output.index('{')
        patch = output[: start_index + 1]
        for c in output[start_index + 1:]:
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

    # Keep the buggy line
    config = 'PLBART_SEQFORM_COMMENTFORM_NOCOMMENT'

    input_file = PLBART_DIR + '../quixbugs/plbart_result/plbart_input_round.json'

    print("==========Preparing input of QuixBugs benchmark to PLBART model, Config: " + config + "==========")
    # Uncomment this to generate input file
    # quixbugs_plbart_input(config, input_file)
    print("==========Input written to " + input_file)
    
    n_runs = 10

    for i in range(n_runs):
        folder_run = PLBART_DIR + f'../quixbugs/plbart_result/run_{i}/'
        print(f"Creating run {i}, folder: {folder_run}")
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)

        for model_name in ['plbart-java-cs-java']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_csharp_batch.json'

            print("==========Generating output of Quixbugs benchmark by " + model_name + ", Config: " + config + "==========")
            quixbugs_plbart_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed, num_output=5, temp=1.0, num_beams=10)
            print("==========Output written to " + output_file)
