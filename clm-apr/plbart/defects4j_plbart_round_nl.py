import codecs
import json
import sys
import os
import re
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

def defects4j_plbart_input(config, output_file, tmp_dir):
    loc_fp = codecs.open(PLBART_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')
    plbart_input = {'config': config, 'data': {}}
    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = PLBART_DIR + '../defects4j/tmp.json'

        subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_plbart_input(tmp_dir + path, start, end, config, tmp_file)
        
        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        if result['input'].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue
        result = json.load(open(tmp_file, 'r'))

        result['input'] = re.sub('/\\* buggy line:', '/* ', result['input']).replace('/*', '').replace('*/',
                                                                                                       '').replace(
            ' <mask>', '')


        #
        subprocess.run(
            ['defects4j', 'checkout', '-p', proj, '-v',
             bug_id + 'f', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_plbart_input(tmp_dir + path, start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result_target = json.load(open(tmp_file, 'r'))
        if result_target['input'].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue
        result_target = json.load(open(tmp_file, 'r'))

        golden = re.sub('/\\* buggy line:', '/* ', result_target['input']).replace('/*', '').replace('*/',
                                                                                                       '').replace(
            ' <mask>', '')
        #
        plbart_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
            'loc': rem_loc,
            'input': re.sub('\\s+', ' ', result['input']).strip(),
            'target': re.sub('\\s+', ' ', golden).strip(),
            'function range': result['function range']
        }

        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(plbart_input, open(output_file, 'w'), indent=2)

def defects4j_plbart_output(input_file, output_file, model_dir, model_name, seed, num_output=5, temp=1.0, num_beams = 10):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_model = f"plbart-single_task-java_en"
    second_model = f"plbart-single_task-en_java"

    tokenizer_summarize = PLBartTokenizer.from_pretrained(model_dir + first_model, local_files_only=True,
                                                          config=model_dir + model_name + '/config.json',
                                                          src_lang="__java__", tgt_lang="__en_XX__",
                                                          language_codes='base')
    model_summarize = PLBartForConditionalGeneration.from_pretrained(model_dir + first_model, local_files_only=True,
                                                                     device_map=DEVICE_MAP)
    tokenizer_code_generate = PLBartTokenizer.from_pretrained(model_dir + second_model, local_files_only=True,
                                                              config=model_dir + model_name + '/config.json',
                                                              src_lang="__en_XX__", tgt_lang="__java__",
                                                              language_codes='base')
    model_code_generate = PLBartForConditionalGeneration.from_pretrained(model_dir + second_model,
                                                                         local_files_only=True, device_map=DEVICE_MAP)

    plbart_output = json.load(open(input_file, 'r'))
    plbart_output['model'] = model_name
    plbart_output['seed'] = seed

    for filename in plbart_output['data']:
        text = plbart_output['data'][filename]['input']
        inp_scope, inp_ret_type, inp_name = parse_method(text)

        temp = float(temp)
        num_beams = int(num_beams)
        print('generating', filename)
        raw_output = []
        output = []
        first_output = []

        input_ids = tokenizer_summarize(text, add_special_tokens=True, return_tensors="pt", padding=True).to(device)
        if input_ids.input_ids.size(1) >= 512 or inp_name == "":
            print('input too long:', input_ids.input_ids.size(1), 'skip')
            for i in range(num_output):
                first_output.append("Input too long")
                for j in range(num_output):
                    raw_output.append("Too long")
                    output.append("")

            plbart_output['data'][filename]['raw_output'] = raw_output
            plbart_output['data'][filename]['mid_translation'] = first_output
            plbart_output['data'][filename]['output'] = output

            continue
        generated_ids = model_summarize.generate(
            input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, max_length=128, num_beams=num_beams,
            num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp,
            decoder_start_token_id=tokenizer_summarize.lang_code_to_id["__en_XX__"]
        )
        for generated_id in generated_ids:
            first_output.append(tokenizer_summarize.decode(generated_id, skip_special_tokens=True))

        inp_generation = tokenizer_code_generate(first_output, add_special_tokens=True, return_tensors="pt",
                                                 padding=True).to(device)
        generated_ids = model_code_generate.generate(
            input_ids=inp_generation.input_ids, attention_mask=inp_generation.attention_mask, max_length=512,
            num_beams=num_beams, num_return_sequences=num_output,
            early_stopping=True,
            temperature=temp,
            decoder_start_token_id=tokenizer_code_generate.lang_code_to_id["__java__"],
            bad_words_ids=[[50005]]
        )

        #

        for generated_id in generated_ids:
            out = tokenizer_code_generate.decode(generated_id, skip_special_tokens=True)
            raw_output.append(out)

            out_scope, out_ret_type, out_name = parse_method(out)
            if out_name == "" or out_ret_type == "":
                output.append("")
                continue
            outputted_code = out[out.index(out_name) + len(out_name):]
            fixed_output = inp_scope + out_ret_type + inp_name + outputted_code

            output.append(fixed_output)

        del input_ids
        del generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        plbart_output['data'][filename]['raw_output'] = raw_output
        plbart_output['data'][filename]['mid_translation'] = first_output
        plbart_output['data'][filename]['output'] = output
        json.dump(plbart_output, open(output_file, 'w'), indent=2)
    json.dump(plbart_output, open(output_file, 'w'), indent=2)


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
    input_file = PLBART_DIR + '../defects4j/plbart_result/plbart_input_round.json'
    config = "PLBART_SEQFORM_COMMENTFORM_NOCOMMENT"

    print("==========Preparing input of Defects4J benchmark to PLBART model, Config: Round ==========")
    # Uncomment this to generate input file
    # defects4j_plbart_input(config, input_file, tmp_dir='/tmp/plbart/')
    print("==========Input written to " + input_file)

    n_runs = 10
    for i in range(n_runs):
        folder_run = PLBART_DIR + f'../defects4j/plbart_result/run_{i}/'
        print(f"Creating run {i}, folder: {folder_run}")
        if not os.path.exists(folder_run):
            command(['mkdir', folder_run])
        seed = random.randint(1, 999999)

        for model_name in ['plbart-java-nl-java']:
            output_file = folder_run + '_'.join(model_name.split('-')) + '_output_round_nl_batch.json'

            print("==========Generating output of Defects4j benchmark by " + model_name + ", Config: " + config + "==========")
            defects4j_plbart_output(input_file=input_file, output_file=output_file, model_dir=model_dir, model_name=model_name, seed=seed, num_output=5, temp=1.0, num_beams=10)
            print("==========Output written to " + output_file)
