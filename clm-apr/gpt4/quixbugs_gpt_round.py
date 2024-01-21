import os
import sys
import re
import json
import codecs
import subprocess
from gpt_config import CodeT5InputConfig
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")
GPT_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = GPT_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err


def get_gpt_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.codet5.CodeT5InputParser',
        filename, start, end, config, tmp_file
    ])


def quixbugs_gpt_input(config, output_file):
    loc_fp = codecs.open(GPT_DIR + '../quixbugs/quixbugs_loc.txt', 'r', 'utf-8')
    gpt_input = {'config': config, 'data': {}}
    # codet5_input = json.load(open(output_file, 'r'))
    for line in loc_fp.readlines():
        filename, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = GPT_DIR + '../quixbugs/tmp.json'

        get_gpt_input(GPT_DIR + '../quixbugs/java_programs/' + filename + '.java', start, end, config, tmp_file)

        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        result['input'] = re.sub('// buggy line', '', result['input'])



        #
        get_gpt_input(GPT_DIR + '../quixbugs/correct_java_programs/' + filename + '.java', start, end, config,
                         tmp_file)
        if not os.path.exists(tmp_file):
            print(filename, 'failed.', output_file, 'not found.')
        print(filename, 'succeeded')
        result_target = json.load(open(tmp_file, 'r'))
        golden = re.sub('// buggy line', '', result_target['input'])
        #

        gpt_input['data'][filename] = {
            'loc': rem_loc,
            'input': result['input'],
            'target': golden,
            'function range': result['function range']
        }
        command(['rm', '-rf', tmp_file])
    json.dump(gpt_input, open(output_file, 'w'), indent=2)



def quixbugs_gpt_output(input_file, output_file, model_name, num_outputs=5, temp=0.2):
    system_prompt = "You are an expert programmer in all programming languages."
    user_prompt_beg = f"""Create a Javadoc for the Java function delimited by triple backquotes. Do not return generate the method again, return only the Javadoc.

    Java function:
    ```
    """

    user_prompt_end = """
    ```
    Javadoc:
    """
    generation_user_prompt_beg = f"""Given the signature of a Java function and its Javadoc delimited by triple backquotes, generate the body of the function. Do not generate any additional methods nor repeat the Javadoc nor give any explanations. Return only the completed function without any comments.
    ```
    """
    generation_user_prompt_end = """
    ```
    """

    try:
        gpt_output = json.load(open(output_file, 'r'))
    except:
        gpt_output = json.load(open(input_file, 'r'))

    gpt_output['model'] = model_name
    total = 0
    for filename in gpt_output['data']:
        if 'output' in gpt_output['data'][filename]:
            continue

        text = gpt_output['data'][filename]['input']

        print('generating', filename)


        user_prompt = user_prompt_beg + text + user_prompt_end

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            n=num_outputs,
            top_p = 0.95
        )

        # response['choices'][0]['message']['content']

        first_raw_output = []
        first_output = []
        for i in range(num_outputs):
            raw_output = response['choices'][i]['message']['content']
            first_raw_output.append(raw_output)
            try:
                raw_output = raw_output.replace('```java', '```')
                javadoc_answer = raw_output[raw_output.index('```') + 3:]
                javadoc_answer = javadoc_answer[:javadoc_answer.index('```')]
                first_output.append(javadoc_answer.strip())
            except:
                javadoc_answer = raw_output.strip()
                first_output.append(javadoc_answer)

        outputs = []
        raw_outputs = []
        gpt_output['data'][filename]['mid_translation'] = first_output
        gpt_output['data'][filename]['raw_mid_translation'] = first_raw_output
        json.dump(gpt_output, open(output_file, 'w'), indent=2)
        for javadoc_answer in first_output:

            function_signature = text[:text.index('{') + 1]
            generation_code_prompt = javadoc_answer + '\n' + function_signature

            generation_user_prompt = generation_user_prompt_beg + generation_code_prompt + generation_user_prompt_end
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": generation_user_prompt},
                ],
                temperature=0.2,
                n=num_outputs,
                top_p = 0.95
            )



            for i in range(num_outputs):
                raw_output = response['choices'][i]['message']['content']
                raw_outputs.append(raw_output)
                try:
                    raw_output = raw_output.replace('```java','```')
                    code_generated = raw_output[raw_output.index('```') + 3:]
                    code_generated = code_generated[:code_generated.index('```')]
                    outputs.append(code_generated.strip())
                except:
                    outputs.append(raw_output.strip())


                signature_ind = outputs[-1].find(function_signature)
                if signature_ind == -1:
                    outputs[-1] = function_signature + '\n' + outputs[-1]
                else:
                    outputs[-1] = outputs[-1][signature_ind:]

        gpt_output['data'][filename]['raw_output'] = raw_outputs
        gpt_output['data'][filename]['output'] = outputs
        json.dump(gpt_output, open(output_file, 'w'), indent=2)
    json.dump(gpt_output, open(output_file, 'w'), indent=2)


def gpt_output_to_patch(output, config):
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
    config = 'CODET5_REFINE_CODEFORM_NOCOMMENT'

    input_file = GPT_DIR + '../quixbugs/gpt_result/gpt_input_round.json'
    print("==========Preparing input of QuixBugs benchmark to GPT model, Config: " + config + "==========")
    # Uncomment this to generate input file
    # quixbugs_gpt_input(config, input_file)
    print("==========Input written to " + input_file)

    # gpt-3.5-turbo == gpt-3.5-turbo-0613
    for model_name in ['gpt-3.5-turbo']:

        output_file = GPT_DIR + '../quixbugs/gpt_result/' + '_'.join(model_name.split('-')) + f'_output_round.json'
        # model_dir = CODET5_DIR + '../models/'
        print("==========Generating output of QuixBugs benchmark by " + model_name + ", Config: " + config + "==========")
        quixbugs_gpt_output(input_file=input_file, output_file=output_file, model_name=model_name, num_outputs=5)
        print("==========Output written to " + output_file)
