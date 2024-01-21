import shutil
import json
import sys
import os
import time
import defects4j_command
import wandb
import pandas as pd
DEFECTS4J_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
from output_to_patch import codet5_output_to_patch, plbart_output_to_patch, incoder_output_to_patch

from evaluation.CodeBLEU.calc_code_bleu import compute_codebleu
import pickle
from crystalbleu import corpus_bleu
from multiprocessing import Lock, current_process, Manager
from multiprocessing.pool import Pool

defects4j12 = open("defects4j_12.txt","r").read().split()
trivially_shared_ngrams12_small = pickle.load(open("ignore_crystal_defects4j12100.pkl", "rb"))
trivially_shared_ngrams12_medium = pickle.load(open("ignore_crystal_defects4j12.pkl", "rb"))
trivially_shared_ngrams12_big = pickle.load(open("ignore_crystal_defects4j121000.pkl", "rb"))
trivially_shared_ngrams20_small = pickle.load(open("ignore_crystal_defects4j20100.pkl", "rb"))
trivially_shared_ngrams20_medium = pickle.load(open("ignore_crystal_defects4j20.pkl", "rb"))
trivially_shared_ngrams20_big = pickle.load(open("ignore_crystal_defects4j201000.pkl", "rb"))

def insert_fix(filename, start_line, end_line, patch):
    """
    end_row is included in the buggy lines / buggy function
    """
    with open(filename, 'r') as file:
        data = file.readlines()

    with open(filename, 'w') as file:
        for i in range(start_line - 1):
            file.write(data[i])
        file.write(patch.strip() + '\n')
        for i in range(end_line, len(data)):
            file.write(data[i])


def validate_defects4j(input_file, output_file, tmp_dir, config_wandb, problem_number, lock, run_number):
    plausible, total = 0, 0
    process_number = current_process().pid
    tmp_dir = tmp_dir[:-1] + str(process_number)+ str(round(time.time())) +'/'

    if not os.path.exists(tmp_dir):
        defects4j_command.command_with_timeout(['mkdir', tmp_dir])



    model_output = json.load(open(input_file, 'r'))
    try:
        validated_result = json.load(open(output_file, 'r'))
    except:
        validated_result = {'config': model_output['config'], 'data': {}}


    key = list(model_output['data'].keys())[problem_number]


    key_list = key.split('_')
    proj, bug_id, loc = key_list[0], key_list[1], key_list[-1]
    path = '_'.join(key_list[2: -1])
    model_name = model_output['model']

    if key in validated_result['data']:
        print('Already done: ', proj, bug_id, flush=True)
        return
    if 'output' not in model_output['data'][key]:
        print('No output: ', proj, bug_id, flush=True)
        return
    print('start validating', proj, bug_id, flush=True)
    total += 1


    problem_name = proj+'_'+bug_id
    if problem_name in defects4j12:
        trivially_shared_ngrams_medium = trivially_shared_ngrams12_medium
        trivially_shared_ngrams_small = trivially_shared_ngrams12_small
        trivially_shared_ngrams_big = trivially_shared_ngrams12_big
    else:
        trivially_shared_ngrams_medium = trivially_shared_ngrams20_medium
        trivially_shared_ngrams_small = trivially_shared_ngrams20_small
        trivially_shared_ngrams_big = trivially_shared_ngrams20_big

    dataset = 'Defects4J12' if problem_name in defects4j12 else 'Defects4J20'
    config_wandb['dataset'] = dataset
    project_name = f'YourProject'
    run = wandb.init(
        mode="disabled",
        project=project_name,
        config = config_wandb,
        settings=wandb.Settings(_disable_stats=True)
    )

    validated_result['data'][key] = {}
    for k, value in model_output['data'][key].items():
        if k != 'output':
            validated_result['data'][key][k] = value
    validated_result['data'][key]['output'] = []
    start_line, end_line = validated_result['data'][key]['loc'].split('-')
    end_line = str(int(end_line) - 1) if end_line != start_line else end_line
    function_start_line, function_end_line = validated_result['data'][key]['function range'].split('-')
    function_start_line, function_end_line = function_start_line.split(',')[0], function_end_line.split(',')[0]

    defects4j_command.clean_tmp_folder(tmp_dir)
    defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)
    if proj == "Mockito":
        defects4j_command.compile_fix(tmp_dir)


    start_time = time.time()
    init_out, init_err = defects4j_command.defects4j_test_suite(tmp_dir)
    standard_time = int(time.time() - start_time)


    failed_test_cases = str(init_out).split(' - ')[1:]
    for i, failed_test_case in enumerate(failed_test_cases):
        failed_test_cases[i] = failed_test_case.strip()
    init_fail_num = len(failed_test_cases)

    trigger, err = defects4j_command.defects4j_trigger(tmp_dir)
    triggers = trigger.strip().split('\n')
    for i, trigger in enumerate(triggers):
        triggers[i] = trigger.strip()

    num_tests = len(triggers)


    current_is_correct = False
    is_compilable = []
    is_plausible = []
    ratios_passed = []
    for rank, patch in enumerate(model_output['data'][key]['output']):
        filename = tmp_dir + path
        shutil.copyfile(filename, filename + '.bak')

        if 'CODET5' in model_output['config']:
            patch = codet5_output_to_patch(patch, model_output['config'])
            if model_output['config']  in ['CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT', 'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT']:
                insert_fix(filename, int(start_line), int(end_line), patch)
            elif model_output['config']  == 'CODET5_REFINE_CODEFORM_NOCOMMENT':
                insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        elif 'PLBART' in model_output['config']:
            patch = plbart_output_to_patch(patch, model_output['config'])
            insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        elif 'INCODER' in model_output['config']:
            # patch = incoder_output_to_patch(patch, model_output['config'])
            insert_fix(filename, int(function_start_line), int(function_end_line), patch)

        if proj == 'Mockito':
            # Mockito needs seperate compile
            defects4j_command.compile_fix(tmp_dir)

        outs = []
        correctness = None
        if patch == '':
            correctness = 'uncompilable'
            ratios_passed.append(0)
            is_compilable.append(False)
            is_plausible.append(False)
        else:

            start_time = time.time()
            out, err = defects4j_command.defects4j_test_suite(tmp_dir, timeout=min(400, int(1.5*standard_time)))

            if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                ratios_passed.append(0)
                is_compilable.append(True)
                is_plausible.append(False)
                correctness = 'timeout'
            elif 'FAIL' in str(err) or 'FAIL' in str(out):
                ratios_passed.append(0)
                is_compilable.append(False)
                is_plausible.append(False)
                correctness = 'uncompilable'
            elif "Failing tests: 0" in str(out):
                ratios_passed.append(1)
                is_compilable.append(True)
                is_plausible.append(True)
                if not current_is_correct:
                    current_is_correct = True
                    plausible += 1
                correctness = 'plausible'
                current_failed_test_cases = str(out).split(' - ')[1:]
            else:
                current_failed_test_cases = str(out).split(' - ')[1:]
                current_passed_test_cases = max(0, num_tests - len(current_failed_test_cases))
                ratios_passed.append(current_passed_test_cases/num_tests)
                is_compilable.append(True)
                is_plausible.append(False)
                correctness = 'wrong'

        validated_result['data'][key]['output'].append({
            'patch': patch, 'correctness': correctness
        })
        shutil.copyfile(filename + '.bak', filename)

    n_output = int(len(is_compilable)**(1/2))
    list_mid_trans = [([t] * n_output) for t in model_output['data'][key]['mid_translation']]
    list_mid_trans = [e for sublist in list_mid_trans for e in sublist]

    all_codebleu_score = []
    all_codebleu_comp = []
    all_bleu = []
    all_crystal = []
    all_crystal_small = []
    all_crystal_big = []
    all_em = []
    for i in range(n_output ** 2):
        codebleu_score, codebleu_comp = compute_codebleu(hypothesis=[model_output['data'][key]['output'][i]],
                                                         references=[[model_output['data'][key]['target']]],
                                                         lang='java')
        em_score = 1 if model_output['data'][key]['target'].split() == model_output['data'][key]['output'][
            i].split() else 0
        bleu_score = round(corpus_bleu(
            [[model_output['data'][key]['target'].split()]], [model_output['data'][key]['output'][i].split()],
            ignoring=None) * 100.0, 2)
        crystal_bleu_score = round(corpus_bleu(
            [[model_output['data'][key]['target'].split()]], [model_output['data'][key]['output'][i].split()],
            ignoring=trivially_shared_ngrams_medium) * 100.0, 2)
        crystal_bleu_score_small = round(corpus_bleu(
            [[model_output['data'][key]['target'].split()]], [model_output['data'][key]['output'][i].split()],
            ignoring=trivially_shared_ngrams_small)*100.0,2)
        crystal_bleu_score_big = round(corpus_bleu(
            [[model_output['data'][key]['target'].split()]], [model_output['data'][key]['output'][i].split()],
            ignoring=trivially_shared_ngrams_big)*100.0,2)
        all_codebleu_score.append(codebleu_score)
        all_codebleu_comp.append(codebleu_comp)
        all_bleu.append(bleu_score)
        all_em.append(em_score)
        all_crystal.append(crystal_bleu_score)
        all_crystal_small.append(crystal_bleu_score_small)
        all_crystal_big.append(crystal_bleu_score_big)

    df = pd.DataFrame({'mid_trans': list_mid_trans, 'raw_output': model_output['data'][key]['raw_output'],
                       'target': model_output['data'][key]['target'],
                       'output': model_output['data'][key]['output'],
                       'compilable': is_compilable, 'plausability': is_plausible, 'test_passed': ratios_passed,
                       'codebleu': all_codebleu_score, 'codebleu_components': all_codebleu_comp,
                       'crystalbleu': all_crystal, 'crystalbleu_small':all_crystal_small,'crystalbleu_big':all_crystal_big, 'bleu': all_bleu,
                       'em_score': all_em})

    mean_compilablity_rate = sum(is_compilable) / len(is_compilable)
    mean_plausability_rate = sum(is_plausible) / len(is_plausible)
    mean_rate_test_passed = sum(ratios_passed) / len(ratios_passed)
    max_rate_test_passed = max(ratios_passed)
    mean_codebleu_score = sum(all_codebleu_score) / len(all_codebleu_score)
    max_codebleu_score = max(all_codebleu_score)
    mean_bleu_score = sum(all_bleu) / len(all_bleu)
    mean_em_score = sum(all_em) / len(all_em)
    mean_crystal_score = sum(all_crystal) / len(all_crystal)
    max_crystal_score = max(all_crystal)
    mean_crystal_score_small = sum(all_crystal_small)/len(all_crystal_small)
    mean_crystal_score_big = sum(all_crystal_big)/len(all_crystal_big)

    run.log(
        {"input": model_output['data'][key]['input'], "model_outputs": df, "problem": problem_name, "model": model_name,
         "mean_compilability_rate": mean_compilablity_rate, "mean_plausability_rate": mean_plausability_rate,
         "mean_rate_test_passed": mean_rate_test_passed, "max_rate_test_passed": max_rate_test_passed,
         "mean_codebleu_score": mean_codebleu_score, "max_codebleu_score": max_codebleu_score,"max_crystal_bleu_score": max_crystal_score ,
         "mean_crystal_bleu_score": mean_crystal_score, "mean_crystal_bleu_score_small": mean_crystal_score_small ,"mean_crystal_bleu_score_big": mean_crystal_score_big,
         "mean_bleu_score": mean_bleu_score,
         "mean_em_score": mean_em_score})
    wandb.finish()

    with lock:
        if os.path.isfile(output_file):
            tmp_save = validated_result['data'][key]['output']
            validated_result = json.load(open(output_file, 'r'))
            validated_result['data'][key] = {}

            for k, value in model_output['data'][key].items():
                if k != 'output':
                    validated_result['data'][key][k] = value
            validated_result['data'][key]['output'] = tmp_save

        json.dump(validated_result, open(output_file, 'w'), indent=2)
        defects4j_command.command_with_timeout(['rm','-rf', tmp_dir])


def custom_error_callback(error):
    print(error, flush=True)

def validate_defects4j_parallel(input_file, output_file, tmp_dir, config_wandb, run_number):
    CUR_DIR = os.getcwd()
    num_problems = len(json.load(open(input_file, 'r'))['data'].keys())
    
    with Manager() as manager:
        lock = manager.Lock()
        items = [(input_file, output_file, tmp_dir, config_wandb, i, lock, run_number) for i in range(num_problems)]
        with Pool(1) as pool:
            pool.starmap(validate_defects4j, items)
    
if __name__ == '__main__':
    wandb.login()

    config = {
        "experiment": "Round-trip C#",
        "num_outputs" : 5,
        "num_beams" : 10,
        "temp" : 1.0,
    }

    total_runs = 1
    for run_number in range(total_runs):
        input_file = f"plbart_result/run_{run_number}/plbart_java_cs_java_output_round_csharp_batch.json"
        output_file = f"plbart_result/run_{run_number}/plbart_java_cs_java_validate_round_csharp_batch.json"
        tmp_dir = "./../../tmp_benchmarks/Defects4J/"

        validate_defects4j_parallel(input_file, output_file, tmp_dir, config_wandb=config, run_number=0)


    


