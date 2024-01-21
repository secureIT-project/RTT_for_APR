import json
import sys
import os
import shutil
import humaneval_command
import wandb
import pandas as pd
HUMANEVAL_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
from output_to_patch import codet5_output_to_patch, plbart_output_to_patch, incoder_output_to_patch
import time
from evaluation.CodeBLEU.calc_code_bleu import compute_codebleu
import pickle
from crystalbleu import corpus_bleu
from multiprocessing import Lock, current_process, Manager
from multiprocessing.pool import Pool

trivially_shared_ngrams_small = pickle.load(open("ignore_crystal_humanevaljava100.pkl", "rb"))
trivially_shared_ngrams_medium = pickle.load(open("ignore_crystal_humanevaljava.pkl", "rb"))
trivially_shared_ngrams_big = pickle.load(open("ignore_crystal_humanevaljava1000.pkl", "rb"))

def insert_fix(filename, start_line, end_line, patch):
    """
    end_row is included in the buggy lines / buggy function
    """
    with open(filename, 'r') as file:
        data = file.readlines()

    with open(filename, 'w') as file:
        for i in range(start_line - 1):
            file.write(data[i] + '\n')
        file.write(patch.strip())
        for i in range(end_line, len(data)):
            file.write(data[i])


def validate_humaneval(input_file, output_file, tmp_dir, config_wandb, problem_number, lock, run_number):
    plausible, total = 0, 0
    process_number = current_process().pid
    tmp_dir = tmp_dir[:-1] + str(process_number)+ str(round(time.time())) +'/'
    if not os.path.exists(tmp_dir):
        humaneval_command.command_with_timeout(['mkdir', tmp_dir])
    humaneval_command.command_with_timeout(['rm', '-rf', tmp_dir + 'src/main/java/humaneval/buggy/'])
    humaneval_command.command_with_timeout(['mkdir', '-p', tmp_dir + 'src/main/java/humaneval/buggy/'])
    humaneval_command.command_with_timeout(['rm', '-rf', tmp_dir + 'src/test/java/humaneval/'])
    humaneval_command.command_with_timeout(['mkdir', '-p', tmp_dir + 'src/test/java/humaneval/'])

    model_output = json.load(open(input_file, 'r'))
    try:
        validated_result = json.load(open(output_file, 'r'))
    except:
        validated_result = {'config': model_output['config'], 'data': {}}
    proj = list(model_output['data'].keys())[problem_number]
    model_name = model_output['model']
    if proj in validated_result['data']:
        print('Already done: ', proj, flush=True)
        return
    if 'output' not in model_output['data'][proj]:
        print('No output: ', proj, flush=True)
        return
    print('start validating', proj)
    total += 1

    project_name = f'Your_Project'
    run = wandb.init(
        mode="disabled",
        project=project_name,
        config = config_wandb,
        settings=wandb.Settings(_disable_stats=True)
    )


    humaneval_command.command_with_timeout(['rm', '-rf', tmp_dir + 'src/main/java/humaneval/buggy/*.java'])
    humaneval_command.command_with_timeout(['rm', '-rf', tmp_dir + 'src/test/java/humaneval/*.java'])
    shutil.copyfile(tmp_dir + '/../HumanEval/src_org/main/java/humaneval/buggy/' + proj + '.java', tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java')
    shutil.copyfile(tmp_dir + '/../HumanEval/src_org/test/java/humaneval/TEST_' + proj + '.java', tmp_dir + 'src/test/java/humaneval/TEST_' + proj + '.java')
    shutil.copyfile(tmp_dir + '/../HumanEval/pom.xml', tmp_dir + 'pom.xml')
    if not os.path.exists(tmp_dir + "/lib"):
        shutil.copytree(tmp_dir + "/../HumanEval/lib", tmp_dir + "/lib")

    validated_result['data'][proj] = {}
    for key, value in model_output['data'][proj].items():
        if key != 'output':
            validated_result['data'][proj][key] = value
    validated_result['data'][proj]['output'] = []


    start_line, end_line = validated_result['data'][proj]['loc'].split('-')
    end_line = str(int(end_line) - 1) if end_line != start_line else end_line
    function_start_line, function_end_line = validated_result['data'][proj]['function range'].split('-')
    function_start_line, function_end_line = function_start_line.split(',')[0], function_end_line.split(',')[0]

    current_is_correct = False

    is_compilable = []
    is_plausible = []
    ratios_passed = []
    for rank, patch in enumerate(model_output['data'][proj]['output']):
        original_patch = patch
        filename = tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java'
        if 'CODET5' in model_output['config']:
            patch = codet5_output_to_patch(patch, model_output['config'])
            if model_output['config']  in ['CODET5_BASE_CODEFORM_MASKFORM_NOCOMMENT', 'CODET5_BASE_CODEFORM_COMMENTFORM_NOCOMMENT']:
                insert_fix(filename, int(start_line), int(end_line), patch)
            elif model_output['config']  == 'CODET5_REFINE_CODEFORM_NOCOMMENT':
                insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        elif 'CODEGEN' in model_output['config']:
            patch = codegen_output_to_patch(patch, model_output['config'])
            insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        elif 'PLBART' in model_output['config']:
            patch = plbart_output_to_patch(patch, model_output['config'])
            insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        elif 'INCODER' in model_output['config']:
            insert_fix(filename, int(function_start_line), int(function_end_line), patch)
        else:
            assert False, 'unrecognized model.'

        if patch != '':
            compile = humaneval_command.compile_fix(os.path.join(os.getcwd(),filename))
        else:
            compile = False
            patch = ' ' if original_patch == '' else original_patch

        correctness = 'uncompilable'
        if compile:
            correctness, ratio_test_passed = humaneval_command.humaneval_test_suite(proj, tmp_dir)
            ratios_passed.append(ratio_test_passed)
            is_compilable.append(True)

            if correctness == 'plausible':
                is_plausible.append(True)
                if not current_is_correct:
                    plausible += 1
                    current_is_correct = True
                # print(plausible, total, rank, "Plausible patch:", patch)
            elif correctness == 'wrong':
                is_plausible.append(False)
                # print(plausible, total, rank, "Wrong patch:", patch)
            elif correctness == 'timeout':
                is_plausible.append(False)
                # print(plausible, total, rank, "Timeout patch:", patch)
            elif correctness == 'uncompilable with tests':
                is_plausible.append(False)
                # print(plausible, total, rank, "Uncompilable with tests patch:", patch)
        else:
            is_compilable.append(False)
            is_plausible.append(False)
            ratios_passed.append(0)

            # print(plausible, total, rank, "Uncompilable patch:", patch)
        validated_result['data'][proj]['output'].append({
            'patch': patch, 'correctness': correctness
        })
        shutil.copyfile(tmp_dir + '/../HumanEval/src_org/main/java/humaneval/buggy/' + proj + '.java',
                        tmp_dir + 'src/main/java/humaneval/buggy/' + proj + '.java')

    #
    n_output = int(len(is_compilable)**(1/2))
    list_mid_trans = [([t] * n_output) for t in model_output['data'][proj]['mid_translation']]
    list_mid_trans = [e for sublist in list_mid_trans for e in sublist]

    all_codebleu_score = []
    all_codebleu_comp = []
    all_bleu = []
    all_crystal = []
    all_crystal_small = []
    all_crystal_big = []
    all_em = []
    for i in range(n_output**2):
        codebleu_score, codebleu_comp = compute_codebleu(hypothesis=[model_output['data'][proj]['output'][i]],
                                                         references=[[model_output['data'][proj]['target']]],
                                                         lang='java')
        em_score = 1 if model_output['data'][proj]['target'].split() == model_output['data'][proj]['output'][
            i].split() else 0
        bleu_score = round(corpus_bleu(
            [[model_output['data'][proj]['target'].split()]], [model_output['data'][proj]['output'][i].split()],
            ignoring=None)*100.0,2)
        crystal_bleu_score = round(corpus_bleu(
            [[model_output['data'][proj]['target'].split()]], [model_output['data'][proj]['output'][i].split()],
            ignoring=trivially_shared_ngrams_medium)*100.0,2)
        crystal_bleu_score_small = round(corpus_bleu(
            [[model_output['data'][proj]['target'].split()]], [model_output['data'][proj]['output'][i].split()],
            ignoring=trivially_shared_ngrams_small)*100.0,2)
        crystal_bleu_score_big = round(corpus_bleu(
            [[model_output['data'][proj]['target'].split()]], [model_output['data'][proj]['output'][i].split()],
            ignoring=trivially_shared_ngrams_big)*100.0,2)
        all_codebleu_score.append(codebleu_score)
        all_codebleu_comp.append(codebleu_comp)
        all_bleu.append(bleu_score)
        all_em.append(em_score)
        all_crystal.append(crystal_bleu_score)
        all_crystal_small.append(crystal_bleu_score_small)
        all_crystal_big.append(crystal_bleu_score_big)

    df = pd.DataFrame({'mid_trans': list_mid_trans, 'raw_output':model_output['data'][proj]['raw_output'],
                       'target':model_output['data'][proj]['target'], 'output':model_output['data'][proj]['output'],
                       'compilable':is_compilable, 'plausability':is_plausible, 'test_passed':ratios_passed,
                       'codebleu': all_codebleu_score, 'codebleu_components': all_codebleu_comp,
                       'crystalbleu': all_crystal, 'crystalbleu_small':all_crystal_small,'crystalbleu_big':all_crystal_big,'bleu':all_bleu, 'em_score':all_em})


    mean_compilablity_rate = sum(is_compilable)/len(is_compilable)
    mean_plausability_rate = sum(is_plausible)/len(is_plausible)
    mean_rate_test_passed = sum(ratios_passed)/len(ratios_passed)
    max_rate_test_passed = max(ratios_passed)
    mean_codebleu_score = sum(all_codebleu_score)/len(all_codebleu_score)
    max_codebleu_score = max(all_codebleu_score)
    mean_bleu_score = sum(all_bleu)/len(all_bleu)
    mean_em_score = sum(all_em)/len(all_em)
    mean_crystal_score = sum(all_crystal)/len(all_crystal)
    max_crystal_score = max(all_crystal)
    mean_crystal_score_small = sum(all_crystal_small)/len(all_crystal_small)
    mean_crystal_score_big = sum(all_crystal_big)/len(all_crystal_big)

    run.log({"input": model_output['data'][proj]['input'], "model_outputs": df, "problem": proj,"model": model_name, "mean_compilability_rate": mean_compilablity_rate, "mean_plausability_rate":mean_plausability_rate, "mean_rate_test_passed":mean_rate_test_passed, "max_rate_test_passed":max_rate_test_passed,
             "mean_codebleu_score": mean_codebleu_score, "max_codebleu_score":max_codebleu_score, "max_crystal_bleu_score": max_crystal_score ,"mean_crystal_bleu_score": mean_crystal_score ,"mean_crystal_bleu_score_small": mean_crystal_score_small ,"mean_crystal_bleu_score_big": mean_crystal_score_big , "mean_bleu_score":mean_bleu_score, "mean_em_score":mean_em_score})

    wandb.finish()

    with lock:
        if os.path.isfile(output_file):
            tmp_save = validated_result['data'][proj]['output']
            validated_result = json.load(open(output_file, 'r'))
            validated_result['data'][proj] = {}

            for key, value in model_output['data'][proj].items():
                if key != 'output':
                    validated_result['data'][proj][key] = value
            validated_result['data'][proj]['output'] = tmp_save

        json.dump(validated_result, open(output_file, 'w'), indent=2)
        humaneval_command.command_with_timeout(['rm','-rf', tmp_dir])


def custom_error_callback(error):
    print(error, flush=True)

def validate_humaneval_parallel(input_file, output_file, tmp_dir, config_wandb, run_number):
    with Manager() as manager:
        lock = manager.Lock()
        num_problems = 164
        items = [(input_file, output_file, tmp_dir, config_wandb, i, lock, run_number) for i in range(num_problems)]
        with Pool(1) as pool:
            pool.starmap(validate_humaneval, items)



if __name__ == '__main__':
    wandb.login()
    config = {
        "dataset":"HumanEvalJava",
        "experiment": "Round-trip C#",
        "num_outputs" : 5,
        "num_beams" : 10,
        "temp" : 1.0,
    }

    total_runs = 1
    for run_number in range(total_runs):
        input_file = f"plbart_result/run_{run_number}/plbart_java_cs_java_output_round_csharp_batch.json"
        output_file = f"plbart_result/run_{run_number}/plbart_java_cs_java_validate_round_csharp_batch.json"
        tmp_dir = "./../../tmp_benchmarks/HumanEval/"

        validate_humaneval_parallel(input_file, output_file, tmp_dir, config_wandb=config, run_number=run_number)

