import os
import time
import subprocess
import re

def command_with_timeout(cmd, timeout=60, shell=False):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, shell=shell)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    out, err = p.communicate()
    return out, err


def compile_fix(filename):
    FNULL = open(os.devnull, 'w')
    p = subprocess.call(["javac",
                         filename], stderr=FNULL)
    return False if p else True

def humaneval_test_suite(algo, humaneval_dir):
    CUR_DIR = os.getcwd()
    FNULL = open(os.devnull, 'w')
    try:
        os.chdir(humaneval_dir)
        out, err = command_with_timeout(["mvn", "test", "-T","1C","-Dtest=TEST_" + algo.upper()], timeout=30, shell=False)
        os.chdir(CUR_DIR)
        msg = (str(out) + str(err)).upper()
        if "compilation problems".upper() in msg or "compilation failure".upper() in msg:
            return 'uncompilable with tests', 0
        elif "timeout".upper() in msg:
            return 'timeout', 0
        elif "build success".upper() in msg:
            return 'plausible',1
        else:
            try:
                total, failures, errors, *drop = re.findall("\d+", str(out)[str(out).index('Tests run: '):])
            except:
                print("OUT:", str(out))
                print("ERR:", str(err))
            return "wrong", (float(total)-float(failures)-float(errors))/float(total)
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable with tests', 0
