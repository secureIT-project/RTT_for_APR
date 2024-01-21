InCoderInputConfig = {
    "INCODER_COMPLETE_CODEFORM_NOCOMMENT": {
        "model_id": "incoder-1B/6B",
        "input": "whole buggy function, with the bugggy line masked by <|mask:0|>",
        "patch": "code generated by the model, which will replace the entire buggy function. need extra analysis to figure out where to stop"
    },
    "INCODER_COMPLETE_CODEFORM_COMMENTFORM_NOCOMMENT": {
        "model_id": "incoder-1B/6B",
        "input": "whole buggy function, with the bugggy line masked by <|mask:0|>",
        "patch": "the buggy function before the buggy lines, with buggy lines start with '// buggy line:'. remove all the other commonts and empty lines in the code"
    }
}
def incoder_output_to_patch(output):
    output = output.strip().split('\n')
    no_comment_output = [line.strip() for line in output]
    output = '\n'.join(no_comment_output)
    stack = ['{']
    try:
        start_index = output.index('{')
        patch = output[: start_index + 1]
        for c in output[start_index + 1: ]:
            patch += c
            if c == '}'and patch[-2] != '\'':
                top = stack.pop()
                if top != '{':
                    return ''
                if len(stack) == 0:
                    return patch.strip()
            elif c == '{'and patch[-2] != '\'':
                stack.append(c)
        return ''
    except Exception as e:
        return ''

def extract_fim_part(s: str):
    FIM_PREFIX = "<fim_prefix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_PAD = "<fim_pad>"
    EOD = "<|endoftext|>"
    # Find the index of
    start = s.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    stop = s.find(EOD, start) or len(s)
    return s[start:stop]